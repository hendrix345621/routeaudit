"""Forward hooks for MoE models.

Four quantities matter to this pipeline:

  - router_logits  : per-layer, pre-top-k. Site of RouteAudit's suffix search (capture
                     AND mutate — the router mutator is how a defense could steer
                     routing back at eval time).
  - gate_input     : the gate's own input (T, d_model). The quantity mHC does NOT break:
                     under a multi-stream residual, `hc_pre` mixes the n streams down to
                     one d-vector before the gate sees it, so this stays (T, d) on
                     DeepSeek-V4 where the per-layer residual has become (T, n, d).
  - routing        : faithful per-layer routing recomputed from gate_input through a
                     :class:`~routeaudit.model.gate_math.GateSpec`. Needed for gates that
                     never expose a pre-selection score tensor — DeepSeek's Gate returns
                     only `(weights, indices)`, so there is nothing to read off the output.
  - residual       : per-layer decoder-layer output. (T, d_model) on a standard model,
                     (T, n, d_model) under mHC — the stream count is recorded alongside
                     so consumers can call `mhc.reduce_streams` deliberately instead of
                     flattening n streams together by accident.

Which attributes hold the MoE block, router, and experts is described by an
:class:`~routeaudit.model.archspec.ArchSpec` (presets for OLMoE, Mixtral, Qwen, Phi-MoE,
DeepSeek). We attach hooks on:

  - block.<router_attr>      : capture logits / gate input / routing, and optionally
                               mutate the logits.
  - layer (forward post)     : capture the residual-stream output.

We deliberately do not patch internals beyond hooks — keeps the loader pluggable.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from . import gate_math, mhc

# ─────────────────────────── Capture container ───────────────────────────


@dataclass
class HookCapture:
    """Holds activations captured during a forward pass.

    All entries are keyed by layer index. Tensors are detached and kept on the
    model's device unless `to_cpu=True` was set.
    """

    router_logits: dict[int, torch.Tensor] = field(default_factory=dict)
    residual: dict[int, torch.Tensor] = field(default_factory=dict)
    gate_input: dict[int, torch.Tensor] = field(default_factory=dict)
    routing: dict[int, "gate_math.RouteResult"] = field(default_factory=dict)

    #: layer index → (T, top_k) selected expert ids. The lightweight alternative to
    #: `routing` for sweeps over many tokens, where keeping five (T, E) tensors per
    #: layer would cost gigabytes.
    expert_indices: dict[int, torch.Tensor] = field(default_factory=dict)

    #: layer index → number of residual streams in `residual[layer]` (1 = standard
    #: residual, n > 1 = mHC multi-stream). Consumers MUST consult this before
    #: reducing a residual to a single vector; see `mhc.reduce_streams`.
    residual_streams: dict[int, int] = field(default_factory=dict)

    def clear(self) -> None:
        self.router_logits.clear()
        self.residual.clear()
        self.gate_input.clear()
        self.routing.clear()
        self.expert_indices.clear()
        self.residual_streams.clear()


# ─────────────────────────── Mutators ───────────────────────────
#
# A router mutator is a callable that takes the pre-truncation logit tensor
# (T, n_experts) and returns a (possibly modified) tensor.


RouterMutator = Callable[[torch.Tensor, int, int], torch.Tensor]
"""(logits[T, n_experts], layer_idx, step_idx) -> logits"""


# ─────────────────────────── Manager ───────────────────────────


class MoEHookManager:
    """Owns the lifecycle of forward hooks on a MoE model.

    Use as a context manager so hooks are always removed:

        with MoEHookManager(model, spec) as hm:
            hm.capture_router_logits()
            out = model(**batch)
            # hm.capture.router_logits is populated

    `spec` is an :class:`ArchSpec`; when omitted the OLMoE preset is used, which
    reproduces the original hardcoded behavior.
    """

    def __init__(self, model: torch.nn.Module, spec=None):
        from .archspec import ArchSpec
        self.model = model
        self.spec = spec or ArchSpec()
        self.capture = HookCapture()

        # Mutator (None = pass-through).
        self._router_mutator: Optional[RouterMutator] = None

        # Per-call step counter. Bumped externally by the caller (one bump per
        # generated token). Used by mutators that depend on decoding step.
        self.step_idx: int = 0

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._capture_router = False
        self._capture_residual = False
        self._capture_gate_input = False
        self._gate_spec: Optional[gate_math.GateSpec] = None
        self._selection_only = False

    # ── lifecycle ────────────────────────────────────────────────────────

    def __enter__(self) -> "MoEHookManager":
        self._install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ── capture switches (set before entering the context, or before forward) ──

    def capture_router_logits(self) -> "MoEHookManager":
        self._capture_router = True
        return self

    def capture_residual(self) -> "MoEHookManager":
        """Capture each decoder layer's residual-stream output, keyed by layer index in
        `capture.residual`, with the stream count in `capture.residual_streams`.

        Standard models give (B, T, d_model); mHC models give (B, T, n, d_model). The
        tensor is stored as-is — reduce it with `mhc.reduce_streams` rather than
        `.view(T, -1)`, which would silently collapse n streams into one vector."""
        self._capture_residual = True
        return self

    def capture_gate_input(self) -> "MoEHookManager":
        """Capture the router's input (T, d_model) in `capture.gate_input`.

        mHC-safe: the gate always sees a single d-dim vector, even when the residual
        stream it came from is multi-stream."""
        self._capture_gate_input = True
        return self

    def capture_routing(self, gate_spec: "gate_math.GateSpec") -> "MoEHookManager":
        """Capture faithful per-layer routing in `capture.routing` as
        :class:`~routeaudit.model.gate_math.RouteResult`.

        This is the path for any gate whose semantics aren't `softmax(logits)`: the
        routing is recomputed from the gate's input under `gate_spec`, so it works on
        gates that expose no pre-selection score tensor at all. On a spec with
        `router_output="recompute"` (DeepSeek) the logits are formed from the gate's
        weight matrix; otherwise the gate's own output is used.

        Hash-routed layers are NOT captured — their routing comes from a token-id table,
        not from the gate input, so a recomputed score there would be fiction. Use
        `gate_math.hash_route` for those.
        """
        self._gate_spec = gate_spec
        self._selection_only = False
        return self

    def capture_expert_selection(self, gate_spec: "gate_math.GateSpec") -> "MoEHookManager":
        """Capture only WHICH experts fire, per layer, in `capture.expert_indices`.

        The corpus-sweep counterpart to `capture_routing`: activation-frequency harvesting
        only needs the top-k membership, and storing full `RouteResult`s over a
        16x1024-token batch on a 43-layer, 256-expert model would run to gigabytes.

        Use this instead of `topk(router_logits)` on any gate that isn't
        `GateSpec.is_plain_topk` — that shortcut ignores the balancing bias and the group
        mask, and on DeepSeek there is no logit tensor to top-k in the first place.
        """
        self._gate_spec = gate_spec
        self._selection_only = True
        return self

    # ── mutator wiring (defensive routing + steering) ────────────────────

    def set_router_mutator(self, fn: RouterMutator | None) -> None:
        self._router_mutator = fn

    # ── decoding-step bookkeeping ────────────────────────────────────────

    def reset_step(self) -> None:
        self.step_idx = 0

    def advance_step(self) -> None:
        self.step_idx += 1

    # ── install hooks (internal) ─────────────────────────────────────────

    def _iter_moe_blocks(self):
        """Yield (layer_idx, moe_block) using the ArchSpec to locate modules."""
        s = self.spec
        base = getattr(self.model, s.base_attr, self.model)
        layers = getattr(base, s.layers_attr)
        for i, layer in enumerate(layers):
            block = None
            for attr in s.moe_block_attrs:   # first existing attr wins (version-robust)
                block = getattr(layer, attr, None)
                if block is not None:
                    break
            if block is None:
                continue
            # MoE blocks expose the router + experts containers named by the spec.
            if hasattr(block, s.experts_attr) and hasattr(block, s.router_attr):
                yield i, block

    def _install(self) -> None:
        # All hooks are installed unconditionally; the capture flags gate only
        # whether data is *stored* inside each hook. This lets callers flip a
        # `capture_*()` switch either before or after entering the context (the
        # documented pattern — see the class docstring). Installing conditionally
        # here is a footgun: a `capture_residual()` called after `__enter__` would
        # otherwise never install its hook and silently capture nothing.
        for layer_idx, block in self._iter_moe_blocks():
            self._install_router_hook(layer_idx, block)
        self._install_residual_hooks()

    def _install_residual_hooks(self) -> None:
        """Hook each decoder layer's forward to capture its residual-stream output.

        HF decoder layers return either a tensor or a tuple whose first element is the
        hidden state. That is (B, T, d_model) on a standard model and (B, T, n, d_model)
        under mHC. We store it undamaged and record the stream count next to it — the
        alternative, flattening to (T, -1) here, is exactly the bug that makes an mHC
        norm profile meaningless (it norms all n streams together).
        """
        s = self.spec
        base = getattr(self.model, s.base_attr, self.model)
        layers = getattr(base, s.layers_attr)
        mgr = self

        for i, layer in enumerate(layers):
            def make_hook(li=i):
                def fwd_hook(_module, _inputs, output):
                    if mgr._capture_residual:
                        tensor = output[0] if isinstance(output, tuple) else output
                        mgr.capture.residual[li] = tensor.detach()
                        mgr.capture.residual_streams[li] = mgr._stream_count(tensor)
                    return output
                return fwd_hook

            self._handles.append(layer.register_forward_hook(make_hook()))

    def _stream_count(self, tensor: torch.Tensor) -> int:
        """Streams in a captured residual. Uses `spec.d_model` when the config supplies
        it; falls back to rank (a 4-D hidden state is multi-stream) when it doesn't."""
        if self.spec.d_model:
            return mhc.stream_count(tensor, self.spec.d_model)
        return int(tensor.shape[-2]) if tensor.dim() == 4 else 1

    def _gate_bias(self, module: torch.nn.Module, recompute: bool) -> Optional[torch.Tensor]:
        """The load-balancing bias, if this gate has one.

        Only tried on gates we recompute, or when the GateSpec explicitly asked for a
        bias: on a plain `nn.Linear` gate, `.bias` is the linear layer's own additive
        bias (already inside the logits) and adding it again would corrupt selection.
        On DeepSeek's `Gate`, `.bias` *is* the auxiliary-loss-free balancing bias, which
        is why the fallback exists at all.
        """
        names = [self.spec.router_bias_attr, "e_score_correction_bias"]
        if recompute:
            names.append("bias")
        for attr in names:
            b = getattr(module, attr, None)
            if isinstance(b, torch.Tensor):
                return b
        return None

    def _logits_from_output(self, output, n_experts: int) -> Optional[torch.Tensor]:
        """Find the `(T, n_experts)` router logits in whatever the gate returned.

        Gate return shapes differ by family and by transformers version:
          * a bare `(T, E)` tensor                       — OLMoE / Mixtral / Qwen / Phi
          * `(scores, topk_weights, topk_indices)`       — fused HF gates
          * `(logits, weights, indices)`                 — `DeepseekV4TopKRouter`
          * `(weights, indices)`                         — DeepSeek's raw inference impl,
                                                           which exposes no logits at all

        Rather than guess from position, take the first element whose trailing dimension
        is `n_experts`: `weights` and `indices` are `(T, top_k)`, so they can't be
        mistaken for logits unless top_k == n_experts (which would mean no sparsity).
        Returns None when nothing matches, so the caller can fall back.
        """
        cands = output if isinstance(output, tuple) else (output,)
        for t in cands:
            if isinstance(t, torch.Tensor) and t.dim() >= 2 and t.shape[-1] == n_experts:
                return t.detach()
        return None

    def _record_routing(self, layer_idx: int, module: torch.nn.Module,
                        hidden: torch.Tensor, output) -> None:
        """Recompute this layer's routing under the GateSpec and store the RouteResult."""
        gs = self._gate_spec
        if gate_math.routing_kind(layer_idx, gs) != gate_math.LEARNED:
            return   # hash/dense layers don't route on content — nothing to recompute
        h = hidden.reshape(-1, hidden.shape[-1]) if hidden.dim() == 3 else hidden
        recompute = self.spec.router_output == "recompute"
        n_experts = self.spec.n_experts
        logits = self._logits_from_output(output, n_experts)
        if logits is None and recompute:
            # No usable logit tensor in the output — form them from the gate's weight
            # matrix. This is the fallback for DeepSeek's raw `inference/model.py`, whose
            # Gate returns only (weights, indices). Note it will NOT work against fp8
            # weights, which is why the output is preferred whenever it carries logits.
            w = getattr(module, "weight", None)
            if not isinstance(w, torch.Tensor):
                return
            logits = torch.nn.functional.linear(h.detach().to(w.dtype), w)
        if logits is None:
            return
        bias = self._gate_bias(module, recompute) if gs.use_bias else None
        if self._selection_only:
            scores = gate_math.affinity(logits.float(), gs.scoring_func)
            sel = gate_math.selection_scores(scores, bias, gs)
            self.capture.expert_indices[layer_idx] = sel.topk(gs.top_k, dim=-1).indices
        else:
            self.capture.routing[layer_idx] = gate_math.route(logits, bias, gs)

    def _install_router_hook(self, layer_idx: int, block: torch.nn.Module) -> None:
        """Hook the router (`block.<router_attr>`) forward to capture and optionally
        mutate router logits.

        The gate is a Linear: input (B*T, d_model) -> logits (B*T, n_experts).
        We mutate the *output* before it leaves the gate, so the downstream top-k
        and softmax see our biased logits.
        """
        gate = getattr(block, self.spec.router_attr)
        mgr = self

        def fwd_hook(module, inputs, output):
            # ── mHC-safe captures, independent of what the gate returns ──
            if (mgr._capture_gate_input or mgr._gate_spec is not None) and inputs:
                h = inputs[0]
                if isinstance(h, torch.Tensor):
                    if mgr._capture_gate_input:
                        mgr.capture.gate_input[layer_idx] = h.detach()
                    if mgr._gate_spec is not None:
                        mgr._record_routing(layer_idx, module, h, output)

            # OLMoE gate output shape depends on transformers version:
            #   - Old (≤ ~4.46): the gate Linear returns raw logits (B*T, n_experts).
            #   - New: a fused gate returns (routing_scores, top_k_weights, top_k_index)
            #     where routing_scores is (B*T, n_experts) — either raw logits or
            #     softmax probs depending on the version. We treat it as routing
            #     scores and topk it ourselves when a mutator is present.
            if isinstance(output, tuple):
                scores = output[0]
                tail = output[1:]
                if mgr._capture_router:
                    mgr.capture.router_logits[layer_idx] = scores.detach()
                if mgr._router_mutator is not None:
                    scores = mgr._router_mutator(scores, layer_idx, mgr.step_idx)
                    # Recompute top-k on the biased scores so the MoE dispatch
                    # actually uses our routing change.
                    if len(tail) >= 2:
                        k = tail[0].shape[-1]   # top_k from old top_k_weights
                        probs = scores.softmax(dim=-1)
                        new_w, new_idx = torch.topk(probs, k=k, dim=-1)
                        new_w = new_w / new_w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
                        new_w = new_w.to(tail[0].dtype)
                        new_idx = new_idx.to(tail[1].dtype)
                        return (scores, new_w, new_idx) + tuple(tail[2:])
                    return (scores,) + tuple(tail)
                return output
            # Legacy path: gate emits raw logits directly.
            logits = output
            if mgr._capture_router:
                mgr.capture.router_logits[layer_idx] = logits.detach()
            if mgr._router_mutator is not None:
                logits = mgr._router_mutator(logits, layer_idx, mgr.step_idx)
            return logits

        self._handles.append(gate.register_forward_hook(fwd_hook))


# ─────────────────────────── Convenience ───────────────────────────


@contextmanager
def captured_forward(model, *, router=False, spec=None):
    """One-shot context for read-only router-logit capture.

        with captured_forward(model, router=True) as cap:
            model(**batch)
        cap.router_logits[2]   # tensor
    """
    hm = MoEHookManager(model, spec)
    if router:
        hm.capture_router_logits()
    with hm:
        yield hm.capture
