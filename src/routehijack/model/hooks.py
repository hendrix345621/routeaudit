"""Forward hooks for MoE models.

Three quantities matter:

  - router_logits  : per-layer, pre-top-k. Site of the RouteHijack attack.
  - expert_out     : per-layer, per-expert, post-MLP, pre weighted sum. Site of SAE
                     training and the SAE-inversion attack (subtract Δh).
  - moe_out        : per-layer residual contribution (sum of weighted experts). Read-only.

Which attributes hold the MoE block, router, and experts is described by an
:class:`~routehijack.model.archspec.ArchSpec` (presets for OLMoE and Mixtral).
We attach hooks on:

  - block.<router_attr>      : capture pre-truncation logits AND optionally mutate them.
  - block.<experts_attr>[i]  : capture/mutate per-expert MLP output.
  - block (forward post)     : capture final moe_out for diagnostics.

We deliberately do not patch internals beyond hooks — keeps the loader pluggable.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


# ─────────────────────────── Capture container ───────────────────────────


@dataclass
class HookCapture:
    """Holds activations captured during a forward pass.

    All entries are keyed by layer index; expert-level entries by (layer, expert).
    Tensors are detached and kept on the model's device unless `to_cpu=True` was set.
    """

    router_logits: dict[int, torch.Tensor] = field(default_factory=dict)
    expert_out: dict[tuple[int, int], torch.Tensor] = field(default_factory=dict)
    moe_out: dict[int, torch.Tensor] = field(default_factory=dict)
    residual: dict[int, torch.Tensor] = field(default_factory=dict)

    def clear(self) -> None:
        self.router_logits.clear()
        self.expert_out.clear()
        self.moe_out.clear()
        self.residual.clear()


# ─────────────────────────── Mutators ───────────────────────────
#
# Each is a callable that takes a tensor and returns a (possibly modified) tensor.
# The router mutator runs on the *pre-truncation* logit tensor of shape (T, n_experts).


RouterMutator = Callable[[torch.Tensor, int, int], torch.Tensor]
"""(logits[T, n_experts], layer_idx, step_idx) -> logits"""

ExpertMutator = Callable[[torch.Tensor, int, int, int], torch.Tensor]
"""(expert_out[T, d_model], layer_idx, expert_idx, step_idx) -> expert_out"""


# ─────────────────────────── Manager ───────────────────────────


class MoEHookManager:
    """Owns the lifecycle of forward hooks on a MoE model.

    Use as a context manager so hooks are always removed:

        with MoEHookManager(model, spec) as hm:
            hm.capture_router_logits()
            hm.capture_expert_out(targets=[(2, 17), (5, 9)])
            out = model(**batch)
            # hm.capture.router_logits, hm.capture.expert_out are populated

    `spec` is an :class:`ArchSpec`; when omitted the OLMoE preset is used, which
    reproduces the original hardcoded behavior.
    """

    def __init__(self, model: torch.nn.Module, spec=None):
        from .archspec import ArchSpec
        self.model = model
        self.spec = spec or ArchSpec()
        self.capture = HookCapture()

        # Mutators (None = pass-through).
        self._router_mutator: Optional[RouterMutator] = None
        self._expert_mutators: dict[tuple[int, int], ExpertMutator] = {}
        self._residual_mutator: Optional[Callable] = None
        self._moe_out_mutators: dict[int, Callable] = {}

        # Per-call step counter. Bumped externally by the caller (one bump per
        # generated token). Used by mutators that depend on decoding step.
        self.step_idx: int = 0

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._capture_router = False
        self._capture_expert_targets: Optional[set[tuple[int, int]]] = None
        self._capture_moe = False
        self._capture_residual = False

    # ── lifecycle ────────────────────────────────────────────────────────

    def __enter__(self) -> "OLMoEHookManager":
        self._install()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # ── capture switches (set before entering the context, or before forward) ──

    def capture_router_logits(self) -> "OLMoEHookManager":
        self._capture_router = True
        return self

    def capture_expert_out(self, targets: list[tuple[int, int]] | None = None) -> "OLMoEHookManager":
        """If targets=None, capture all experts (heavy)."""
        self._capture_expert_targets = set(targets) if targets is not None else None
        self._warn_fused_if_needed()
        return self

    def capture_moe_out(self) -> "OLMoEHookManager":
        self._capture_moe = True
        return self

    def capture_residual(self) -> "OLMoEHookManager":
        """Capture each decoder layer's residual-stream output (T, d_model),
        keyed by layer index in `capture.residual`. Generic residual-space probe."""
        self._capture_residual = True
        return self

    # ── mutator wiring (defensive routing + steering) ────────────────────

    def set_router_mutator(self, fn: RouterMutator | None) -> None:
        self._router_mutator = fn

    def set_expert_mutator(self, layer: int, expert: int, fn: ExpertMutator | None) -> None:
        if fn is None:
            self._expert_mutators.pop((layer, expert), None)
        else:
            self._expert_mutators[(layer, expert)] = fn
            self._warn_fused_if_needed()

    def set_residual_mutator(self, fn: Optional[Callable]) -> None:
        """Install a residual-stream mutator `fn(hidden[T, d_model], layer_idx, step_idx) -> hidden`,
        applied to every decoder layer's output. Generic residual-stream steering hook."""
        self._residual_mutator = fn

    def set_moe_out_mutator(self, layer: int, fn: Optional[Callable]) -> None:
        """Install a mutator `fn(moe_out[...,d_model], layer_idx, step_idx) -> moe_out` on the
        MoE block's OUTPUT at `layer` (the weighted sum of experts that feeds the residual).
        Used by the `moe_out`-tap SAE ablation — a much stronger lever than a single expert's
        output because it edits the whole MoE contribution to the residual stream."""
        if fn is None:
            self._moe_out_mutators.pop(layer, None)
        else:
            self._moe_out_mutators[layer] = fn

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
        # documented pattern — see the class docstring), matching how the router
        # and per-expert hooks already behave. Installing conditionally here is a
        # footgun: a `capture_residual()` / `capture_moe_out()` called after
        # `__enter__` would otherwise never install its hook and silently capture
        # nothing.
        for layer_idx, block in self._iter_moe_blocks():
            self._install_router_hook(layer_idx, block)
            self._install_expert_hooks(layer_idx, block)
            self._install_moe_out_hook(layer_idx, block)
        self._install_residual_hooks()

    def _install_residual_hooks(self) -> None:
        """Hook each decoder layer's forward to capture and optionally mutate its
        residual-stream output.

        HF decoder layers return either a tensor or a tuple whose first element is
        the hidden state (T, d_model). When `_capture_residual` is set we store the
        full (T, d_model) detached (selecting the last token is left to the caller).
        When a residual mutator is installed (`set_residual_mutator`) we apply it and
        feed the modified hidden state back into the network.
        """
        s = self.spec
        base = getattr(self.model, s.base_attr, self.model)
        layers = getattr(base, s.layers_attr)
        mgr = self

        for i, layer in enumerate(layers):
            def make_hook(li=i):
                def fwd_hook(_module, _inputs, output):
                    is_tuple = isinstance(output, tuple)
                    tensor = output[0] if is_tuple else output
                    if mgr._capture_residual:
                        mgr.capture.residual[li] = tensor.detach()
                    if mgr._residual_mutator is not None:
                        tensor = mgr._residual_mutator(tensor, li, mgr.step_idx)
                        if is_tuple:
                            return (tensor,) + tuple(output[1:])
                        return tensor
                    return output
                return fwd_hook

            self._handles.append(layer.register_forward_hook(make_hook()))

    def _install_router_hook(self, layer_idx: int, block: torch.nn.Module) -> None:
        """Hook the router (`block.<router_attr>`) forward to capture and optionally
        mutate router logits.

        The gate is a Linear: input (B*T, d_model) -> logits (B*T, n_experts).
        We mutate the *output* before it leaves the gate, so the downstream top-k
        and softmax see our biased logits.
        """
        gate = getattr(block, self.spec.router_attr)
        mgr = self

        def fwd_hook(_module, _inputs, output):
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

    def _install_expert_hooks(self, layer_idx: int, block: torch.nn.Module) -> None:
        """Hook every expert in this layer. Tolerates two HF transformers layouts:

          - `block.<experts_attr>: nn.ModuleList`  — older HF (OLMoE/Mixtral ≤ ~4.46).
            Iterable; we attach one forward hook per expert. Per-expert capture /
            mutation works.
          - fused experts module (e.g. `OlmoeExperts` / `MixtralExperts`, newer HF).
            No per-expert modules to hook; we mark the manager as fused and warn
            lazily when the caller actually requests per-expert work."""
        experts = getattr(block, self.spec.experts_attr)
        # Fused experts (one module holding all experts as stacked params) have no
        # per-expert submodules to hook. Detect: not a ModuleList, or a ModuleList
        # whose length doesn't match the configured n_experts.
        is_module_list = isinstance(experts, torch.nn.ModuleList)
        if not is_module_list or (self.spec.n_experts and len(experts) != self.spec.n_experts):
            self._has_fused_experts = True
            self._fused_experts_class = type(experts).__name__
            return
        module_list = list(enumerate(experts))

        mgr = self
        for expert_idx, expert in module_list:

            def make_hook(li=layer_idx, ei=expert_idx):
                def fwd_hook(_module, _inputs, output):
                    # output: (n_tokens_routed_to_this_expert, d_model)
                    capture_set = mgr._capture_expert_targets
                    if capture_set is None or (li, ei) in capture_set:
                        mgr.capture.expert_out[(li, ei)] = output.detach()
                    fn = mgr._expert_mutators.get((li, ei))
                    if fn is not None:
                        output = fn(output, li, ei, mgr.step_idx)
                    return output

                return fwd_hook

            self._handles.append(expert.register_forward_hook(make_hook()))

    def _warn_fused_if_needed(self) -> None:
        """Emit the downgrade-transformers warning the first time someone asks
        for per-expert capture or mutation on a fused-experts model."""
        if not getattr(self, "_has_fused_experts", False):
            return
        if getattr(self, "_warned_fused", False):
            return
        import warnings
        cls = getattr(self, "_fused_experts_class", "fused experts")
        warnings.warn(
            f"MoE experts are fused ({cls}) on this transformers version — "
            "per-expert capture / mutation is unavailable, so the SAE pipeline "
            "(scripts 06-09) and the SAE-inversion attack won't function. "
            "RouteHijack (router-logit capture) is unaffected. To restore "
            "per-expert hooks, pin a version that exposes experts as an "
            "`nn.ModuleList`:\n\n"
            "    pip install \"transformers>=4.45,<4.47\" --force-reinstall\n",
            RuntimeWarning,
            stacklevel=2,
        )
        self._warned_fused = True

    def _install_moe_out_hook(self, layer_idx: int, block: torch.nn.Module) -> None:
        mgr = self

        def fwd_hook(_module, _inputs, output):
            # OLMoE's block returns (hidden, router_logits) in some versions; take hidden.
            is_tuple = isinstance(output, tuple)
            tensor = output[0] if is_tuple else output
            if mgr._capture_moe:
                mgr.capture.moe_out[layer_idx] = tensor.detach()
            fn = mgr._moe_out_mutators.get(layer_idx)
            if fn is not None:
                tensor = fn(tensor, layer_idx, mgr.step_idx)
                return (tensor,) + tuple(output[1:]) if is_tuple else tensor
            return output

        self._handles.append(block.register_forward_hook(fwd_hook))


# ─────────────────────────── Convenience ───────────────────────────


# Backward-compat alias: the manager used to be OLMoE-specific.
OLMoEHookManager = MoEHookManager


@contextmanager
def captured_forward(model, *, router=False, experts=None, moe=False, spec=None):
    """One-shot context for read-only capture.

        with captured_forward(model, router=True, experts=[(2, 17)]) as cap:
            model(**batch)
        cap.router_logits[2]   # tensor
        cap.expert_out[(2,17)] # tensor
    """
    hm = MoEHookManager(model, spec)
    if router:
        hm.capture_router_logits()
    if experts is not None:
        hm.capture_expert_out(targets=experts)
    if moe:
        hm.capture_moe_out()
    with hm:
        yield hm.capture
