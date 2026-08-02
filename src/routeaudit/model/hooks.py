"""Forward hooks for MoE models.

Two quantities matter to this pipeline:

  - router_logits  : per-layer, pre-top-k. Site of RouteAudit's suffix search (capture
                     AND mutate — the router mutator is how a defense could steer
                     routing back at eval time).
  - residual       : per-layer decoder-layer output (T, d_model). Read-only capture,
                     used by the mHC diagnostic (experiments/mhc/) to reach the MoE gate's
                     C-dim input on models whose residual stream isn't plain (T, C).

Which attributes hold the MoE block, router, and experts is described by an
:class:`~routeaudit.model.archspec.ArchSpec` (presets for OLMoE, Mixtral, Qwen,
Phi-MoE). We attach hooks on:

  - block.<router_attr>      : capture pre-truncation logits AND optionally mutate them.
  - block (forward post)     : capture final moe_out for diagnostics — NOT wired up
                                (no capture switch reads it); see note below.

We deliberately do not patch internals beyond hooks — keeps the loader pluggable.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


# ─────────────────────────── Capture container ───────────────────────────


@dataclass
class HookCapture:
    """Holds activations captured during a forward pass.

    All entries are keyed by layer index. Tensors are detached and kept on the
    model's device unless `to_cpu=True` was set.
    """

    router_logits: dict[int, torch.Tensor] = field(default_factory=dict)
    residual: dict[int, torch.Tensor] = field(default_factory=dict)

    def clear(self) -> None:
        self.router_logits.clear()
        self.residual.clear()


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
        """Capture each decoder layer's residual-stream output (T, d_model),
        keyed by layer index in `capture.residual`. Generic residual-space probe."""
        self._capture_residual = True
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

        HF decoder layers return either a tensor or a tuple whose first element is
        the hidden state (T, d_model). When `_capture_residual` is set we store the
        full (T, d_model) detached (selecting the last token is left to the caller).
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
