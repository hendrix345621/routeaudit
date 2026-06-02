"""Compose attack mutators onto the shared hook sites, and the prompt-side
RouteHijack suffix helper.

When several mutators target the same router/expert hook they apply in
registered order. `merge_bundles` chains per-hook mutators left-to-right so a
caller can stack a router-side and an expert-side attack on the model under test.
"""
from __future__ import annotations

import torch

from ..eval.generate import DefenseBundle


# ─────────────────────────── Generic composition ───────────────────────────


def compose_router_mutators(*mutators):
    """Chain N router mutators left-to-right. Each is `f(logits, layer, step) -> logits`."""
    mutators = [m for m in mutators if m is not None]
    if not mutators:
        return None
    if len(mutators) == 1:
        return mutators[0]

    def composed(logits: torch.Tensor, layer_idx: int, step_idx: int) -> torch.Tensor:
        for m in mutators:
            logits = m(logits, layer_idx, step_idx)
        return logits

    return composed


def compose_expert_mutators(*mutators):
    """Chain N expert mutators left-to-right. Each: `f(expert_out, layer, expert, step) -> tensor`."""
    mutators = [m for m in mutators if m is not None]
    if not mutators:
        return None
    if len(mutators) == 1:
        return mutators[0]

    def composed(expert_out, layer, expert, step):
        for m in mutators:
            expert_out = m(expert_out, layer, expert, step)
        return expert_out

    return composed


def merge_bundles(*bundles: DefenseBundle) -> DefenseBundle:
    """Merge N DefenseBundles into one, composing per-hook mutators in order."""
    router_chain = []
    expert_chains: dict[tuple[int, int], list] = {}
    for b in bundles:
        if b.router_mutator is not None:
            router_chain.append(b.router_mutator)
        for k, fn in b.expert_mutators.items():
            expert_chains.setdefault(k, []).append(fn)

    out = DefenseBundle()
    out.router_mutator = compose_router_mutators(*router_chain)
    out.expert_mutators = {k: compose_expert_mutators(*chain) for k, chain in expert_chains.items()}
    return out


# ─────────────────────────── Prompt-side helper ───────────────────────────


def apply_routehijack_suffix(prompts: list[str], suffix: str) -> list[str]:
    """Append a previously-derived universal RouteHijack suffix to every prompt."""
    return [f"{p} {suffix}" for p in prompts]
