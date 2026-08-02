"""Prompt-side RouteHijack suffix helper."""
from __future__ import annotations


def apply_routehijack_suffix(prompts: list[str], suffix: str) -> list[str]:
    """Append a previously-derived universal RouteHijack suffix to every prompt."""
    return [f"{p} {suffix}" for p in prompts]
