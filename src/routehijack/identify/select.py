"""Top-fraction selection over Score_safe.

RouteHijack §5 Table 10 (p. 11) uses top-20% of (layer, expert) pairs by score.
We replicate that as the default but keep `top_pct` configurable.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class SafetyExpert:
    layer: int
    expert: int
    score: float


def select_safety_experts(
    score: torch.Tensor,
    *,
    top_pct: float = 0.20,
) -> list[SafetyExpert]:
    """Pick the top `top_pct` fraction of layer-expert pairs globally.

    Returns a list sorted by descending score.
    """
    L, E = score.shape
    flat = score.reshape(-1)
    k = max(1, int(L * E * top_pct))
    vals, idx = flat.topk(k)
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        layer = i // E
        expert = i % E
        out.append(SafetyExpert(layer=layer, expert=expert, score=float(v)))
    return out


# Harmful-side identification reuses the same selection mechanic — the only
# difference is which score tensor you pass in. RouteHijack p. 4: Score_harm
# omits the utility penalty so chosen harmful experts stay fluent.
select_harmful_experts = select_safety_experts


def next_tier(
    score_safe: torch.Tensor,         # (L, E)
    *,
    top_pct: float = 0.20,
    next_pct: float = 0.20,
) -> list[SafetyExpert]:
    """Experts ranked in the band [top_pct, top_pct + next_pct] by Score_safe.

    The "shadow safety experts" — high enough to carry residual refusal behavior
    but excluded from the top-pct set. Used by the coverage sweep to test whether
    refusal features hide outside the RouteHijack-flagged experts (the hydra risk).
    """
    L, E = score_safe.shape
    total = L * E
    top_n = int(top_pct * total)
    next_n = int(next_pct * total)
    flat = score_safe.reshape(-1)
    vals, idx = flat.topk(top_n + next_n)
    out = []
    for v, i in zip(vals.tolist()[top_n : top_n + next_n], idx.tolist()[top_n : top_n + next_n]):
        out.append(SafetyExpert(layer=i // E, expert=i % E, score=float(v)))
    return out


def _elbow_index(values_desc: list[float]) -> int:
    """Kneedle-style elbow on a descending sequence: the count (1-based) at the
    point of maximum distance to the chord joining the first and last points.

    Statistically picks "how many experts are worth it" from the Score_safe curve
    instead of a fixed fraction — the elbow is where marginal score gain collapses.
    """
    n = len(values_desc)
    if n <= 2:
        return n
    x0, y0 = 0.0, float(values_desc[0])
    x1, y1 = float(n - 1), float(values_desc[-1])
    dx, dy = x1 - x0, y1 - y0
    denom = (dx * dx + dy * dy) ** 0.5 or 1.0
    best_i, best_d = 0, -1.0
    for i, v in enumerate(values_desc):
        # perpendicular distance from point (i, v) to the chord
        d = abs(dy * i - dx * (float(v) - y0)) / denom
        if d > best_d:
            best_d, best_i = d, i
    return best_i + 1


def select_sae_targets(
    experts: list[SafetyExpert],
    *,
    freq_safe: "torch.Tensor | None" = None,
    min_freq: float = 0.0,
    max_n: int | None = None,
    auto: bool = True,
) -> list[SafetyExpert]:
    """Choose which experts are worth training an SAE on (a subset of `experts`,
    which are already the top-pct safety experts sorted by Score_safe).

    Two statistical criteria:
      1. **Data sufficiency** — drop experts whose safe-side routing frequency
         F_safe(l,e) < `min_freq`. A rarely-routed expert yields too few cached
         tokens to fit a usable SAE, so its SAE is noise regardless of its score.
      2. **Auto-N via elbow** — when `auto`, cut the sorted-Score_safe list at the
         elbow (diminishing marginal score) instead of a fixed count. `max_n` caps it.
    """
    cand = list(experts)  # already sorted by descending score
    if freq_safe is not None and min_freq > 0:
        cand = [e for e in cand if float(freq_safe[e.layer, e.expert]) >= min_freq]
    if not cand:
        return []
    n = _elbow_index([e.score for e in cand]) if auto else len(cand)
    if max_n is not None:
        n = min(n, max_n)
    return cand[: max(1, n)]


def load_sae_targets(
    safety_path: str | Path,
    *,
    diagnostics_path: str | Path | None = None,
    top_n: int = 3,
    select: str = "topn",
    min_freq: float = 0.0,
) -> list[SafetyExpert]:
    """Resolve the SAE-training target experts (shared by scripts 03/04/05).

    `select="topn"` → first `top_n` safety experts (default, backward-compatible).
    `select="auto"` → frequency-floor + Score_safe elbow, capped at `top_n`
    (needs `diagnostics_path` for F_safe; falls back to score-only if absent).
    """
    experts = load_experts(safety_path)
    if select != "auto":
        return experts[:top_n]
    freq = None
    if diagnostics_path is not None and Path(diagnostics_path).exists():
        d = torch.load(diagnostics_path, map_location="cpu")
        freq = d.get("F_safe")
    return select_sae_targets(experts, freq_safe=freq, min_freq=min_freq, max_n=top_n, auto=True)


def save_experts(experts: list[SafetyExpert], path: str | Path) -> None:
    payload = [{"layer": e.layer, "expert": e.expert, "score": e.score} for e in experts]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load_experts(path: str | Path) -> list[SafetyExpert]:
    with open(path, "r", encoding="utf-8") as fh:
        return [SafetyExpert(**row) for row in json.load(fh)]
