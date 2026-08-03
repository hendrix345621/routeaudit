"""Validate our gate/mHC reimplementation against fixtures from the released model.

Level 1 of the validation ladder (see `extract.py`). This is the gate on plan.md's Phase
P0: "corrected diagnostic reproduces the released Gate's (weights, indices) bit-for-bit
on a saved tensor fixture."

Status: **PENDING  --  requires DeepSeek-V4-Flash access.** With no fixture file present
this exits cleanly saying so, rather than passing vacuously. A green run here is the only
thing that turns "we implemented the paper's equations" into "we reproduce the shipped
model", and until it happens the P0 gate is unmet  --  say so in any writeup.

    python experiments/mhc/fixtures/validate.py
    python experiments/mhc/fixtures/validate.py --fixtures path/to/v4_flash_fixtures.pt
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# Run from a fresh clone without `pip install -e .`; an installed package wins.
if importlib.util.find_spec("routeaudit") is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

import torch

from routeaudit import ui
from routeaudit.model import gate_math, mhc
from routeaudit.model.gate_math import GateSpec

DEFAULT_FIXTURES = Path("experiments/mhc/fixtures/v4_flash_fixtures.pt")


def _cmp(name: str, got: torch.Tensor, want: torch.Tensor, atol: float) -> tuple[bool, str]:
    if got.shape != want.shape:
        return False, f"{name}: shape {tuple(got.shape)} != {tuple(want.shape)}"
    dev = float((got.float() - want.float()).abs().max())
    ok = torch.equal(got, want) if atol == 0 else dev <= atol
    return ok, f"{name}: max|Δ| = {dev:.3e} ({'exact' if dev == 0 else f'atol={atol:g}'})"


def validate_gate(fx: dict, atol: float) -> list[tuple[bool, str]]:
    """Recompute routing from the fixture's gate input and compare to the shipped I/O.

    Selection is compared as a SET: two implementations can order tied scores differently
    without disagreeing about which experts fire. The weights are then aligned to our
    index order before comparison, so an ordering difference can't masquerade as a
    numerical one.
    """
    g = fx.get("gate")
    if not g:
        return [(False, "no 'gate' fixture in file")]
    gs = GateSpec(**fx["meta"]["gate_spec"])
    scores = g["scores"]
    bias = (g["sel_scores"] - scores)[0] if gs.use_bias else None

    # Re-derive selection + weighting from the scores the fixture recorded.
    sel = gate_math.selection_scores(scores, bias, gs)
    idx = sel.topk(gs.top_k, dim=-1).indices
    w = gate_math.gate_weights(scores, idx, gs)

    out = [_cmp("sel_scores", sel, g["sel_scores"], atol)]
    same_set = torch.equal(idx.sort(-1).values, g["indices"].sort(-1).values)
    out.append((same_set, f"indices: {'same expert set' if same_set else 'DIFFERENT experts fire'}"))
    if same_set:
        order, order_ref = idx.argsort(-1), g["indices"].argsort(-1)
        ref = torch.empty_like(w)
        ref.scatter_(-1, order, torch.gather(g["weights"], -1, order_ref))
        out.append(_cmp("weights", w, ref, atol))
    return out


def validate_residual(fx: dict) -> list[tuple[bool, str]]:
    """Check the captured residual is shaped the way the config claims, and that the
    documented reduction applies to it. A stream-count mismatch means every residual-space
    number from that model was computed on the wrong tensor."""
    r = fx.get("residual")
    if not r:
        return [(False, "no residual fixture")]
    n = int(r["n_streams"])
    h = r["hidden"]
    expected = 4
    ok = h.dim() == 4 and n == expected and h.shape[-2] == expected
    lines = [
        (
            ok,
            (
                f"residual streams: recorded n={n}, tensor rank {h.dim()} "
                f"{'consistent' if ok else 'INCONSISTENT'} (expected {expected})"
            ),
        )
    ]
    if n > 1:
        red = mhc.reduce_residual(h, n, "mean")
        lines.append(
            (
                red.shape[-1] == h.shape[-1] and red.dim() == h.dim() - 1,
                f"stream-mean reduction: {tuple(h.shape)} → {tuple(red.shape)}",
            )
        )
    return lines


def validate_hash(fx: dict) -> list[tuple[bool, str]]:
    """The free oracle: hash routing must reproduce the static table exactly."""
    h = fx.get("hash")
    if not h:
        return [(False, "no hash fixture")]
    table = h["table"]
    ids = torch.arange(min(256, table.shape[0]))
    ok = torch.equal(gate_math.hash_route(ids, table), table[ids])
    return [
        (
            ok,
            (
                f"hash routing reproduces the token-id table for {len(ids)} ids "
                f"({'exact' if ok else 'MISMATCH'})"
            ),
        )
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    ap.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="0 = bitwise (same dtype + kernel stack). Relax to fp8 "
        "resolution when comparing an fp32 reimplementation to fp8 truth.",
    )
    args = ap.parse_args()

    if not args.fixtures.exists():
        ui.warn(
            f"no fixtures at {args.fixtures} — Level 1 validation is PENDING.\n"
            f"  It needs access to the released checkpoint. Generate with:\n"
            f"    python experiments/mhc/fixtures/extract.py --config deepseek-v4-flash\n"
            f"  Until then plan.md's P0 gate ('reproduces the released Gate bit-for-bit') "
            f"is UNMET — report it as unmet rather than assuming it."
        )
        raise SystemExit(0)

    fx = torch.load(args.fixtures, map_location="cpu", weights_only=False)
    ui.section(f"validating against {fx['meta']['hf_id']}")

    results = validate_gate(fx, args.atol) + validate_residual(fx) + validate_hash(fx)
    for ok, line in results:
        (ui.ok if ok else ui.fail)(line)

    if all(ok for ok, _ in results):
        ui.print_done("Level 1 PASSED — the reimplementation matches the shipped model.")
    else:
        ui.fail("Level 1 FAILED — fix the component above before trusting any diagnostic.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
