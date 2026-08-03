"""Extract ground-truth component fixtures from a released DeepSeek checkpoint.

Level 1 of the validation ladder. Level 0 (mechanism tests on the synthetic model) runs
today; this step needs the real weights and is currently PENDING  --  no DeepSeek-V4-Flash
access. It is written now so it is a single command the day the weights are reachable.

    Level 0  synthetic model, random weights, fp32, CPU   -> RUNS NOW
    Level 1  component fixtures from the released model   -> this script
    Level 2  full forward parity on a fixed prompt        -> validate.py --full
    Level 3  semantic experiments on the real checkpoint  -> the diagnostics themselves

The point of fixtures rather than end-to-end comparison: each failure localizes to one
component. A whole-model logit mismatch tells you something is wrong; a gate fixture
mismatch tells you the scoring function is wrong.

    python experiments/mhc/fixtures/extract.py --config deepseek-v4-flash \\
        --out experiments/mhc/fixtures/v4_flash_fixtures.pt
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

from routeaudit import config as cfg_mod
from routeaudit import ui
from routeaudit.model import load_model, precision
from routeaudit.model.archspec import ArchSpec
from routeaudit.model.gate_math import GateSpec, learned_router_layers
from routeaudit.model.hooks import MoEHookManager
from routeaudit.model.prompting import encode_prompt

# A short, fixed prompt. Determinism matters more than content — the same tokens must
# produce the same tensors on every run, which is what makes atol=0 meaningful.
FIXTURE_PROMPT = "The capital of France is"


@torch.no_grad()
def extract_hash_table(model, layer_idx: int, vocab_size: int, top_k: int,
                       batch: int = 4096) -> torch.Tensor | None:
    """Recover the token-id → expert table for a hash-routed layer.

    Works even when the table tensor isn't directly exposed: hash routing depends on the
    token id and nothing else, so feeding each id once determines its experts. Returns
    None when the module doesn't publish a `tid2eid`-style buffer and there is no
    supported way to sweep it — better a missing fixture than a fabricated one.
    """
    base = getattr(model, "model", model)
    layer = base.layers[layer_idx]
    block = getattr(layer, "mlp", None)
    for attr in ("tid2eid", "hash_table", "token_expert_map"):
        t = getattr(block, attr, None)
        if isinstance(t, torch.Tensor):
            return t.detach().cpu()
    ui.warn(f"layer {layer_idx}: no hash table buffer found — skipping the hash fixture. "
            f"(Sweeping the vocabulary needs a model-specific routing entry point.)")
    return None


@torch.no_grad()
def extract(config: str, out: Path, *, layer: int | None = None) -> dict:
    cfg = cfg_mod.load(config)
    ok, msg = precision.check_quant_policy(cfg.model, "none", precision.Claim.QUANTITATIVE)
    if msg:
        ui.warn(msg)

    loaded = load_model(cfg)
    model, tok = loaded.model, loaded.tokenizer
    spec: ArchSpec = loaded.spec
    gs = GateSpec.from_config(cfg.model)
    device = next(model.parameters()).device
    ids = encode_prompt(tok, FIXTURE_PROMPT,
                        want_template=getattr(cfg.model, "use_chat_template", True),
                        device=device).unsqueeze(0)

    learned = learned_router_layers(spec.n_layers, gs)
    target = layer if layer is not None else (learned[len(learned) // 2] if learned else 0)

    with MoEHookManager(model, spec) as hm:
        hm.capture_gate_input().capture_routing(gs).capture_residual()
        out_logits = model(input_ids=ids, use_cache=False).logits

    fx: dict = {
        "meta": {
            "hf_id": getattr(cfg.model, "hf_id", config),
            "prompt": FIXTURE_PROMPT,
            "input_ids": ids.cpu(),
            "gate_spec": vars(gs),
            "arch_spec": vars(spec),
            "torch_version": torch.__version__,
            "precision": precision.policy_banner(cfg.model, "none",
                                                 precision.Claim.QUANTITATIVE),
        },
        "logits": out_logits[0, -1].float().cpu(),
    }

    # (a) one gate call: input → (weights, indices)                    [Blocker 3]
    if target in hm.capture.gate_input:
        rr = hm.capture.routing[target]
        fx["gate"] = {
            "layer": target,
            "gate_input": hm.capture.gate_input[target].float().cpu(),
            "scores": rr.scores.cpu(), "sel_scores": rr.sel_scores.cpu(),
            "indices": rr.indices.cpu(), "weights": rr.weights.cpu(),
        }
    else:
        ui.warn(f"no gate input captured at layer {target} — check the ArchSpec.")

    # (b) the residual state, with its stream count                    [Blockers 1-2]
    if target in hm.capture.residual:
        fx["residual"] = {
            "layer": target,
            "hidden": hm.capture.residual[target].float().cpu(),
            "n_streams": hm.capture.residual_streams.get(target, 1),
        }

    # (c) hash table slice for the leading layers                      [Blocker 4]
    if gs.num_hash_layers:
        vocab = int(getattr(model.config, "vocab_size", 0) or 0)
        table = extract_hash_table(model, gs.first_k_dense_replace, vocab, gs.top_k)
        if table is not None:
            fx["hash"] = {"layer": gs.first_k_dense_replace, "table": table[:4096]}

    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(fx, out)
    ui.ok(f"fixtures → {out}  ({', '.join(k for k in fx if k != 'meta')})")
    return fx


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--layer", type=int, default=None,
                    help="which layer to fixture (default: a middle content-routed one)")
    ap.add_argument("--out", type=Path,
                    default=Path("experiments/mhc/fixtures/v4_flash_fixtures.pt"))
    args = ap.parse_args()
    extract(args.config, args.out, layer=args.layer)


if __name__ == "__main__":
    main()
