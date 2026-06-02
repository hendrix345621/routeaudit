"""Stage 01 — harvest: expert identification (model loaded once).

  • Identify: F_l(e|safe), F_l(e|harm), F_l(e|gen) over the contrast pairs (response
    tokens) → Score_safe / Score_harm → top-pct safety + harmful experts (RouteHijack §5).

Outputs:
  artifacts/safety_experts.json, artifacts/harmful_experts.json
  artifacts/identify_diagnostics.pt   (score_safe / score_harm tensors)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from routehijack import config as cfg_mod
from routehijack import ui
from routehijack.data import iter_general, iter_harm_pairs, iter_safe_pairs
from routehijack.identify.activation_freq import compute_expert_freq
from routehijack.identify.delta_s import score_harm, score_safe
from routehijack.identify.select import save_experts, select_harmful_experts, select_safety_experts
from routehijack.model import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--out-safety", default="artifacts/safety_experts.json")
    p.add_argument("--out-harmful", default="artifacts/harmful_experts.json")
    p.add_argument("--out-diag", default="artifacts/identify_diagnostics.pt")
    p.add_argument("--freq-batch-size", type=int, default=16,
                   help="sequences per forward in the expert-frequency sweeps (lower if VRAM-tight)")
    args = p.parse_args()

    cfg = cfg_mod.load(args.config)
    ui.step_header(2, "Harvest — identify experts", total=4)
    loaded = load_model(cfg)
    model, tok, spec = loaded.model, loaded.tokenizer, loaded.spec
    L, E, K = cfg.model.n_layers, cfg.model.n_experts, cfg.model.top_k

    # ── Identify (response-token routing frequencies) ──
    ui.section("Activation-frequency sweeps")
    use_tmpl = getattr(cfg.model, "use_chat_template", True)
    fk = dict(n_layers=L, n_experts=E, top_k=K, spec=spec, batch_size=args.freq_batch_size,
              use_chat_template=use_tmpl)
    safe = compute_expert_freq(model, tok, iter_safe_pairs(cfg.identify.pairs_path), desc="F_safe", **fk)
    harm = compute_expert_freq(model, tok, iter_harm_pairs(cfg.identify.pairs_path), desc="F_harm", **fk)
    gen = compute_expert_freq(model, tok, iter_general(cfg.identify.general_corpus_path), desc="F_gen", **fk)

    s_safe = score_safe(safe, harm, gen)
    s_harm = score_harm(safe, harm)
    top_pct = cfg.identify.top_pct
    safety_experts = select_safety_experts(s_safe, top_pct=top_pct)
    harmful_experts = select_harmful_experts(s_harm, top_pct=top_pct)
    save_experts(safety_experts, args.out_safety)
    save_experts(harmful_experts, args.out_harmful)
    Path(args.out_diag).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"score_safe": s_safe, "score_harm": s_harm,
                "F_safe": safe.freq, "F_harm": harm.freq, "F_gen": gen.freq}, args.out_diag)
    ui.ok(f"safety={len(safety_experts)}  harmful={len(harmful_experts)} → {args.out_safety}")
    ui.print_done("Harvest complete")


if __name__ == "__main__":
    main()
