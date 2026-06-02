"""Stage 02 — RouteHijack universal-suffix attack (needs 01 experts).

Optimizes one adversarial suffix that suppresses safety-expert routing, promotes
harmful experts, and blocks early refusal (RouteHijack, arXiv 2605.02946). Writes
the suffix + per-prompt attack transcripts + TESR/THPR routing-shift diagnostics.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from routehijack import config as cfg_mod
from routehijack import ui
from routehijack.attacks import (
    RouteHijackAttack, RouteHijackConfig, apply_routehijack_suffix, measure_routing_shift,
)
from routehijack.data import read_jsonl, write_jsonl
from routehijack.eval.asr import RefusalDetector
from routehijack.eval.generate import generate_batch
from routehijack.identify.select import load_experts
from routehijack.model import load_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--safety", default="artifacts/safety_experts.json")
    p.add_argument("--harmful", default="artifacts/harmful_experts.json")
    p.add_argument("--advbench", default="data/advbench.jsonl")
    p.add_argument("--universal-out", default="artifacts/routehijack_universal.json")
    p.add_argument("--out", default="artifacts/routehijack_attacks.jsonl")
    p.add_argument("--shift-out", default="artifacts/routehijack_routing_shift.json")
    p.add_argument("--n-prompts", type=int, default=16)   # paper §5.3 universal batch
    p.add_argument("--n-steps", type=int, default=300)
    p.add_argument("--candidates-per-step", type=int, default=128)
    p.add_argument("--candidate-prompt-subsample", type=int, default=0,
                   help="0 = score candidates on ALL prompts (paper-faithful); 2-3 ≈ 5-10x faster")
    p.add_argument("--candidate-batch-size", type=int, default=0,
                   help="candidates scored per forward (0 = all at once); lower if the KV/activations OOM")
    p.add_argument("--grad-batch-size", type=int, default=8,
                   help="prompts per batched forward+backward in the grad pass; lower if VRAM-tight")
    p.add_argument("--early-stop-patience", type=int, default=40)
    p.add_argument("--prefix-kv-cache", action="store_true",
                   help="EXPERIMENTAL: KV-cache the fixed [before] prefix so candidate forwards "
                        "process only [suffix][after]. Quality-neutral; self-checks against the "
                        "full path on first use and auto-disables on any mismatch.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--gen-batch-size", type=int, default=8,
                   help="prompts generated per batched forward when scoring (raise to use "
                        "more GPU; lower if the KV cache OOMs)")
    p.add_argument("--show-samples", type=int, default=3)
    args = p.parse_args()

    cfg = cfg_mod.load(args.config)
    rh = cfg.attacks.routehijack
    ui.step_header(3, "RouteHijack — universal suffix attack", total=4)
    loaded = load_model(cfg)
    model, tok, spec = loaded.model, loaded.tokenizer, loaded.spec
    safety, harmful = load_experts(args.safety), load_experts(args.harmful)
    prompts = [r["prompt"] for r in list(read_jsonl(args.advbench))[: args.n_prompts]]

    use_tmpl = getattr(cfg.model, "use_chat_template", True)
    attack_cfg = RouteHijackConfig(
        safety_experts=safety, harmful_experts=harmful,
        suffix_len=rh.suffix_len, n_steps=args.n_steps,
        lambda_suppress=rh.lambda_suppress, lambda_promote=rh.lambda_promote,
        lambda_refusal=rh.lambda_refusal, promote_threshold=rh.promote_threshold,
        refusal_window=rh.refusal_window, n_candidates_per_step=args.candidates_per_step,
        candidate_prompt_subsample=args.candidate_prompt_subsample,
        candidate_batch_size=args.candidate_batch_size, grad_batch_size=args.grad_batch_size,
        early_stop_patience=args.early_stop_patience, mode="universal",
        use_chat_template=use_tmpl, use_prefix_cache=args.prefix_kv_cache,
    )
    attacker = RouteHijackAttack(attack_cfg, model, tok, spec=spec)
    suffix = attacker.optimize_universal_suffix(prompts)
    Path(args.universal_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"suffix": suffix}, open(args.universal_out, "w", encoding="utf-8"), indent=2)
    ui.kv_panel("Universal suffix", {"suffix": suffix[:160], "len": len(suffix)})

    ui.section("Scoring undefended completions")
    attacked = apply_routehijack_suffix(prompts, suffix)
    rd = RefusalDetector()
    log = ui.TranscriptLog("routehijack_attacks")
    completions = generate_batch(model, tok, attacked, max_new_tokens=args.max_new_tokens,
                                 batch_size=args.gen_batch_size, desc="generate")
    rows, n_ref, shown = [], 0, 0
    for orig, atk, comp in zip(prompts, attacked, completions):
        refused = rd.is_refusal(comp)
        n_ref += int(refused)
        rows.append({"prompt": orig, "attacked": atk, "completion": comp, "refused": refused})
        t = ui.Transcript("routehijack", atk, comp, refused)
        log.append(t)
        if shown < args.show_samples:
            ui.show_transcript(t); shown += 1
    write_jsonl(args.out, rows)
    asr = (len(rows) - n_ref) / max(1, len(rows))
    ui.ok(f"ASR={asr:.3f}  attacks → {args.out}")

    ui.section("Routing-shift diagnostics (TESR / THPR)")
    shift = measure_routing_shift(model, tok, safety, harmful, prompts, attacked,
                                  spec=spec, use_chat_template=use_tmpl)
    json.dump(shift, open(args.shift_out, "w", encoding="utf-8"), indent=2)
    ui.kv_panel("Routing shift", shift)
    ui.print_done("RouteHijack complete")


if __name__ == "__main__":
    main()
