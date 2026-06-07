"""Stage 02 — RouteHijack universal-suffix attack (needs 01 experts).

Optimizes one adversarial suffix that suppresses safety-expert routing, promotes
harmful experts, and blocks early refusal (RouteHijack, arXiv 2605.02946). Writes
the suffix + per-prompt attack transcripts + TESR/THPR routing-shift diagnostics.
"""
from __future__ import annotations

import argparse

from routehijack import config as cfg_mod
from routehijack import ui
from routehijack.model import load_model
from routehijack.pipeline import attack_run


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
    p.add_argument("--auto-batch", action="store_true",
                   help="size candidate/grad batches + n_prompts to the model (avoids OOM on large "
                        "models); also turns on prefix-cache + grad-checkpointing past ~20B. "
                        "Overrides the manual batch flags.")
    p.add_argument("--grad-checkpointing", action="store_true",
                   help="checkpoint the backward pass (trades compute for memory; quality-neutral)")
    p.add_argument("--checkpoint", default=None,
                   help="JSON path to dump the best suffix on every improvement (spot-friendly)")
    p.add_argument("--resume", action="store_true",
                   help="warm-resume the suffix search from --checkpoint if it exists")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--gen-batch-size", type=int, default=8,
                   help="prompts generated per batched forward when scoring (raise to use "
                        "more GPU; lower if the KV cache OOMs)")
    p.add_argument("--show-samples", type=int, default=3)
    args = p.parse_args()

    cfg = cfg_mod.load(args.config)
    ui.step_header(3, "RouteHijack — universal suffix attack", total=4)
    loaded = load_model(cfg)
    attack_run(loaded, cfg, args)
    ui.print_done("RouteHijack complete")


if __name__ == "__main__":
    main()
