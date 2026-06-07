"""Single-load target session for large models (e.g. Qwen3-235B-A22B).

Loads the (expensive) target model ONCE and runs the forward-only phases —
`harvest` then `eval` — against that single resident model, avoiding repeated
100s-of-GB reloads on a billed multi-GPU node. Two modes:

  • Surrogate-transfer (cost-effective default): optimize the suffix on a small
    sibling elsewhere, then here just measure on the target with that suffix.

        # on a cheap box: produce the suffix
        python scripts/run_all.py --model qwen3 --yes --stop-after attack
        # on the big node: one load, harvest + eval the 235B with that suffix
        python scripts/target_session.py --model qwen3-235b \
            --suffix artifacts/routehijack_universal.json --judge

  • Full white-box on the target (configurable): add --attack to also run the
    gradient attack here, in the same single load (auto-scaled batches + grad
    checkpointing + prefix cache + checkpoint/resume).

        python scripts/target_session.py --model qwen3-235b --attack --checkpoint \
            artifacts/attack.ckpt.json --resume --judge

Everything is resumable (harvest sweep cache, attack checkpoint) so spot
preemption is cheap — keep `data/` + `artifacts/` on a PERSISTENT volume, not
/dev/shm (which is wiped on restart).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

from routehijack import config as cfg_mod
from routehijack import ui
from routehijack.model import load_model
from routehijack.pipeline import attack_run, eval_run, harvest_run

SAFETY = "artifacts/safety_experts.json"
HARMFUL = "artifacts/harmful_experts.json"
SUFFIX = "artifacts/routehijack_universal.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="base", help="target nickname / config path / HF id")
    p.add_argument("--suffix", default=SUFFIX,
                   help="suffix JSON to evaluate (from a surrogate). Ignored if --attack is set.")
    p.add_argument("--attack", action="store_true",
                   help="run the white-box attack ON the target in this same load (else use --suffix)")
    p.add_argument("--skip-harvest", action="store_true",
                   help="reuse existing safety/harmful expert files instead of re-harvesting")
    p.add_argument("--resume", action="store_true",
                   help="resume harvest sweeps + (with --attack) the attack checkpoint")
    p.add_argument("--checkpoint", default="artifacts/attack.ckpt.json",
                   help="attack suffix checkpoint path (spot-friendly; used with --attack)")
    p.add_argument("--judge", action="store_true", help="re-grade eval ASR with HarmBench")
    p.add_argument("--n-prompts", type=int, default=100, help="eval prompt count")
    p.add_argument("--attack-n-prompts", type=int, default=16, help="attack universal-batch size")
    p.add_argument("--n-steps", type=int, default=300, help="attack steps")
    p.add_argument("--freq-batch-size", type=int, default=16)
    p.add_argument("--gen-batch-size", type=int, default=8)
    p.add_argument("--mmlu-batch-size", type=int, default=16)
    args = p.parse_args()

    cfg = cfg_mod.load(args.model)
    ui.big_banner(f"Target session (single load) — {getattr(cfg.model, 'hf_id', args.model)}")
    loaded = load_model(cfg)          # ← the ONLY model load for all phases below
    ui.ok("model loaded once; running all phases against this resident model.")

    # ── Phase: harvest (forward-only) ──
    if args.skip_harvest and Path(SAFETY).exists() and Path(HARMFUL).exists():
        ui.info(f"skipping harvest; reusing {SAFETY} / {HARMFUL}")
    else:
        ui.step_header(2, "Harvest — identify experts (target)", total=4)
        harvest_run(loaded, cfg, SimpleNamespace(
            out_safety=SAFETY, out_harmful=HARMFUL, out_diag="artifacts/identify_diagnostics.pt",
            freq_batch_size=args.freq_batch_size, resume=args.resume))

    # ── Phase: obtain the suffix (white-box here, or transferred) ──
    if args.attack:
        ui.step_header(3, "RouteHijack attack ON target (single load)", total=4)
        res = attack_run(loaded, cfg, SimpleNamespace(
            safety=SAFETY, harmful=HARMFUL, advbench="data/advbench.jsonl",
            universal_out=SUFFIX, out="artifacts/routehijack_attacks.jsonl",
            shift_out="artifacts/routehijack_routing_shift.json",
            n_prompts=args.attack_n_prompts, n_steps=args.n_steps,
            candidates_per_step=128, candidate_prompt_subsample=0,
            early_stop_patience=40, auto_batch=True, grad_checkpointing=True,
            prefix_kv_cache=True, checkpoint=args.checkpoint, resume=args.resume,
            gen_batch_size=args.gen_batch_size, max_new_tokens=128, show_samples=3))
        suffix_path = res["suffix_path"]
    else:
        suffix_path = args.suffix
        if not Path(suffix_path).exists():
            ui.fail(f"no suffix at {suffix_path}. Produce one on a surrogate "
                    f"(run_all --stop-after attack) or pass --attack to optimize here.")
            raise SystemExit(2)
        ui.info(f"surrogate-transfer: evaluating target with suffix from {suffix_path}")

    # ── Phase: eval (forward-only) ──
    ui.step_header(4, "Evaluate on target (single load)", total=4)
    eval_run(loaded, cfg, SimpleNamespace(
        config=args.model, safety=SAFETY, harmful=HARMFUL, suffix=suffix_path,
        advbench="data/advbench.jsonl", mmlu="data/mmlu_subset.jsonl",
        n_prompts=args.n_prompts, max_new_tokens=128, judge=args.judge,
        judge_device="cuda", asr_threshold=0.5, gen_batch_size=args.gen_batch_size,
        mmlu_batch_size=args.mmlu_batch_size, out="artifacts/eval_cells.jsonl",
        results="artifacts/eval_results.json"))

    ui.print_done("Target session complete (one model load). See artifacts/eval_results.md")


if __name__ == "__main__":
    main()
