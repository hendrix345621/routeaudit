"""One-shot RouteAudit runner — pick a model, run all four phases end to end.

    python scripts/run_all.py                       # interactive: choose model, run
    python scripts/run_all.py --model qwen3 --yes    # non-interactive (automation)
    python scripts/run_all.py --model microsoft/Phi-3.5-MoE-instruct   # by HF id

Phases (each loads the model once, reusing the prior phase's artifacts):

    1 data → 2 harvest → 3 routeaudit → 4 eval  →  SAFE / AT-RISK verdict

This runner ENDS AT THE VERDICT. RouteAudit is input-only: the deployable artifact
is the suffix text (artifacts/routeaudit_universal.json), which the eval phase
prints and records — there is no model to export.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from routeaudit import config as cfg_mod
from routeaudit import ui

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _resolve_model(model: str):
    """Resolve a nickname / path / HF id to a config namespace, or exit with a
    clean message. Returns (cfg, arch_name)."""
    try:
        cfg = cfg_mod.load(model)
    except cfg_mod.UnsupportedModelError as e:
        ui.fail(str(e))
        sys.exit(2)
    except Exception as e:  # noqa: BLE001
        ui.fail(f"could not resolve model '{model}': {type(e).__name__}: {e}")
        sys.exit(2)
    arch = getattr(getattr(cfg.model, "arch", None), "name", "olmoe")
    return cfg, arch


def _pick_model_interactive() -> str:
    """Step 0 — choose the target model (the first thing the user is asked)."""
    ui.section("Step 0 — pick the target MoE model")
    nicks = cfg_mod.list_models()
    ui.console().print("  Known presets: " + ", ".join(f"[bold]{n}[/bold]" for n in nicks))
    ui.info("…or paste any HuggingFace MoE id, e.g. Qwen/Qwen3-30B-A3B, "
            "microsoft/Phi-3.5-MoE-instruct")
    choice = input("\n  Model (nickname or hf user/model): ").strip()
    if not choice:
        ui.fail("no model given.")
        sys.exit(2)
    return choice


def _run_phase(title: str, cmd: list[str]) -> None:
    ui.section(f"▶ {title}")
    ui.info(" ".join(cmd))
    res = subprocess.run(cmd, cwd=REPO)
    if res.returncode != 0:
        ui.fail(f"phase '{title}' exited with code {res.returncode}. Stopping.")
        sys.exit(res.returncode)


def main() -> None:
    p = argparse.ArgumentParser(description="Run the full RouteAudit pipeline end to end.")
    p.add_argument("--model", help="nickname (olmoe/mixtral/qwen3/…), config path, or HF id. "
                                    "Omit for an interactive prompt.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt (automation)")
    p.add_argument("--skip-data", action="store_true", help="reuse existing corpora (skip phase 1)")
    p.add_argument("--capture-only", action="store_true",
                   help="run data + harvest, then stop before the suffix attack/eval")
    p.add_argument("--stop-after", choices=["data", "harvest", "attack", "eval"], default="eval",
                   help="stop after this phase. Use 'attack' on a cheap SURROGATE box to produce a "
                        "transferable suffix, then evaluate it on the big model with target_session.py.")
    p.add_argument("--auto-batch", action="store_true", default=True,
                   help="size attack batches to the model (avoids OOM on large models); on by default")
    p.add_argument("--no-auto-batch", dest="auto_batch", action="store_false",
                   help="disable auto-batch and use the manual attack batch flags")
    p.add_argument("--checkpoint", default=None,
                   help="attack suffix checkpoint path (spot-friendly; pairs with --resume)")
    p.add_argument("--resume", action="store_true",
                   help="resume harvest sweeps + attack from checkpoints")
    p.add_argument("--judge", action="store_true", default=True,
                   help="grade eval ASR with the config's judge (default Llama-Guard-3-1B); on by default")
    p.add_argument("--no-judge", dest="judge", action="store_false",
                   help="skip the judge and report the (string-detector) ASR only")
    args = p.parse_args()

    ui.big_banner("RouteAudit — end-to-end run")

    model = args.model or _pick_model_interactive()
    cfg, arch = _resolve_model(model)

    stop_after = "harvest" if args.capture_only else args.stop_after

    md = cfg.model
    ui.kv_panel("Target", {
        "model": getattr(md, "hf_id", model), "arch": arch,
        "layers": getattr(md, "n_layers", "?"), "experts": getattr(md, "n_experts", "?"),
        "top_k": getattr(md, "top_k", "?"), "d_model": getattr(md, "d_model", "?"),
        "stop after": stop_after, "auto-batch": args.auto_batch,
    })

    if not args.yes:
        ans = input("\n  Proceed with this run? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            ui.info("aborted.")
            return

    # Phase 1 — data
    if not args.skip_data:
        _run_phase("data", [PYTHON, "scripts/00_data.py", "--data-dir", args.data_dir])
    else:
        ui.info("skipping data phase (--skip-data); reusing existing corpora.")
    if stop_after == "data":
        ui.print_done("stopped after data."); return

    # Phase 2 — harvest (expert localization)
    harvest_cmd = [PYTHON, "scripts/01_harvest.py", "--config", model]
    if args.resume:
        harvest_cmd.append("--resume")
    _run_phase("harvest", harvest_cmd)
    if stop_after == "harvest":
        ui.print_done("stopped after harvest (safety/harmful experts written)."); return

    # Phase 3 — routeaudit (universal suffix attack)
    attack_cmd = [PYTHON, "-u", "scripts/02_suffix_search.py", "--config", model,
                  "--n-steps", "300", "--candidates-per-step", "128",
                  "--candidate-prompt-subsample", "0", "--early-stop-patience", "30"]
    if args.auto_batch:
        attack_cmd.append("--auto-batch")          # sizes batches + prefix-cache + grad-ckpt to the model
    else:
        attack_cmd += ["--n-prompts", "16", "--grad-batch-size", "8", "--candidate-batch-size", "128"]
    if args.checkpoint:
        attack_cmd += ["--checkpoint", args.checkpoint]
    if args.resume:
        attack_cmd.append("--resume")
    _run_phase("routeaudit (attack)", attack_cmd)
    if stop_after == "attack":
        ui.print_done("stopped after attack — suffix at artifacts/routeaudit_universal.json. "
                      "Transfer it to a big model with: python scripts/target_session.py "
                      "--model <target> --suffix artifacts/routeaudit_universal.json")
        return

    # Phase 4 — eval (ASR + MMLU + routing shift + verdict)
    eval_cmd = [PYTHON, "scripts/03_eval.py", "--config", model,
                "--judge" if args.judge else "--no-judge"]
    _run_phase("eval", eval_cmd)

    ui.print_done("End-to-end run complete — see artifacts/results/ for the full bundle "
                  "(summary.md · per_prompt.md with every prompt's clean vs attacked + verdict).")


if __name__ == "__main__":
    main()
