"""One-shot RouteHijack runner — pick a model, run all four phases end to end.

    python scripts/run_all.py                       # interactive: choose model, run
    python scripts/run_all.py --model qwen3 --yes    # non-interactive (automation)
    python scripts/run_all.py --model microsoft/Phi-3.5-MoE-instruct   # by HF id

Phases (each loads the model once, reusing the prior phase's artifacts):

    1 data → 2 harvest → 3 routehijack → 4 eval  →  SAFE / AT-RISK verdict

This runner ENDS AT THE VERDICT. RouteHijack is input-only: the deployable artifact
is the suffix text (artifacts/routehijack_universal.json), which the eval phase
prints and records — there is no model to export.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from routehijack import config as cfg_mod
from routehijack import ui

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
    p = argparse.ArgumentParser(description="Run the full RouteHijack pipeline end to end.")
    p.add_argument("--model", help="nickname (olmoe/mixtral/qwen3/…), config path, or HF id. "
                                    "Omit for an interactive prompt.")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt (automation)")
    p.add_argument("--skip-data", action="store_true", help="reuse existing corpora (skip phase 1)")
    p.add_argument("--capture-only", action="store_true",
                   help="run data + harvest, then stop before the suffix attack/eval")
    p.add_argument("--judge", action="store_true", help="re-grade eval ASR with HarmBench (phase 4)")
    args = p.parse_args()

    ui.big_banner("RouteHijack — end-to-end run")

    model = args.model or _pick_model_interactive()
    cfg, arch = _resolve_model(model)

    md = cfg.model
    ui.kv_panel("Target", {
        "model": getattr(md, "hf_id", model), "arch": arch,
        "layers": getattr(md, "n_layers", "?"), "experts": getattr(md, "n_experts", "?"),
        "top_k": getattr(md, "top_k", "?"), "d_model": getattr(md, "d_model", "?"),
        "mode": "capture-only" if args.capture_only else "full attack",
    })

    if not args.yes:
        ans = input("\n  Proceed with this run? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            ui.info("aborted.")
            return

    # Phase 1 — data
    if not args.skip_data:
        _run_phase("1/4 data", [PYTHON, "scripts/00_data.py", "--data-dir", args.data_dir])
    else:
        ui.info("skipping phase 1 (--skip-data); reusing existing corpora.")

    # Phase 2 — harvest (expert localization)
    _run_phase("2/4 harvest", [PYTHON, "scripts/01_harvest.py", "--config", model])

    if args.capture_only:
        ui.print_done("capture-only run complete (harvest done; suffix attack skipped). "
                      "See artifacts/safety_experts.json, harmful_experts.json")
        return

    # Phase 3 — routehijack (universal suffix attack); flags mirror the Makefile.
    _run_phase("3/4 routehijack", [
        PYTHON, "-u", "scripts/02_routehijack.py", "--config", model,
        "--n-prompts", "16", "--n-steps", "300", "--candidates-per-step", "128",
        "--candidate-prompt-subsample", "0", "--grad-batch-size", "8",
        "--candidate-batch-size", "128", "--early-stop-patience", "40",
    ])

    # Phase 4 — eval (ASR + MMLU + routing shift + verdict)
    eval_cmd = [PYTHON, "scripts/03_eval.py", "--config", model]
    if args.judge:
        eval_cmd.append("--judge")
    _run_phase("4/4 eval", eval_cmd)

    ui.print_done("End-to-end run complete — see artifacts/eval_cells.jsonl for the "
                  "SAFE/AT-RISK verdict + the deployable suffix, and artifacts/transcripts/ "
                  "for samples.")


if __name__ == "__main__":
    main()
