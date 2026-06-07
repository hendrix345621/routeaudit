# RouteHijack runbook

## Quick start (cloud pod)

```bash
cd /workspace
git clone <repo-url> && cd routehijack

source ./setup_ram.sh        # run everything in RAM (disk too small); see notes below
pip install -e .
hf auth login                 # HF token (gated weights)

make run                      # asks which model first, then runs all 4 phases → verdict
```

`make run` is the one-shot: it prompts for the model (a preset nickname or any HF `user/model`
id), runs `data → harvest → routehijack → eval`, and **stops at the SAFE/AT-RISK verdict.**
It uploads nothing. For non-interactive use: `make run MODEL=qwen3`.

> `setup_ram.sh` grows `/dev/shm` to 26 GB and points HF cache + `data/ cache/ artifacts/` at
> RAM. Fine for OLMoE-1B-7B-class models. It is nowhere near enough for large targets (e.g.
> DeepSeek-V4-Flash ≈ 570 GB in bf16 — that needs real multi-GPU, not a RAM disk).

## What the phases produce

- `artifacts/eval_results.json` + `eval_results.md` — **consolidated results**: model, suffix,
  ASR/MMLU/routing-shift, SAFE/AT-RISK verdict, timestamp (the `.md` is the readable report).
- `artifacts/eval_cells.jsonl` — raw per-cell numbers for programmatic re-grading.
- `artifacts/transcripts/*.md` — readable samples.
- `artifacts/routehijack_universal.json` — the optimized suffix (the deployable artifact).
- `artifacts/safety_experts.json`, `harmful_experts.json` — localized experts.

## Supported MoE families

| Family | nickname / how to select | attack | notes |
|---|---|---|---|
| OLMoE | `olmoe` / `base` (· `smoke` = tiny sanity run) | ✓ | default target |
| Mixtral | `mixtral` | ✓ | Mixtral-8x7B; fused experts on newer HF → router capture still fine |
| Qwen2-MoE | `qwen2` | ✓ | Qwen1.5-MoE-A2.7B; shared_expert intentionally not hooked |
| Qwen3-MoE | `qwen3` | ✓ | Qwen3-30B-A3B; no shared expert |
| Qwen3-235B-A22B | `qwen3-235b` | ✓ | 94L · 128 experts · top-8 · no shared expert; ~470 GB → multi-GPU |
| Qwen3.6-35B-A3B | `qwen3.6` | ✓ | hybrid attention (linear+full); 40L · 256 experts · top-8 · shared expert (unhooked); dims verified |
| Qwen3.5 MoE | `qwen3.5` | ~ best-effort | hybrid-attention MoE; dims unconfirmed — verify (config header) |
| Phi-3.5-MoE | HF id `microsoft/Phi-3.5-MoE-instruct` | ✓ | clean Linear gate, Mixtral-like |

Passing a raw HF id to any script (or `make run MODEL=<id>`) auto-detects family + dims for
supported `model_type`s, else raises `UnsupportedModelError` with guidance.

DBRX / GPT-OSS / Granite-MoE are **not** wired in: their gates return tuples / sit at non-standard
paths and need an ArchSpec router-path generalization first.

### DeepSeek-V4 / mHC — separate experiment, not in this pipeline

DeepSeek-V4's grouped/biased top-k gate cannot be steered by the suffix attack, so it is **not**
part of the main pipeline. It lives on its own under [mhc/](mhc/README.md): a faithful routing
diagnostic (`python mhc/route_mhc.py`), a best-effort config + verification checklist, and a
written explanation (`mhc/README.md`) of why the attack fails and what would be required to try.

## The attack artifact is the suffix (no model export)

RouteHijack is input-only and modifies no weights — there is no checkpoint to "merge" or export.
The deployable result is the **suffix text** in `artifacts/routehijack_universal.json`. The eval
phase (`scripts/03_eval.py`) prints it and records it into `artifacts/eval_cells.jsonl` next to the
ASR / MMLU / routing-shift numbers, so the verdict and the exact suffix that produced it travel
together.
