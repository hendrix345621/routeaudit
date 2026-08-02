# RouteAudit runbook

## Quick start (cloud pod)

```bash
cd /workspace
git clone https://github.com/hendrix345621/routeaudit && cd routeaudit

source ./setup_ram.sh        # run everything in RAM (disk too small); see notes below
pip install -e .
hf auth login                 # HF token (gated weights)

make run                      # asks which model first, then runs all 4 phases → verdict
```

`make run` is the one-shot: it prompts for the model (a preset nickname or any HF `user/model`
id), runs `data → harvest → routeaudit → eval`, and **stops at the SAFE/AT-RISK verdict.**
It uploads nothing. For non-interactive use: `make run MODEL=qwen3`.

> `setup_ram.sh` grows `/dev/shm` to 26 GB and points HF cache + `data/ cache/ artifacts/` at
> RAM. Fine for OLMoE-1B-7B-class models. It is nowhere near enough for large targets (e.g.
> DeepSeek-V4-Flash ships fp8 at ≈ 160 GB — that needs a 2×80 GB node, not a RAM disk).

## What the phases produce

- `artifacts/eval_results.json` + `eval_results.md` — **consolidated results**: model, suffix,
  ASR/MMLU/routing-shift, SAFE/AT-RISK verdict, timestamp (the `.md` is the readable report).
- `artifacts/eval_cells.jsonl` — raw per-cell numbers for programmatic re-grading.
- `artifacts/results/` — **full auditable bundle**: `summary.md` + `per_prompt.md`/`.jsonl`
  (every prompt's clean vs attacked completion + string **and** judge verdict) + `transcripts/`.
  The judge (Llama-Guard-3-1B by default) is language-agnostic, so non-English refusals are
  scored correctly — `make run` runs it by default (`--no-judge` to skip).
- `artifacts/routeaudit_universal.json` — the optimized suffix (the deployable artifact).
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

### DeepSeekMoE / mHC — routing analysis supported, attack not ported

The gate is supported for **routing analysis**: `arch.name: deepseek` plus a `routing:` block
gives correct capture on V2/V3 (sigmoid + node-limited top-k) and V4-Flash (`sqrt(softplus)` +
flat top-6 + selection-only bias + hash-routed first layers). mHC's 4-stream residual is handled
at the hook layer. The **suffix attack is not ported** — its losses assume `softmax(logits)`.

The research lives under [experiments/mhc/](experiments/mhc/README.md):

```bash
python experiments/mhc/tests/run_synthetic.py                 # CPU, seconds — mechanism check
python experiments/mhc/route_mhc.py --config deepseek-v2-lite # routing mass at t*
python experiments/mhc/tests/run_diagnostics.py --config deepseek-v2-lite --quant nf4 \
    --tests margin,leverage,selection,routing,reachability,norm,conservation
```

Precision note: V4-Flash's fp8/fp4 weights are QAT-native, so bitsandbytes NF4 on top adds error
the deployed model doesn't have — `model/precision.py` refuses that combination. NF4 stays fine
for the bf16 sibling (V2-Lite ≈ 9 GB).

## Large models on spot/rented GPUs — cost playbook (Qwen3-235B and up)

For ~235B (~470 GB bf16) the gradient attack is the only phase needing a big white-box node;
**harvest and eval are forward-only**. So the cheapest correct flow never runs the attack on the
big model — it transfers a suffix from the closest sibling and uses the big node only for
forward-only work, loaded once.

```bash
# 1) CHEAP 1-GPU box — optimize the suffix on the closest sibling (shares the routing backbone)
make surrogate MODEL=qwen3            # = run_all --stop-after attack (+ checkpoint/resume)
#   → artifacts/routeaudit_universal.json   (copy this to the big node's persistent volume)

# 2) BIG NODE — ONE model load does forward-only harvest + eval with that suffix
make target MODEL=qwen3-235b          # = target_session --suffix … --judge --resume
```

Full white-box on 235B instead (faithful, expensive): `python scripts/target_session.py
--model qwen3-235b --attack --checkpoint artifacts/attack.ckpt.json --resume` — one load runs
harvest → attack → eval with auto-scaled batches, gradient checkpointing, and prefix cache.

**Why it's cheap & smooth**
- **Surrogate split** keeps the expensive grad attack off the 235B node entirely (per the threat
  model, suffixes transfer across siblings sharing a routing backbone). Transfer is still *measured*
  on the target, so the verdict is honest.
- **Single load** (`target_session.py`) loads the 470 GB once for harvest+eval — no 2-3× reloads.
- **Auto-batch** (`--auto-batch`, on by default in `run_all`) sizes candidate/grad batches +
  n_prompts to the model so the attack doesn't OOM on step 1 (quality-neutral). bf16 only.
- **`model.load:`** in [configs/qwen3_235b_a22b.yaml](configs/qwen3_235b_a22b.yaml) sets
  `attn_implementation`, per-GPU `max_memory`, and an optional `offload_folder` (forward-only).

**Spot-resilience (critical):** everything is resumable — harvest caches each frequency sweep,
the attack checkpoints the best suffix and warm-resumes (`--checkpoint` + `--resume`), and eval
re-runs are cheap. **Put `data/`, `artifacts/`, and `HF_HOME` on a PERSISTENT volume — NOT
`/dev/shm`** ([setup_ram.sh](setup_ram.sh)), or a preemption wipes the checkpoints and the 470 GB
download and resume buys nothing.

**bf16 only:** quantization is deliberately unsupported — it shifts the router logits harvest
localizes and the attack/verdict depend on. Fit via more GPUs / the surrogate / forward-only
offload instead.

## The attack artifact is the suffix (no model export)

RouteAudit is input-only and modifies no weights — there is no checkpoint to "merge" or export.
The deployable result is the **suffix text** in `artifacts/routeaudit_universal.json`. The eval
phase (`scripts/03_eval.py`) prints it and records it into `artifacts/eval_cells.jsonl` next to the
ASR / MMLU / routing-shift numbers, so the verdict and the exact suffix that produced it travel
together.
