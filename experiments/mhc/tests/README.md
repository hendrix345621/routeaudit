# mHC diagnostics — design the suffix search from real frontier-model signals (cheaply)

Isolated under `experiments/mhc/` (gitignored) so it never touches the main project. The premise:
**toy models don't tell you anything real about refusal behavior — you need the actual frontier
model.** So fit the real model in minimal VRAM (4-bit, where the policy allows it), run
forward-only refusal/routing probes, and craft the loss from those signals.

The exception is *mechanism*: the synthetic mHC model (`synthetic_mhc.py`) is a true mHC
architecture running the same gate code as the real thing, so code correctness and the
conservation properties are validated on CPU in seconds before anything is rented.

```bash
# cheapest real probe of a frontier MoE — 4-bit fits ~35B on a single 24 GB card:
python experiments/mhc/tests/run_diagnostics.py --config qwen3.6 --quant nf4 --n-prompts 48
# → artifacts/mhc_diagnostics.md  (takeaways + raw signals)
```

## Runbook — run everything

```bash
# 0) one-time setup (on the pod)
pip install -e .                       # the routeaudit package
pip install bitsandbytes               # for 4-bit/int8 diagnostic quantization
hf auth login                          # gated weights + the Llama-Guard judge
python scripts/00_data.py              # AdvBench prompts → data/ (if not present)

# 1) NO-GPU: unit tests, then Level 0 mechanism validation on a TRUE mHC architecture.
#    Confirms the gate math, the Sinkhorn projection, hash-layer handling and the
#    conservation tests all work before you spend a cent on a real model.
pytest experiments/mhc/tests/ -q
python experiments/mhc/tests/run_synthetic.py

# 2) real-model diagnostics, cheaply (pick the model for the axis you study):
#    a) hybrid-attention MoE on ~1×24 GB:
python experiments/mhc/tests/run_diagnostics.py --config qwen3.6 --quant nf4 --n-prompts 48
#    b) DeepSeek grouped gate (real weights) on ~1×16 GB:
python experiments/mhc/tests/run_diagnostics.py --config deepseek-v2-lite --quant nf4 \
    --tests margin,affirm,leverage,selection,routing,reachability,norm
#    c) the only real mHC model — DeepSeek-V4-Flash — AS SHIPPED (fp8, ~160 GB, 2×80 GB).
#       NOT --quant nf4: see the precision note below.
python experiments/mhc/tests/run_diagnostics.py --config deepseek-v4-flash --quant none

# 3) read the signals → craft the loss (see "How the signals craft the method" below)
cat artifacts/mhc_diagnostics.md
```

`run_diagnostics.py` flags: `--quant {nf4,int8,none}`, `--n-prompts N`,
`--tests <comma-list>`, `--leverage-steps K`, `--multilingual-file f.jsonl`, `--out path`.

## Why quantize for diagnostics — and when you must not

Diagnostics *characterize* behavior (where safety fires, whether an input can move it) and are
robust to 4-bit. Exact scores, margins and flip thresholds are not: confirm those as-shipped.

**DeepSeek-V4-Flash is the exception in the other direction.** Its fp8 weights and fp4 experts
are QAT-native — that *is* the deployed model, not an approximation of a bf16 one — and FP4→FP8
dequantization is lossless. Running bitsandbytes NF4 on top adds error the real model does not
have, so `model/precision.py` **refuses** that combination rather than warning about it. NF4
stays correct for bf16 checkpoints (V2-Lite, Qwen). Quote margins against the published 99.7%
indexer top-k recall — anything finer is below the architecture's own noise floor.

## Cost ladder — fit the real model as small as possible

| Model (real arch) | bf16 | **NF4 4-bit** | int8 | cheapest box |
|---|---|---|---|---|
| **DeepSeek-V2-Lite** (16B, *real grouped+MLA gate*) | ~32 GB | **~9 GB** | ~16 GB | 1×16 GB |
| Qwen1.5-MoE-A2.7B (softmax MoE) | ~29 GB | ~8 GB | ~15 GB | 1×16 GB |
| Qwen3-30B-A3B (softmax MoE, thinking) | ~60 GB | ~17 GB | ~31 GB | 1×24 GB |
| **Qwen3.6-35B-A3B** (*hybrid linear/full ATTENTION; standard residual — NOT mHC*) | ~70 GB | **~19 GB** | ~36 GB | 1×24 GB |
| Qwen3-235B-A22B | ~470 GB | ~120 GB | ~235 GB | 2×80 GB |

Plus: **MoE expert offload** — routing/refusal probes mostly need the router+attention
resident; experts can stream from CPU (`--quant nf4` + a `max_memory` cap that spills
experts to CPU). So even a 235B routing probe can run on a small box, slowly.

**⚠ Three SEPARATE architecture axes** (don't conflate — Qwen3.6 ≠ mHC; verified from its
config.json: no hyper-connection / expansion-rate / Sinkhorn fields, standard single-stream
residual). Develop on the cheapest model with the *binding* feature:
- **biased gate** → **DeepSeek-V2-Lite** (real DeepSeek gate, ~9 GB @ 4-bit). Note it is a
  *lower-fidelity* proxy: V2 is sigmoid + node-limited top-k, where **V4-Flash is
  `sqrt(softplus)` + FLAT top-k** with hash-routed leading layers. It exercises the grouped code
  path, which V4-Flash does not use at all. *V2-Lite is NOT mHC.*
- **mHC residual stream** (DeepSeek-V4 only) → **no small public model exists** (the mHC paper's
  3B/9B/27B aren't released). Cheapest *real* mHC is V4-Flash itself (fp8, ~160 GB, 2×80 GB).
  For mHC *mechanism*, `synthetic_mhc.py` is a genuine build from the paper's equations
  (column-then-row Sinkhorn `H^res`, n=4 streams, hash layers, clamped SwiGLU) running the same
  `gate_math` code as the real model — CPU, seconds.
- **hybrid linear/full attention** (Qwen3.5/3.6, Qwen3-Next) → **Qwen3.6-35B @ 4-bit** (~19 GB).
- *softmax MoE + thinking + multilingual* → Qwen3-30B or Qwen1.5-MoE.

## The test battery (what each signal designs)

| # | test | tells you | cost |
|---|---|---|---|
| 1 | **refusal-margin census** | how hard it refuses per prompt → soft targets, λ_refusal scale | fwd-only |
| 2 | **affirmative receptivity** | how close to complying → whether λ_target has signal to amplify | fwd-only |
| 3 | **suffix-leverage probe** | **GO/NO-GO**: can a (soft-embedding) input even move the decision at t\*? if not, no text suffix can → robustness result | few grad steps |
| 4 | **routing fingerprint** | which experts gate refusal → where to aim the routing loss / harvest | fwd-only |
| — | **selection-margin census** (`margin_census.py`) | **the other half of the GO/NO-GO**: how far each safety expert sits from falling out of the top-k, in `score + bias` units. #3's upper bound is only meaningful against this | fwd-only |
| 7 | **routing reachability vs depth** *(paper-grounded)* | how far an input perturbation propagates into per-layer routing — decay with depth means deep safety experts are unreachable from the input. Cross with #4: can the input reach the layers that gate refusal? | fwd-only |
| 8 | **residual norm conservation** *(paper-grounded)* | boundary hidden-norm vs depth (on the **stream-mean** under mHC) — FLAT = conservation; growth = standard residual | fwd-only |
| 9 | **mHC conservation profile** *(paper-grounded)* | the mechanism itself, not the symptom: is every `B` on the Birkhoff polytope, is `‖B‖₂ ≤ 1`, and what is the perturbation gain vs depth | fwd-only, mHC models |
| 5 | **thinking sensitivity** | does the decision leave t\* when CoT is on? (reasoning models) | fwd-only |
| 6 | **multilingual refusal** | cross-lingual refusal gaps → keep multilingual tokens / attack surface | fwd-only |

Tests **7–9 operationalize the mHC paper's signal-propagation analysis** (its "Amax Gain
Magnitude" — ≈1 for mHC vs ≈3000 for unconstrained HC): they measure whether the conservation
property actually starves an input-only routing attack of leverage at depth. #7 and #8 run on any
model; #9 needs the n-stream internals, so it reports `skipped` on a standard residual instead of
inventing a number.

`--tests margin,affirm,leverage,routing,reachability,norm,thinking` (default); add
`selection` and `conservation` for the two above. Multilingual needs a `--multilingual-file`
jsonl `{lang: [translated prompts]}`.

Hash-routed and dense layers are excluded from every content-based statistic — routing there is
a token-id lookup or absent, so including them dilutes per-layer numbers with layers that cannot
move.

## How the signals craft the method
- **Leverage ≈ 0** (3) **or reachability decays with depth at the refusal-gating layers** (7 vs 4)
  → input-only attack infeasible; write the robustness result, and attribute *why* (bias? grouping?
  norm conservation per 8?).
- **Leverage reachable + the input reaches the refusal-gating experts** (3 + 7 + 4) → focus
  `L_suppress` on those exact (layer, expert)/group cells instead of the generic top-pct.
- **High affirmative receptivity** (2) → `λ_target` will bite; weight it up.
- **Decision leaves t\* with thinking on** (5) → keep `enable_thinking: false`, or build the
  post-`</think>` re-anchoring (see ../scoping.md).
- **Multilingual gap** (6) → keep `ascii_only: false`; the cross-lingual surface is real.
- **Flat residual norm** (8) corroborates a conservation-driven robustness story.

## Files
- `diag_common.py` — loader (subject to the precision policy) + **one** boundary
  logits/routing path for every gate, via `gate_math.GateSpec`. Returns the full
  `RouteResult` so the bias-free gating weights and the bias-inclusive selection scores stay
  distinguishable.
- `refusal_tests.py` — tests 1–9 (reuse the `routeaudit` package read-only).
- `margin_census.py` — the selection-margin census + `compare_to_leverage`, which states the
  P1 verdict from the two halves together. Also runnable standalone.
- `run_diagnostics.py` — load once, run the battery, write `artifacts/mhc_diagnostics.{json,md}`.
- `synthetic_mhc.py` — a TINY, CPU-runnable, **genuine mHC** model: column-then-row Sinkhorn
  `B`, n-stream residual, hash-routed leading layers, clamped SwiGLU experts, and both released
  gate configurations (`FLASH_LIKE`, `V2_LIKE`) routed through the *same* `gate_math` code as a
  real model. Random weights ⇒ refusal semantics meaningless; it validates **code + mechanism**.
- `run_synthetic.py` — Level 0 of the validation ladder. Asserts the Sinkhorn order, both gate
  configs, hash-table exactness, mHC replay, perturbation gain and the residual reduction.
  Exits non-zero on failure (CPU, seconds).
- `test_gate_math.py` / `test_mhc.py` / `test_no_regression.py` — pytest. The last one pins the
  softmax-gate hook behavior so shared changes can't shift the existing pipeline's numbers.
- `../fixtures/{extract,validate}.py` — ladder Level 1, **pending** DeepSeek-V4-Flash access.

## Caveats
- 4-bit shifts routing — use it for **direction/structure**, confirm exact numbers as-shipped.
  It is refused outright on QAT-native fp8/fp4 checkpoints (see the precision section above).
- Needs `bitsandbytes` installed (`pip install bitsandbytes`) and the model's gated access.
- The leverage probe backprops to input embeddings (works on 4-bit; frozen weights, live activations).
- The synthetic model validates mechanism, never semantics. Any claim about *refusal* needs a
  real trained model, and any claim about *mHC* specifically needs DeepSeek-V4.
