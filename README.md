# RouteHijack

**A routing-aware, input-only jailbreak for Mixture-of-Experts (MoE) LLMs.**

RouteHijack optimizes a single adversarial **suffix** that, appended to a harmful prompt,
steers the model's internal **routing** away from the experts responsible for refusal and
toward experts associated with compliance — bypassing safety alignment using **text input
only**, with no access to weights or inference code at attack time.

> Reproduces *RouteHijack: Routing-Aware Attack on Mixture-of-Experts LLMs*
> ([arXiv:2605.02946](https://arxiv.org/abs/2605.02946)).

---

## The idea in one minute

In an MoE layer, a small **router** sends each token to a few of many expert FFNs:

```
MoE(x) = Σ_{e ∈ TopK}  p_e(x) · E_e(x)        p = softmax(router · x)
```

Safety behaviour isn't spread evenly across the model — it **concentrates in a small set
of "safety experts"** that fire preferentially when the model refuses. If you can nudge the
router away from those experts (and toward harmful-leaning ones) at the moment generation
begins, the model proceeds as if its refusal machinery were never consulted.

Because routing is driven by **continuous** router scores, you can optimize an input suffix
to shift them — even though the Top-K selection itself is discrete. That is what RouteHijack
does, and it's why the attack remains **input-only** at deployment.

---

## How it works

**1. Response-driven expert localization.** For every `(layer, expert)`, measure how often
it fires on **safe refusals** vs **harmful completions** (counting *response* tokens, not the
prompt — response-driven profiling is far more discriminative). Define a safety differential
and rank experts:

```
F_l(e | a)        = activation frequency of expert e (layer l) over response a
Δ_S(l,e)          = F(e | safe) − F(e | harmful)
Score_safe(l,e)   = Δ_S − F(e | general)²       # utility penalty: drop general-purpose experts
Score_harm(l,e)   = −Δ_S
```

The top-20% by `Score_safe` are the **safety experts** `E_safe`; the top-20% by `Score_harm`
are the **harmful experts** `E_harm`.

**2. Ternary-loss suffix optimization.** Optimize a `T`-token suffix (GCG-style discrete
search over the input) against three terms evaluated at the **boundary token** `t*` (the last
input position before decoding, rendered through the model's chat template):

```
L = λ₁·L_suppress  +  λ₂·L_promote  +  λ₃·L_refusal      (λ = 3 : 1 : 1)

L_suppress = routing mass on safety experts at t*                       (push down)
L_promote  = max(0, m_harm − routing mass on harmful experts at t*)     (push up, bounded)
L_refusal  = unlikelihood of refusal-opener tokens at the first step    (block "I'm sorry…")
```

Optimization is gradient-guided discrete search: gradients of `L` w.r.t. the suffix one-hots
(through the **soft** router probabilities) propose top-k token swaps per position; candidates
are scored in a batched forward and the best improvement is kept. A decode-then-re-encode
length filter keeps the suffix's tokenization stable so what you optimize is what deploys.

**3. Deploy.** Append the optimized suffix to any harmful prompt as **plain text**.

---

## Pipeline (4 phases)

```bash
make data         # 1. corpora: LLM-LAT contrast pairs, C4, AdvBench, MMLU
make harvest      # 2. localize safety + harmful experts        → artifacts/*_experts.json
make routehijack  # 3. optimize the universal suffix            → artifacts/routehijack_universal.json
make eval         # 4. ASR + MMLU utility + routing shift + SAFE/AT-RISK verdict
# or:
make all
```

Each phase reuses the prior phase's artifacts; the model is loaded once per phase.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .
hf auth login                                         # HF token, for gated model weights
```

Requires Python ≥ 3.10 and a CUDA GPU for real runs. The default target
(`allenai/OLMoE-1B-7B-0924-Instruct`) needs a 24 GB+ card to stay fully on-GPU; smaller cards
fall back to CPU/disk offload via `device_map: auto` and run far slower.

---

## Threat model

Two stages, matching how open-weight backbones get repackaged into deployed products:

- **Offline (white-box surrogate):** full access to a related open-weight MoE — weights,
  activations, router logits — used to localize experts and optimize the suffix.
- **Deployment (input-only):** the suffix is appended to prompts as text. No weight edits,
  no expert pruning, no inference-code changes. The optimized suffix also transfers zero-shot
  across sibling models that share a routing backbone.

---

## Configuration

Everything is driven by [configs/base.yaml](configs/base.yaml):

- **Swap the target model** under `model:` — any MoE whose router/experts the `ArchSpec` can
  locate. Presets ship for **OLMoE**, **Mixtral**, and **Qwen** MoE; add a family by adding a
  preset in [model/archspec.py](src/routehijack/model/archspec.py) and the dims in the config.
- **`use_chat_template`** renders prompts through the instruct template so the boundary token
  `t*` is the real decision point. Auto-falls back to raw text if the tokenizer has none.
- **Attack budget** (`attacks.routehijack`): suffix length `T`, steps, the `λ` weights, the
  harmful-promotion margin. CLI flags on `scripts/02_routehijack.py` tune the search
  (candidates per step, prompt subsample, grad/candidate batch sizes, early-stop patience).

---

## Outputs

| Artifact | Phase | Contents |
|---|---|---|
| `artifacts/safety_experts.json`, `harmful_experts.json` | harvest | flagged `(layer, expert)` sets + scores |
| `artifacts/identify_diagnostics.pt` | harvest | per-expert score / frequency tensors |
| `artifacts/routehijack_universal.json` | routehijack | the optimized suffix |
| `artifacts/routehijack_attacks.jsonl` | routehijack | per-prompt attacked completions |
| `artifacts/routehijack_routing_shift.json` | routehijack | TESR / THPR routing-shift metrics |
| `artifacts/eval_cells.jsonl` | eval | clean vs attacked ASR, MMLU, verdict |
| `artifacts/transcripts/*.md` | eval | readable sample conversations |

**Metrics.** *ASR* — fraction of harmful prompts that get a non-refusal (string detector,
optionally re-graded by the HarmBench classifier with `--judge`). *MMLU* — multiple-choice
accuracy with the attack active, to confirm the model isn't merely broken. *TESR / THPR* —
how much routing mass shifts off safety experts / onto harmful experts at the boundary token.

---

## Performance & optimizations

The pipeline is tuned to keep the GPU busy and avoid recomputation. Everything below is
**quality-neutral** (identical results) unless flagged otherwise.

**Harvest (expert localization)**
- The activation-frequency sweeps **pre-tokenize** the whole corpus once, then run **batched**
  forwards (length-sorted to minimize padding) with on-GPU count accumulation, and call the
  base transformer (skipping the `lm_head`) since only router logits are needed.
- A **model-placement check** warns loudly when `device_map: auto` has offloaded layers to
  CPU/disk — the usual cause of 10–100× slow runs.

**Suffix optimization (the attack)**
- **One persistent hook manager** for the whole run (no per-forward hook install/remove).
- **Prefix-embedding cache**: each prompt's `[before]` (template + query) embeddings are
  computed once and reused across all optimization steps.
- **Batched candidate evaluation** (`--candidate-batch-size`): all candidates for a prompt are
  scored in a *single* batched forward instead of one-at-a-time — the dominant per-step cost.
- **Batched grad pass** (`--grad-batch-size`): prompts are processed in right-padded chunks,
  one forward+backward each; mathematically identical to per-prompt accumulation
  (∇ of a sum = sum of ∇s), just far better GPU utilization.
- **Decode-then-re-encode length filter** (Algorithm 1): candidates whose suffix re-tokenizes
  to a different length are rejected, so the optimized tokens survive deployment as text
  (this also prevents the failure mode where the deployed suffix differs from the optimized one).
- **Early stop** (`--early-stop-patience`): halts once the best loss plateaus.
- **Prefix KV-cache** (experimental, `--prefix-kv-cache`): the `[before]` prefix is fixed and
  shared across all candidates and all 300 steps. With this flag its KV cache is computed
  **once per prompt**, and candidate forwards process only `[suffix][after]` (~25 tokens),
  attending to the cached prefix instead of recomputing the full `[before][suffix][after]`.
  Quality-neutral; **self-checked** against the full path on first use and **auto-disabled** on
  any numeric mismatch or HF-version incompatibility, so it can never silently corrupt the attack.

**Evaluation**
- Completions are generated in **left-padded batches** (`--gen-batch-size`) via the model's own
  `generate`, rather than decoding one prompt at a time.

> The boundary token `t*` is placed correctly by rendering prompts through the chat template
> (see *How it works*); this is a correctness requirement the optimizations are built on, not a
> speed tweak.

---

## Layout

```
src/routehijack/
  model/      loader.py · archspec.py · hooks.py (router/expert capture + mutate) · prompting.py
  identify/   activation_freq.py (Eq. 3) · delta_s.py (Eq. 4–5) · select.py (top-pct)
  attacks/    routehijack.py (ternary loss + GCG search) · compose.py
  eval/       asr.py (RefusalDetector + HarmBench) · mmlu.py · generate.py · harness.py
  config.py · data.py · ui.py
scripts/      00_data.py · 01_harvest.py · 02_routehijack.py · 03_eval.py
configs/      base.yaml
```

---

## Note on responsible use

This is a red-teaming / safety-evaluation tool for measuring how susceptible open-weight MoE
models are to routing-level manipulation. Use it to audit models you are authorized to test.
