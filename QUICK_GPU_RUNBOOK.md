# Fast, low-cost GPU runbook

This runbook answers two narrow questions:

1. Does RouteAudit correctly request, detect, and segment reasoning output?
2. Does the real DeepSeek-V4-Flash checkpoint agree with the project's mHC/gate assumptions?

It is a smoke-test ladder, not the full safety experiment. Stop at the first failure so
you do not pay for a larger GPU while debugging ordinary code or environment problems.

## Recommended order

| Order | Check | Hardware | What a pass means | Stop condition |
|---|---|---|---|---|
| 0 | Unit tests + synthetic mHC | Local CPU, no rental | Project mechanisms and device paths work | Any test fails |
| 1 | Qwen3-4B reasoning smoke | 1 x 16–24 GB GPU | Thinking is emitted, closes, segments, and yields answers | Format, truncation, or answer check fails |
| 2 | Real DeepSeek-V4 fixture | Prefer 2 x 94/96 GB H100-class; >=180 GB aggregate VRAM | Shipped gate output and four-stream residual are compatible with the implementation | CPU/disk offload, missing capture, or fixture mismatch |

Do reasoning before real mHC. It uses a much smaller checkpoint and catches template,
generation, and answer-span bugs before the expensive rental. The synthetic mHC check is
real code coverage but **not** evidence that DeepSeek's released weights are compatible.

## 0. Free preflight (run before renting)

From the repository root:

```bash
python -m pip install -e '.[dev]'
pytest tests experiments/mhc/tests -q
python experiments/mhc/tests/run_synthetic.py
```

Pass gate: all tests pass and the synthetic script reports all Level 0 checks as passed.
If not, fix locally and do not rent a GPU yet.

## Vast.ai selection

Use an Ubuntu/PyTorch image with direct SSH. Select a **verified** host, reliability of at
least 0.99, a recent CUDA stack, fast download bandwidth, and adequate fixed disk space.
Sort by total dollars/hour, then compare download speed: model download and load time can
cost more than the tiny forwards in this runbook.

For the cheapest reasoning-only rental:

- 1 x 16–24 GB GPU; an RTX 3090/4090-class offer is plenty;
- 25 GB disk minimum; 40 GB is comfortable;
- use `Qwen/Qwen3-4B` in its normal BF16/automatic checkpoint dtype.

For the genuine mHC rental:

- fastest/safest simple choice: 2 x H100 NVL 94 GB (or another node with at least
  180 GB aggregate VRAM and a fast interconnect);
- 220 GB disk minimum; 250 GB is comfortable;
- do not add NF4/int8 quantization: DeepSeek-V4-Flash already ships as mixed FP4/FP8.

Nominal 2 x 80 GB exactly matches the checkpoint's roughly 160 GB file size and leaves
little room for runtime state. Use it only if the host/config is already known to load
fully on GPU; any CPU or disk offload defeats the speed goal and weakens the test setup.

## Common setup on a rented instance

Connect using the SSH command shown by Vast, then copy or clone the project. In the
project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
nvidia-smi
python -c "import torch, transformers; print(torch.__version__, transformers.__version__); print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Keep the shell alive with `tmux` if desired. Authenticate to Hugging Face directly on the
instance; do not send the token through chat or put it in a committed file.

## 1. Cheap reasoning smoke

Run:

```bash
python scripts/quick_reasoning_check.py \
  --max-new-tokens 512 \
  --batch-size 1 \
  --out artifacts/quick_reasoning.json
```

The script uses Qwen's recommended thinking sampling (`temperature=0.6`, `top_p=0.95`,
`top_k=20`) rather than greedy decoding. It passes only when both outputs contain a
completed reasoning trace, the post-`</think>` answer is scoreable, and the two trivial
answers are correct.

If it fails only because a trace was truncated, rerun once:

```bash
python scripts/quick_reasoning_check.py --max-new-tokens 1024
```

Do not keep raising the budget repeatedly. A missing trace or malformed output after that
is a feature/configuration failure to diagnose, not a reason to start the mHC rental.

This cheapest run proves the **model-independent thinking path**: chat-template switching,
token-level delimiter handling, truncation accounting, and answer extraction. It does not
exercise a MoE router. If it passes and you specifically want a Qwen MoE confirmation, run:

```bash
python scripts/quick_reasoning_check.py --config qwen3-think-smoke
```

That optional target is the official ~32.5 GB Qwen3-30B-A3B FP8 checkpoint and requires a
compute-capability 9+ GPU in the Transformers fine-grained FP8 path (normally an H100).
It is still a structural smoke result, not an exact router-score claim. Use `--config qwen3`
on an 80 GB GPU when you later need the 61.1 GB BF16 checkpoint for quantitative routing.

Download `artifacts/quick_reasoning.json`, then destroy the reasoning instance if using
the lowest-cost two-rental plan.

## 2. Genuine mHC compatibility check

On the larger instance, first repeat the free synthetic check. Then capture one short
forward from the real checkpoint and validate it:

```bash
python experiments/mhc/tests/run_synthetic.py
python experiments/mhc/fixtures/extract.py \
  --config deepseek-v4-flash \
  --out artifacts/v4_flash_fixtures.pt
python experiments/mhc/fixtures/validate.py \
  --fixtures artifacts/v4_flash_fixtures.pt
```

Watch the loader output. The run is not a speed-valid GPU test if it warns that layers
were offloaded to CPU or disk.

Pass gate:

- fixture extraction captures `gate` and `residual` entries;
- validation reports the same selected expert set and matching gate weights;
- residual validation records four streams and a valid stream-mean reduction;
- the command exits zero.

A missing hash fixture may be reported as skipped if the released module does not expose
its table. Record that as an unverified hash-routing sub-check; do not call it a pass.
Likewise, this fixture check validates compatibility, not the full semantic safety claim.

Copy out `artifacts/v4_flash_fixtures.pt` immediately. It is small compared with the model
cache and is all you need for repeat CPU-side validation.

## Cheapest versus fastest rental strategy

**Lowest expected spend:** use two rentals. Run Qwen3-4B on one cheap 16–24 GB GPU,
destroy it after the JSON is copied, and rent the >=180 GB aggregate node only for the
one-forward DeepSeek fixture. Choose on-demand for the expensive first attempt; an
interruptible instance is sensible only after the procedure has already succeeded and
you know the run can restart cleanly.

**Fastest/simple administration:** use one 2 x 94/96 GB node with about 250 GB disk. Run
the Qwen smoke first, unload/exit that process, then run the DeepSeek fixture. This avoids
a second setup but pays for an idle second GPU during the Qwen step, so it is usually not
the cheapest.

Do not run harvesting, AdvBench, a judge, MMLU, or suffix optimization in this smoke
session. Those are full experiments and multiply GPU time without answering the two
compatibility questions above.

## Cost shutdown checklist

1. Confirm both artifact files exist and are non-empty.
2. Copy them off the instance.
3. Record GPU model/count, CUDA, PyTorch, Transformers, checkpoint revision, and command.
4. **Destroy** the Vast instance when finished. Stopping ends compute billing but storage
   charges continue; destroying ends both and deletes the instance data.

Useful primary references:

- [Qwen3-30B-A3B-FP8 model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8)
- [Qwen3-4B model card](https://huggingface.co/Qwen/Qwen3-4B)
- [Transformers fine-grained FP8 hardware requirements](https://huggingface.co/docs/transformers/quantization/finegrained_fp8)
- [DeepSeek-V4-Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [Vast.ai offer filters](https://docs.vast.ai/api-reference/search/search-offers)
- [Vast.ai instance lifecycle](https://docs.vast.ai/guides/instances/manage-instances)
