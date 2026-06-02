"""Response-driven expert activation frequency.

Implements RouteHijack Eq. 3 (paper p. 4):

    F_l(e | a) = (1/|a|) · Σ_t  𝟙[e ∈ TopK(logits_{l,t})]

`a` is a response sequence. Query/prompt tokens are masked out per the paper's
response-driven profiling (RouteHijack p. 5: response-driven beats prompt-driven,
69.3% vs 30.5% ASR — masking matters).

Performance notes
─────────────────
The naive version ran one sequence per forward (batch size 1): the GPU finished
each tiny forward in milliseconds, then sat idle while Python set up the next one
— latency-bound, low GPU utilisation. The reworked version:

  - **Pre-tokenizes everything** before touching the GPU (CPU work happens once).
  - **Batches `batch_size` sequences per forward**, right-padded to the batch's
    longest sequence, with an attention mask so real tokens never attend to pads.
    Router logits come back flattened (B*T, E); we reshape to (B, T, E) and mask
    out prompt + padding positions per sequence. This is the big win — it keeps
    the GPU busy and cuts wall-time ~10× on short sequences. Right-padding + the
    attention mask yields per-position logits identical to the batch-1 path, so
    the frequencies are unchanged — only speed differs.
  - **Sorts by length** before batching so each batch pads to a similar size
    (minimal wasted compute on padding).
  - **Reuses one hook manager** across the whole sweep.
  - **Accumulates counts on the GPU**, syncing to CPU only at the end.
  - **Truncates responses** to `max_response_tokens` so the long tail doesn't
    dominate wall-time.

Memory cost: peak VRAM scales with `batch_size * max_total_tokens * n_experts`
(a bool top-k mask, processed one layer at a time) — a few MB at the defaults.
Lower `batch_size` if VRAM is tight."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .. import ui
from ..model.hooks import MoEHookManager


@dataclass
class ExpertFreq:
    """Per-(layer, expert) activation frequency over a corpus.

    Stored as a dense (n_layers, n_experts) float tensor on CPU."""

    freq: torch.Tensor  # (L, E), float64
    n_tokens: int

    def __getitem__(self, key: tuple[int, int]) -> float:
        layer, expert = key
        return float(self.freq[layer, expert])


@torch.no_grad()
def compute_expert_freq(
    model: torch.nn.Module,
    tokenizer,
    sequences: Iterable[dict],
    *,
    n_layers: int,
    n_experts: int,
    top_k: int,
    device: str | torch.device | None = None,
    desc: str = "freq",
    max_response_tokens: int | None = 256,
    max_total_tokens: int = 1024,
    batch_size: int = 16,
    spec=None,
    use_chat_template: bool = True,
) -> ExpertFreq:
    """Compute F_l(e | a) over a corpus of sequences.

    Each sequence is a dict with:
      - 'prompt'   : str, query text (its tokens are MASKED)
      - 'response' : str, response text (its tokens are COUNTED)

    Args:
      max_response_tokens: truncate the response to at most this many tokens.
                           Most refusals / completions don't need >256 tokens to
                           characterise routing. Set None to disable.
      max_total_tokens:   cap prompt+response combined. Prevents one pathological
                           long sequence from dominating wall-time.
      batch_size:         sequences per forward pass. Higher = better GPU
                          utilisation; lower if VRAM is tight.
    """
    device = device or next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    # ── 1) Pre-tokenize everything on CPU before touching the GPU. ──
    # The tokenizer call is a hidden cost; doing it inside the GPU loop would
    # leave the GPU idle. Pull it out so the GPU gets a steady stream of work.
    # Each entry is (ids[CPU,long], n_prompt) — kept on CPU; batched onto the GPU below.
    from ..model.prompting import profiling_ids

    prepped: list[tuple[torch.Tensor, int]] = []
    for item in ui.iter_with_progress(list(sequences), desc=f"{desc} (tokenize)"):
        prompt = item["prompt"]
        response = item["response"]
        if not response:
            continue
        # Render the query through the chat template (query + assistant marker are
        # the CONTEXT to mask); count only the response content tokens (Eq. 3, §5.1).
        full_ids, n_prompt = profiling_ids(tokenizer, prompt, response,
                                           want_template=use_chat_template)
        if full_ids.shape[0] <= n_prompt:
            continue
        if max_response_tokens is not None:
            full_ids = full_ids[: n_prompt + max_response_tokens]
        if full_ids.shape[0] > max_total_tokens:
            # Drop from the head of the response, not the prompt — the prompt
            # carries the query that the response refers to.
            keep = max_total_tokens - n_prompt
            if keep <= 0:
                continue
            full_ids = torch.cat([full_ids[:n_prompt], full_ids[n_prompt:n_prompt + keep]])
        prepped.append((full_ids, n_prompt))

    if not prepped:
        raise RuntimeError("No valid sequences after tokenization.")

    # Sort longest-first so each batch pads to a similar length (minimal waste).
    # Counting order is irrelevant — we only accumulate sums.
    prepped.sort(key=lambda p: p[0].shape[0], reverse=True)
    batches = [prepped[i:i + batch_size] for i in range(0, len(prepped), batch_size)]

    # ── 2) GPU-resident accumulators. Sync to CPU only at the end. ──
    counts = torch.zeros(n_layers, n_experts, dtype=torch.float64, device=device)
    total_tokens = torch.zeros((), dtype=torch.float64, device=device)

    # ── 3) One persistent hook manager; one forward per batch. ──
    # Call the BASE transformer (model.model), not the causal-LM wrapper: the router
    # hooks live inside the decoder layers, so we get identical router logits while
    # skipping the lm_head, whose (B, T, vocab) logits tensor is a large VRAM spike
    # we never use. Falls back to the full model if there's no `.model` attribute.
    fwd = getattr(model, "model", model)
    with MoEHookManager(model, spec) as hm:
        hm.capture_router_logits()

        for batch in ui.iter_with_progress(batches, desc=desc):
            B = len(batch)
            lens = [ids.shape[0] for ids, _ in batch]
            T_pad = max(lens)

            input_ids = torch.full((B, T_pad), pad_id, dtype=torch.long)
            attn = torch.zeros((B, T_pad), dtype=torch.long)
            n_prompts = torch.empty(B, dtype=torch.long)
            real_lens = torch.empty(B, dtype=torch.long)
            for b, (ids, n_prompt) in enumerate(batch):
                L = ids.shape[0]
                input_ids[b, :L] = ids
                attn[b, :L] = 1
                n_prompts[b] = n_prompt
                real_lens[b] = L
            input_ids = input_ids.to(device)
            attn = attn.to(device)

            fwd(input_ids=input_ids, attention_mask=attn, use_cache=False)

            # Response-token mask: positions in [n_prompt, real_len) per sequence.
            ar = torch.arange(T_pad, device=device).unsqueeze(0)            # (1, T_pad)
            count_mask = (ar >= n_prompts.to(device).unsqueeze(1)) & \
                         (ar < real_lens.to(device).unsqueeze(1))           # (B, T_pad)

            for layer, logits in hm.capture.router_logits.items():
                E = logits.shape[-1]
                lg = logits.view(B, T_pad, E)                              # un-flatten B*T
                _, idx = lg.topk(top_k, dim=-1)
                tk = torch.zeros_like(lg, dtype=torch.bool)
                tk.scatter_(-1, idx, True)                                 # (B, T_pad, E) top-k mask
                # Zero out prompt + padding positions, then sum over (B, T_pad).
                counts[layer] += (tk & count_mask.unsqueeze(-1)).sum(dim=(0, 1)).double()

            total_tokens += count_mask.sum().double()

    n_resp = int(total_tokens.item())
    if n_resp == 0:
        raise RuntimeError("No response tokens were counted — check your sequences.")
    # Single CPU sync at the very end.
    freq = (counts / float(n_resp)).cpu()
    return ExpertFreq(freq=freq, n_tokens=n_resp)
