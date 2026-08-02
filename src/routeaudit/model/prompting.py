"""Chat-template-aware prompt assembly, shared by expert profiling, the
RouteHijack optimizer, and generation.

RouteHijack (arXiv 2605.02946) targets the pre-truncation router distribution at
the **boundary token** t* — "the last input token before autoregressive decoding
begins" (§4.2). For an instruction-tuned model that boundary is the final token of
the chat template's assistant-generation prompt, NOT the last token of the raw
query. Running the model without its chat template puts t* in the wrong place and
drives generation off-distribution, so every stage must render prompts through the
template. When the tokenizer has no chat template (a base model) we fall back to
raw text and the boundary collapses to the last query/suffix token.
"""
from __future__ import annotations

import torch

# Rare sentinel (RECORD SEPARATOR) used to locate where the adversarial suffix
# sits inside the rendered user turn. Must not occur in real prompts.
_SUFFIX_SLOT = "␞"


def has_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) is not None


def use_template(tokenizer, want: bool) -> bool:
    return bool(want) and has_chat_template(tokenizer)


# Extra kwargs forwarded to `apply_chat_template` for every render in this process.
# Set once (e.g. by the loader from `model.enable_thinking`). The key use is
# {'enable_thinking': False} on REASONING models (Qwen3 family): with thinking on,
# the boundary token t* is the start of a chain-of-thought, not the answer — which
# breaks expert localization, the L_refusal term, AND the refusal detector (it scores
# the thinking preamble, not the answer). Turning thinking off makes t* the real
# answer-decision token the attack/metrics assume.
_CHAT_TEMPLATE_KWARGS: dict = {}
_CHAT_KW_OK = True


def set_chat_template_kwargs(kw: dict | None) -> None:
    global _CHAT_KW_OK
    _CHAT_TEMPLATE_KWARGS.clear()
    _CHAT_TEMPLATE_KWARGS.update(kw or {})
    _CHAT_KW_OK = True


def render_user_turn(tokenizer, content: str, *, want_template: bool = True) -> str:
    """Render a single user turn as the string actually fed to the model, with the
    assistant generation prompt appended. Raw `content` if no template."""
    if not use_template(tokenizer, want_template):
        return content
    global _CHAT_KW_OK
    msgs = [{"role": "user", "content": content}]
    if _CHAT_TEMPLATE_KWARGS and _CHAT_KW_OK:
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, **_CHAT_TEMPLATE_KWARGS)
        except TypeError:
            _CHAT_KW_OK = False   # this template rejects the kwargs; fall back (once)
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def encode_prompt(tokenizer, content: str, *, want_template: bool = True,
                  device=None) -> torch.Tensor:
    """Token ids for a full user turn (template + generation prompt, or raw)."""
    templated = use_template(tokenizer, want_template)
    s = render_user_turn(tokenizer, content, want_template=want_template)
    # The template already injects BOS / special tokens; don't double them.
    ids = tokenizer(s, add_special_tokens=not templated).input_ids
    out = torch.tensor(ids, dtype=torch.long)
    return out.to(device) if device is not None else out


def profiling_ids(tokenizer, query: str, response: str, *, want_template: bool = True):
    """Token ids + response span for response-driven expert profiling (Eq. 3).

    Returns (full_ids, n_context) where positions [n_context, len) are the response
    tokens to COUNT and [0, n_context) is the query + chat-template special tokens
    to MASK (RouteHijack §5.1). The response is appended without special tokens so
    only its content tokens are counted."""
    ctx = encode_prompt(tokenizer, query, want_template=want_template)        # query + assistant marker
    resp = torch.tensor(tokenizer(response, add_special_tokens=False).input_ids, dtype=torch.long)
    full = torch.cat([ctx, resp])
    return full, int(ctx.shape[0])


def suffix_slot_ids(tokenizer, query: str, *, want_template: bool = True, device=None):
    """Return (before_ids, after_ids) bracketing the adversarial suffix so the full
    input is `before_ids ++ <suffix> ++ after_ids` with the suffix at the end of the
    user content, just before the assistant generation prompt. The boundary token
    (last of `after_ids`) is the routing decision point t* (§4.2).

    Without a chat template the suffix sits at the very end (`after_ids` empty) and
    t* is the last suffix token — the original behavior."""
    if use_template(tokenizer, want_template):
        rendered = render_user_turn(tokenizer, f"{query} {_SUFFIX_SLOT}", want_template=True)
        left, _, right = rendered.partition(_SUFFIX_SLOT)
        before = tokenizer(left, add_special_tokens=False).input_ids
        after = tokenizer(right, add_special_tokens=False).input_ids
    else:
        before = tokenizer(query, add_special_tokens=True).input_ids
        after = []
    b = torch.tensor(before, dtype=torch.long)
    a = torch.tensor(after, dtype=torch.long)
    if device is not None:
        b, a = b.to(device), a.to(device)
    return b, a
