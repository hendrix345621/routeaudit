"""MMLU multiple-choice log-prob accuracy.

Used to confirm that an attack-induced perturbation (e.g. ablating refusal
features via SAE inversion) doesn't tank utility — so a maintainer can tell a
real vulnerability (harm elicited, utility intact) from a model that was merely
broken by the attack.
"""
from __future__ import annotations

from typing import Iterable

import torch

from ..model.hooks import MoEHookManager
from .generate import DefenseBundle


PROMPT_TEMPLATE = "Question: {q}\nA) {a}\nB) {b}\nC) {c}\nD) {d}\nAnswer:"


@torch.no_grad()
def mmlu_logprob_accuracy(
    model,
    tokenizer,
    questions: Iterable[dict],
    *,
    defense: DefenseBundle = DefenseBundle(),
    device=None,
    spec=None,
    batch_size: int = 16,
) -> float:
    """`questions` items: {question, choices: [4 strings], answer: int 0..3}.

    Batched for GPU utilisation: questions are RIGHT-padded and run in chunks, and
    we read each row's last *real* token (index = attention_mask.sum-1). With right
    padding the real tokens keep positions 0..L-1, so this is numerically identical
    to scoring each question alone — just far fewer forward launches.
    """
    device = device or next(model.parameters()).device
    letter_ids = [tokenizer(" " + L, add_special_tokens=False).input_ids[-1] for L in "ABCD"]
    letter_ids_t = torch.tensor(letter_ids, device=device)

    qs = list(questions)
    prompts = [PROMPT_TEMPLATE.format(q=q["question"], a=q["choices"][0], b=q["choices"][1],
                                      c=q["choices"][2], d=q["choices"][3]) for q in qs]
    answers = [int(q["answer"]) for q in qs]

    prev_side = tokenizer.padding_side
    if tokenizer.pad_token_id is None:                 # decoder-only tokenizers often lack a pad token
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    correct = 0
    try:
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i:i + batch_size]
            enc = tokenizer(chunk, return_tensors="pt", padding=True).to(device)
            last_idx = enc["attention_mask"].sum(dim=1) - 1            # (B,) last real token per row

            with MoEHookManager(model, spec) as hm:
                if defense.router_mutator is not None:
                    hm.set_router_mutator(defense.router_mutator)
                for (layer, expert), fn in defense.expert_mutators.items():
                    hm.set_expert_mutator(layer, expert, fn)
                logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                               use_cache=False).logits                 # (B, T, vocab)

            rows = logits[torch.arange(logits.shape[0], device=device), last_idx]   # (B, vocab)
            preds = rows[:, letter_ids_t].argmax(dim=-1)               # (B,) in 0..3
            for j, pred in enumerate(preds.tolist()):
                correct += int(pred == answers[i + j])
    finally:
        tokenizer.padding_side = prev_side
    return correct / max(1, len(prompts))
