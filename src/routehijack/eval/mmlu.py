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
) -> float:
    """`questions` items: {question, choices: [4 strings], answer: int 0..3}."""
    device = device or next(model.parameters()).device
    letter_ids = [tokenizer(" " + L, add_special_tokens=False).input_ids[-1] for L in "ABCD"]

    correct, total = 0, 0
    for q in questions:
        prompt = PROMPT_TEMPLATE.format(
            q=q["question"], a=q["choices"][0], b=q["choices"][1],
            c=q["choices"][2], d=q["choices"][3],
        )
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with MoEHookManager(model, spec) as hm:
            if defense.router_mutator is not None:
                hm.set_router_mutator(defense.router_mutator)
            for (layer, expert), fn in defense.expert_mutators.items():
                hm.set_expert_mutator(layer, expert, fn)
            logits = model(input_ids=ids, use_cache=False).logits[0, -1]

        scores = torch.stack([logits[i] for i in letter_ids])
        pred = int(scores.argmax().item())
        correct += int(pred == q["answer"])
        total += 1
    return correct / max(1, total)
