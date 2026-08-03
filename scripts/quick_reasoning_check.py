"""Cheap GPU smoke test for thinking-mode generation and answer segmentation.

This deliberately avoids datasets, a safety judge, expert harvesting, and suffix
optimization. It answers two tiny deterministic questions, but samples with Qwen's
recommended thinking-mode settings. Passing proves that the mode switch, closing
delimiter, token-level answer extraction, and truncation accounting work together.
It is not a reasoning benchmark or a quantitative router experiment.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from routeaudit import config as cfg_mod
from routeaudit import ui
from routeaudit.eval.asr import answers_from_ids
from routeaudit.eval.generate import generate_batch_ids
from routeaudit.model import load_model, prompting
from routeaudit.model.thinking import TRUNCATED, Anchor, ThinkSpec, audit_format

CASES = [
    {
        "prompt": (
            "Calculate 17 multiplied by 23. Reason carefully, then put only the final "
            "number in the final answer."
        ),
        "answer_pattern": r"\b391\b",
    },
    {
        "prompt": (
            "All bloops are razzies. No razzies are green. Can any bloop be green? "
            "Reason carefully, then begin the final answer with Yes or No."
        ),
        "answer_pattern": r"^\s*no\b",
    },
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    target = ap.add_mutually_exclusive_group()
    target.add_argument(
        "--model",
        default=None,
        help="direct Hugging Face model id (default: Qwen/Qwen3-4B, the cheapest smoke target)",
    )
    target.add_argument(
        "--config",
        help="RouteAudit config/nickname; use qwen3-think-smoke for the 30B MoE FP8 check",
    )
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("artifacts/quick_reasoning.json"))
    args = ap.parse_args()

    if args.max_new_tokens < 32:
        ap.error("--max-new-tokens must be at least 32")

    ui.section("loading reasoning smoke target")
    if args.config:
        cfg = cfg_mod.load(args.config)
        cfg.model.enable_thinking = True
        loaded = load_model(cfg)
        model, tokenizer = loaded.model, loaded.tokenizer
        model_id = cfg.model.hf_id
    else:
        model_id = args.model or "Qwen/Qwen3-4B"
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    # Force the request after loading as well, so a reused config cannot silently leave
    # Qwen in its non-thinking setting. The generated format remains the source of truth.
    prompting.set_chat_template_kwargs({"enable_thinking": True})
    torch.manual_seed(args.seed)

    spec = ThinkSpec.from_tokenizer(tokenizer)
    if not spec.available:
        ui.warn("tokenizer has no single-token </think> delimiter; using the weaker text fallback")

    prompts = [case["prompt"] for case in CASES]
    gen_ids = generate_batch_ids(
        model,
        tokenizer,
        prompts,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        batch_size=args.batch_size,
        desc="thinking smoke",
    )
    answers, anchors = answers_from_ids(tokenizer, gen_ids, spec=spec)
    if spec.available:
        # Qwen's template can place <think> in the prompt rather than in generated ids.
        # In that layout a budget-exhausted trace contains neither generated <think> nor
        # </think>, and the generic parser cannot distinguish it from plain generation.
        # We know thinking was requested here, so absence of the closing delimiter is
        # conservatively unscoreable rather than treating private deliberation as answer.
        for i, ids in enumerate(gen_ids):
            if spec.close_id not in ids:
                anchors[i] = Anchor(None, (0, len(ids)), TRUNCATED, len(ids))
                answers[i] = ""
    audit = audit_format(anchors, requested_thinking=True)

    results = []
    for case, ids, answer, anchor in zip(CASES, gen_ids, answers, anchors):
        answer_ok = bool(anchor.scoreable and re.search(case["answer_pattern"], answer, re.IGNORECASE))
        results.append(
            {
                "prompt": case["prompt"],
                "raw_completion": tokenizer.decode(ids, skip_special_tokens=False),
                "answer": answer,
                "status": anchor.status,
                "think_tokens": anchor.think_len,
                "generated_tokens": anchor.n_generated,
                "scoreable": anchor.scoreable,
                "answer_ok": answer_ok,
            }
        )

    passed = audit.passed and all(row["scoreable"] and row["answer_ok"] for row in results)
    payload = {
        "model": model_id,
        "requested_thinking": True,
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
        "max_new_tokens": args.max_new_tokens,
        "format_audit": {
            "passed": audit.passed,
            "message": audit.message(),
            "trace_rate": audit.trace_rate,
            "truncation_rate": audit.truncation_rate,
        },
        "cases": results,
        "passed": passed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    (ui.ok if audit.passed else ui.fail)(audit.message())
    for i, row in enumerate(results, 1):
        marker = ui.ok if row["scoreable"] and row["answer_ok"] else ui.fail
        marker(
            f"case {i}: status={row['status']}, think={row['think_tokens']} tokens, "
            f"answer={row['answer'].strip()!r}"
        )
    ui.info(f"machine-readable result: {args.out}")

    if not passed:
        if any(not row["scoreable"] for row in results):
            ui.warn(
                "At least one trace did not close. Re-run once with --max-new-tokens 1024; "
                "do not treat the unfinished trace as an answer."
            )
        raise SystemExit(1)
    ui.print_done("Reasoning smoke PASSED")


if __name__ == "__main__":
    main()
