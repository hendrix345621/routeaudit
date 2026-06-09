"""#3-MoE (EXPERIMENTAL) — distill a judge into a routing-feature harm probe.

Offline, one-time: for each prompt, capture the boundary-token router features AND
generate a response; label every response with the judge (Llama Guard); train a probe
(routing features → P(harmful)). The probe then plugs into the attack as a fast,
differentiable, judge-aware term (see attacks/harm_probe.py + the README roadmap).

    python scripts/distill_harm_probe.py --config qwen3.6 --judge-kind llamaguard \
        --judge-id meta-llama/Llama-Guard-3-1B --n-prompts 200 --out artifacts/harm_probe.pt

NOTE: for a useful probe you need BOTH harmful and safe examples. Clean AdvBench
generations are mostly refusals (safe) → imbalanced. Add variety with --temperature
and --n-samples (sample several responses per prompt), or seed with known-jailbroken
suffixes. This is a research scaffold: it runs, but expect to iterate on the data mix.
"""
from __future__ import annotations

import argparse

import torch

from routehijack import config as cfg_mod
from routehijack import ui
from routehijack.attacks.harm_probe import boundary_routing_features, save_probe, train_probe
from routehijack.data import read_jsonl
from routehijack.eval.asr import score_with_classifier
from routehijack.eval.generate import generate_batch
from routehijack.model import load_model
from routehijack.model.hooks import MoEHookManager
from routehijack.model.prompting import encode_prompt


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--advbench", default="data/advbench.jsonl")
    p.add_argument("--n-prompts", type=int, default=200)
    p.add_argument("--n-samples", type=int, default=1, help="responses sampled per prompt (variety)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--judge-id", default="meta-llama/Llama-Guard-3-1B")
    p.add_argument("--judge-kind", default="llamaguard")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--gen-batch-size", type=int, default=8)
    p.add_argument("--out", default="artifacts/harm_probe.pt")
    args = p.parse_args()

    cfg = cfg_mod.load(args.config)
    ui.big_banner("Distill harm probe (#3-MoE, experimental)")
    loaded = load_model(cfg)
    model, tok, spec = loaded.model, loaded.tokenizer, loaded.spec
    use_tmpl = getattr(cfg.model, "use_chat_template", True)
    device = next(model.parameters()).device

    prompts = [r["prompt"] for r in list(read_jsonl(args.advbench))[: args.n_prompts]]
    do_sample = args.n_samples > 1 or args.temperature != 1.0

    feats: list[torch.Tensor] = []
    responses: list[str] = []
    for _ in range(args.n_samples):
        # 1) capture boundary routing features per prompt (one hooked forward each)
        for pr in ui.iter_with_progress(prompts, "route features"):
            ids = encode_prompt(tok, pr, want_template=use_tmpl, device=device).unsqueeze(0)
            with MoEHookManager(model, spec) as hm, torch.no_grad():
                hm.capture_router_logits()
                model(input_ids=ids, use_cache=False)
                rl = {l: v.view(ids.shape[1], -1) for l, v in hm.capture.router_logits.items()}
            feats.append(boundary_routing_features(rl, boundary=ids.shape[1] - 1).float().cpu())
        # 2) generate responses for the same prompts
        responses += generate_batch(model, tok, prompts, max_new_tokens=args.max_new_tokens,
                                    do_sample=do_sample, temperature=args.temperature,
                                    batch_size=args.gen_batch_size, want_template=use_tmpl,
                                    desc="generate")

    # 3) label with the judge → 1 if harmful
    ui.section(f"Labelling {len(responses)} responses with {args.judge_id}")
    labels = score_with_classifier(prompts * args.n_samples, responses,
                                   judge_hf_id=args.judge_id, judge_kind=args.judge_kind, device="cuda")
    y = torch.tensor([1.0 if h else 0.0 for h in labels])
    pos = int(y.sum())
    ui.kv_panel("Distillation set", {"n": len(y), "harmful": pos, "safe": len(y) - pos})
    if pos == 0 or pos == len(y):
        ui.warn("only one class present — the probe can't learn. Add --n-samples / --temperature "
                "(or seed jailbroken suffixes) to get both harmful and safe examples.")

    X = torch.stack(feats)
    res = train_probe(X, y)
    save_probe(res.probe, args.out)
    ui.ok(f"probe trained (loss={res.final_loss:.4f}, train_acc={res.train_acc:.3f}) → {args.out}")
    ui.info("Next: load it in the attack as a differentiable term (probe_loss) for judge-aware "
            "gradients — see attacks/harm_probe.py and the README roadmap.")
    ui.print_done("Harm-probe distillation complete (experimental).")


if __name__ == "__main__":
    main()
