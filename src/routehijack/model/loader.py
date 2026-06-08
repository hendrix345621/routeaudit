"""MoE model loader. Architecture-agnostic: the module layout is described by an
:class:`ArchSpec` (attached to the returned :class:`LoadedModel`) and consumed by
the hooks in `hooks.py`. Presets exist for OLMoE and Mixtral."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .. import ui
from .archspec import ArchSpec


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    cfg: SimpleNamespace  # the model config slice, not the global config
    spec: ArchSpec        # how to reach the router/experts for this family


_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _coerce_max_memory(mm):
    """`model.load.max_memory` arrives as a SimpleNamespace (YAML dict → ns) or dict.
    transformers wants a plain dict keyed by device int / "cpu" → "78GiB"."""
    if mm is None:
        return None
    items = vars(mm).items() if isinstance(mm, SimpleNamespace) else dict(mm).items()
    out = {}
    for k, v in items:
        key = int(k) if str(k).lstrip("-").isdigit() else str(k)   # "0"→0, "cpu"→"cpu"
        out[key] = v
    return out or None


def _load_opts(model_ns) -> dict:
    """Optional `model.load:` block — bf16-only fit/speed knobs. NO quantization
    (it would perturb the router logits harvest localizes and the attack depends on)."""
    lo = getattr(model_ns, "load", None)
    g = (lambda k, d=None: getattr(lo, k, d)) if lo is not None else (lambda k, d=None: d)
    return dict(
        attn_implementation=g("attn_implementation", "sdpa"),
        max_memory=_coerce_max_memory(g("max_memory")),
        offload_folder=g("offload_folder"),
        offload_state_dict=bool(g("offload_state_dict", False)),
    )


def load_model(cfg) -> LoadedModel:
    dtype = _DTYPES[cfg.model.dtype]
    tok = AutoTokenizer.from_pretrained(cfg.model.hf_id, trust_remote_code=True)
    opts = _load_opts(cfg.model)

    kwargs = dict(torch_dtype=dtype, device_map=cfg.model.device_map, trust_remote_code=True)
    if opts["max_memory"]:
        kwargs["max_memory"] = opts["max_memory"]
    if opts["offload_folder"]:
        kwargs["offload_folder"] = opts["offload_folder"]
        kwargs["offload_state_dict"] = opts["offload_state_dict"]

    impl = opts["attn_implementation"]
    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model.hf_id, attn_implementation=impl, **kwargs)
        if impl:
            ui.info(f"attention impl: {impl}")
    except (TypeError, ValueError) as e:
        # Some trust_remote_code models don't accept attn_implementation — fall back.
        ui.warn(f"attn_implementation='{impl}' rejected ({type(e).__name__}); loading default.")
        model = AutoModelForCausalLM.from_pretrained(cfg.model.hf_id, **kwargs)

    model.eval()
    _report_placement(model)

    # Honor chat-template options (e.g. enable_thinking: false on Qwen3 reasoning
    # models) across every phase that renders prompts.
    from . import prompting
    prompting.set_chat_template_kwargs(_chat_template_kwargs(cfg.model))

    return LoadedModel(model=model, tokenizer=tok, cfg=cfg.model,
                       spec=ArchSpec.from_config(cfg.model))


def _chat_template_kwargs(model_ns) -> dict:
    """Map `model.*` config into apply_chat_template kwargs. `enable_thinking: false`
    turns off a reasoning model's chain-of-thought; `chat_template_kwargs:` (a block)
    is a generic passthrough for anything else a template supports."""
    kw: dict = {}
    et = getattr(model_ns, "enable_thinking", None)
    if et is not None:
        kw["enable_thinking"] = bool(et)
    extra = getattr(model_ns, "chat_template_kwargs", None)
    if extra is not None:
        kw.update(vars(extra) if isinstance(extra, SimpleNamespace) else dict(extra))
    return kw


def enable_grad_checkpointing(model) -> bool:
    """Turn on gradient checkpointing for the white-box attack's backward pass:
    trades recompute for memory so a larger grad batch fits. Mathematically identical
    results. Must be OFF for generation (incompatible with use_cache). Returns success."""
    try:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False
        ui.info("gradient checkpointing: enabled (attack grad pass)")
        return True
    except Exception as e:  # noqa: BLE001
        ui.warn(f"gradient checkpointing unavailable ({type(e).__name__}: {e})")
        return False


def disable_grad_checkpointing(model) -> None:
    """Re-enable the KV cache after the attack so generation (eval) is fast again."""
    try:
        model.gradient_checkpointing_disable()
        if hasattr(model, "config"):
            model.config.use_cache = True
    except Exception:  # noqa: BLE001
        pass


def _report_placement(model) -> None:
    """Print where the model actually lives. The #1 cause of mysteriously slow
    stages is `device_map: auto` quietly spilling layers to CPU/disk when they
    don't fit in VRAM — every forward then shuttles activations over PCIe and
    runs 10-100× slower. Surface that loudly instead of letting it hide."""
    from collections import Counter

    dmap = getattr(model, "hf_device_map", None)
    if not dmap:
        ui.info(f"model placement: all on {next(model.parameters()).device}")
        return

    counts = Counter(str(v) for v in dmap.values())
    summary = "  ".join(f"{n}×{d}" for d, n in counts.items())
    offloaded = [d for d in counts if d == "cpu" or d.startswith("disk")]
    if offloaded:
        ui.warn(
            f"model is OFFLOADED across devices ({summary}). `device_map: auto` "
            "spilled part of the model off the GPU, so every forward pass copies "
            "activations over PCIe — this is the usual cause of 10-100× slow "
            "harvest / routehijack stages. Fix: fit the model on one GPU (a 24 GB+ "
            "card for OLMoE-1B-7B), or set `model.device_map` to a single device "
            "like \"cuda:0\". A bigger RAM disk does NOT help — this is VRAM, not disk."
        )
    else:
        ui.info(f"model placement: {summary} (fully on accelerator)")
