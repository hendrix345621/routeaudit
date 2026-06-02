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


def load_model(cfg) -> LoadedModel:
    dtype = _DTYPES[cfg.model.dtype]
    tok = AutoTokenizer.from_pretrained(cfg.model.hf_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.hf_id,
        torch_dtype=dtype,
        device_map=cfg.model.device_map,
        trust_remote_code=True,
    )
    model.eval()
    _report_placement(model)
    return LoadedModel(model=model, tokenizer=tok, cfg=cfg.model,
                       spec=ArchSpec.from_config(cfg.model))


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
