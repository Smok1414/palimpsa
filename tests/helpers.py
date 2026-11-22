"""Shared helpers for the T1-T6 tests."""
from __future__ import annotations

import torch

from palimpsa.model import Config, PalimpsaLM

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def tiny_config(variant: str, **overrides) -> Config:
    base = dict(vocab=32, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, head_dim=16, d_ff=128,
                variant=variant)
    base.update(overrides)
    return Config(**base)


@torch.no_grad()
def activate_channels(model: PalimpsaLM, seed: int = 0) -> PalimpsaLM:
    """Random non-zero forget weights and match table so that tests exercise both channels."""
    gen = torch.Generator().manual_seed(seed)
    for block in model.blocks:
        attn = block.attn
        if attn.wf is not None:
            attn.wf.weight.copy_(0.3 * torch.randn(attn.wf.weight.shape, generator=gen))
        if attn.b_free is not None:
            attn.b_free.copy_(2.0 * torch.randn(attn.b_free.shape, generator=gen))
    return model


def random_docs(bsz: int, length: int, n_docs: int, gen: torch.Generator) -> torch.Tensor:
    """Monotone document ids with random boundaries."""
    starts = torch.zeros(bsz, length, dtype=torch.long)
    for b in range(bsz):
        cut = torch.randperm(length - 1, generator=gen)[: n_docs - 1] + 1
        starts[b, cut] = 1
    return starts.cumsum(1)
