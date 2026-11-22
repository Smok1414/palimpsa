"""Leakage canary (T5): no causal model can beat ln(V) on i.i.d. uniform tokens.

The positive control feeds a deliberately leaky match channel that marks every earlier position holding
the *next* token. A model that can use the match table must then fall well below ln(V); if it cannot, the
canary would not detect a real leak and the honest result means nothing.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .model import Config, PalimpsaLM
from .symbolic import causal_mask

ALPHABET = 64
LN_ALPHABET = math.log(ALPHABET)


def leaky_ell(x: torch.Tensor) -> torch.Tensor:
    """ell[b, t, j] = 1 if x[b, j] == x[b, t + 1] and j <= t: points straight at the target (a future leak)."""
    length = x.shape[1]
    nxt = torch.roll(x, -1, dims=1)
    leak = (nxt[:, :, None] == x[:, None, :]) & causal_mask(length, x.device)
    leak[:, -1] = False
    return leak.to(torch.uint8)


def leak_canary(variant: str = "symA", leaky: bool = False, steps: int = 400, seed: int = 0,
                device: str | None = None, batch: int = 32, length: int = 128) -> float:
    """Train a tiny model on uniform tokens and return the held-out next-token loss in nats."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    cfg = Config(vocab=ALPHABET, d_model=128, n_layers=2, n_heads=2, n_kv_heads=2, head_dim=64, d_ff=256,
                 variant=variant)
    model = PalimpsaLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.0)
    gen = torch.Generator().manual_seed(seed + 1)

    def step_loss(x: torch.Tensor) -> torch.Tensor:
        logits = model(x, ell=leaky_ell(x) if leaky else None)
        return F.cross_entropy(logits[:, :-1].reshape(-1, ALPHABET).float(), x[:, 1:].reshape(-1))

    model.train()
    for _ in range(steps):
        x = torch.randint(0, ALPHABET, (batch, length), generator=gen).to(device)
        loss = step_loss(x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        losses = [step_loss(torch.randint(0, ALPHABET, (batch, length), generator=gen).to(device)).item()
                  for _ in range(8)]
    return sum(losses) / len(losses)
