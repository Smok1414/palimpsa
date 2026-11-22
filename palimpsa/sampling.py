"""Sampling from PalimpsaLM and repetition metrics for generated text.

generate(): batched sampling for prompts of equal length. Each prompt is preceded by EOS so it starts a new
document (as in training); no KV cache, every step re-runs the model on the last `context` tokens. Supports
temperature (0 = greedy), top-k, top-p and the CTRL repetition penalty (Keskar et al. 2019, as in HF transformers):
the logit of every token already present in the prompt or the continuation is divided by the penalty if positive and
multiplied by it if negative.

distinct_n() and repeated_tail() measure degeneration of a continuation.
"""
from __future__ import annotations

import torch

from .loader import EOS_ID


def _doc_ids(x: torch.Tensor) -> torch.Tensor:
    ids = torch.zeros_like(x)
    ids[:, 1:] = torch.cumsum(x[:, :-1] == EOS_ID, dim=1)
    return ids


def apply_repetition_penalty(logits: torch.Tensor, seen: torch.Tensor, penalty: float) -> torch.Tensor:
    """logits, seen: (B, V); seen marks token ids already in the context."""
    if penalty == 1.0:
        return logits
    penalised = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(seen, penalised, logits)


def _filter(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, order = torch.sort(logits, descending=True, dim=-1)
        probs = sorted_logits.softmax(-1)
        drop = probs.cumsum(-1) - probs > top_p  # keep the smallest prefix whose mass reaches top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, order, sorted_logits)
    return logits


@torch.no_grad()
def generate(model, prompts: list[list[int]], max_new: int, temperature: float = 0.8, top_k: int = 0,
             top_p: float = 1.0, repetition_penalty: float = 1.0, context: int = 1024,
             generator: torch.Generator | None = None, amp: bool = True) -> list[list[int]]:
    """Continuations (token ids, cut at the first EOS) for equal-length prompts."""
    if len({len(p) for p in prompts}) != 1:
        raise ValueError("generate() needs prompts of equal length; group them by length")
    device = next(model.parameters()).device
    vocab = model.cfg.vocab
    cur = torch.tensor([[EOS_ID] + p for p in prompts], device=device)
    seen = torch.zeros(len(prompts), vocab, dtype=torch.bool, device=device)
    seen.scatter_(1, cur[:, 1:], True)
    done = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    for _ in range(max_new):
        x = cur[:, -context:]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(x, _doc_ids(x), last_only=True)[:, -1].float()
        logits = apply_repetition_penalty(logits, seen, repetition_penalty)
        if temperature <= 0:
            nxt = logits.argmax(-1)
        else:
            probs = _filter(logits / temperature, top_k, top_p).softmax(-1)
            nxt = torch.multinomial(probs.cpu(), 1, generator=generator).squeeze(1).to(device)
        nxt = torch.where(done, torch.full_like(nxt, EOS_ID), nxt)
        cur = torch.cat([cur, nxt[:, None]], dim=1)
        seen.scatter_(1, nxt[:, None], True)
        done |= nxt == EOS_ID
        if bool(done.all()):
            break
    outs = []
    for row in cur[:, 1 + len(prompts[0]):].tolist():
        outs.append(row[: row.index(EOS_ID)] if EOS_ID in row else row)
    return outs


def distinct_n(tokens: list[int], n: int = 4) -> float | None:
    """Unique n-grams / all n-grams of the sequence (None if it has no n-gram)."""
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams) if grams else None


def repeated_tail(tokens: list[int], max_period: int = 40) -> float:
    """Share of the sequence covered by its longest periodic tail that repeats at least twice: the largest
    (count + p) / len over periods p <= max_period, where the last `count` tokens satisfy t[i] == t[i - p] and
    count >= p. 0 if the sequence does not end in a loop."""
    n, best = len(tokens), 0
    for p in range(1, min(max_period, n // 2) + 1):
        count = 0
        while count < n - p and tokens[n - 1 - count] == tokens[n - 1 - count - p]:
            count += 1
        if count >= p:
            best = max(best, count + p)
    return best / n if n else 0.0
