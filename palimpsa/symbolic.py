"""Symbolic match-length channel and the fused SymA attention bias.

ell[b, t, j] is the length (capped at ``n_max``) of the longest common suffix of x[b, :t+1] and x[b, :j].
Key j is therefore the position right after an earlier occurrence of the current suffix, so its value
carries the continuation of that occurrence. The additive attention bias is

    bias[b, h, t, j] = (F[b, h, t] - F[b, h, j]) + table[h, bucket(ell[b, t, j])]   for j <= t
    bias[b, h, t, j] = -inf                                                          for j > t

where F is the cumulative sum of log forget gates (FoX) and table[:, 0] == 0 (the model pads a zero
column in front of its learnable buckets).
"""
from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

N_MATCH = 16
N_BUCKETS = 8
# ell = 0..16 -> bucket: 0 | 1 | 2 | 3 | 4 | 5-7 | 8-15 | 16
BUCKET_OF_LEN = (0, 1, 2, 3, 4, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7)


def bucket_lut(device: torch.device | str) -> torch.Tensor:
    return torch.tensor(BUCKET_OF_LEN, dtype=torch.uint8, device=device)


def match_len(x: torch.Tensor, doc: torch.Tensor | None = None, n_max: int = N_MATCH) -> torch.Tensor:
    """Match lengths for keys of every query. x, doc: (B, L) integer tensors. Returns (B, L, L) uint8."""
    if x.dim() != 2:
        raise ValueError(f"expected (B, L) token ids, got shape {tuple(x.shape)}")
    if not 1 <= n_max <= len(BUCKET_OF_LEN) - 1:
        raise ValueError(f"n_max must be in [1, {len(BUCKET_OF_LEN) - 1}], got {n_max}")
    eq = x[:, :, None] == x[:, None, :]
    if doc is not None:
        eq &= doc[:, :, None] == doc[:, None, :]
    eq = torch.tril(eq, diagonal=-1)  # eq[t, i] = [x_t == x_i] for i < t
    run = eq.to(torch.uint8)
    active = eq.clone()
    for s in range(1, n_max):  # active[t, i] <=> common suffix ending at (t, i) is at least s + 1 long
        active[:, s:, s:] &= eq[:, :-s, :-s]
        active[:, :s] = False
        active[:, :, :s] = False
        if not bool(active.any()):
            break
        run += active.to(torch.uint8)
    ell = torch.zeros_like(run)
    ell[:, :, 1:] = run[:, :, :-1]  # key j continues the occurrence that ends at i = j - 1
    return ell


@dataclass
class SymIndex:
    """Per-batch structures shared by all layers: dense buckets and the sparse list of matches."""

    bucket: torch.Tensor  # (B, L, L) uint8, bucket of ell for every (b, t, j)
    flat: torch.Tensor  # (n * H,) int64, flat index into a (B, H, L, L) tensor of every match
    head_bucket: torch.Tensor  # (n * H,) int64, h * N_BUCKETS + bucket for every match
    n_heads: int


def match_index(ell: torch.Tensor, n_heads: int) -> SymIndex:
    _, length, _ = ell.shape
    bucket = bucket_lut(ell.device)[ell.long()]
    b, t, j = bucket.nonzero(as_tuple=True)
    beta = bucket[b, t, j].long()
    heads = torch.arange(n_heads, device=ell.device)
    flat = (((b[:, None] * n_heads + heads) * length + t[:, None]) * length + j[:, None]).reshape(-1)
    head_bucket = (heads * N_BUCKETS + beta[:, None]).reshape(-1)
    return SymIndex(bucket, flat, head_bucket, n_heads)


_CAUSAL: dict[tuple[int, str], torch.Tensor] = {}


def causal_mask(length: int, device: torch.device | str) -> torch.Tensor:
    key = (length, str(device))
    mask = _CAUSAL.get(key)
    if mask is None:
        mask = torch.ones(length, length, dtype=torch.bool, device=device).tril()
        _CAUSAL[key] = mask
    return mask


def _want_compile(args: tuple) -> bool:
    mode = os.environ.get("PALIMPSA_COMPILE", "auto")
    if mode == "0":
        return False
    on_cuda = any(isinstance(a, torch.Tensor) and a.is_cuda for a in args)
    if mode == "1":
        return on_cuda
    return on_cuda and os.name != "nt"  # no Triton on native Windows


class _Kernel:
    """Runs ``fn`` through torch.compile on CUDA when available, falling back to eager once on failure."""

    def __init__(self, fn):
        self.fn = fn
        self.compiled = None
        self.failed = False

    def __call__(self, *args):
        if not self.failed and _want_compile(args):
            if self.compiled is None:
                self.compiled = torch.compile(self.fn, dynamic=False)
            try:
                return self.compiled(*args)
            except Exception as exc:  # noqa: BLE001 - any compiler failure means: use eager
                warnings.warn(f"torch.compile failed for {self.fn.__name__}, using eager: {exc}")
                self.failed = True
        return self.fn(*args)


def _bias_forward(f_cum, table, bucket, causal, out_dtype):
    score = None
    if f_cum is not None:
        score = f_cum[..., :, None] - f_cum[..., None, :]
    if table is not None:
        heads = torch.arange(table.shape[0], device=table.device).view(1, -1, 1, 1)
        gathered = table[heads, bucket[:, None].long()]
        score = gathered if score is None else score + gathered
    return torch.where(causal, score, float("-inf")).to(out_dtype)


def _bias_backward_f(grad, causal, dtype):
    masked = torch.where(causal, grad.to(dtype), 0.0)
    return masked.sum(-1) - masked.sum(-2)


_forward_kernel = _Kernel(_bias_forward)
_backward_kernel = _Kernel(_bias_backward_f)


class SymBias(torch.autograd.Function):
    """Fused attention bias with a cheap custom backward.

    forward: one elementwise kernel writes the dense (B, H, L, L) bias.
    backward: dF from masked row/column sums of the bias gradient; dtable from a sparse gather over the
    matched entries only (bucket 0 is a fixed zero in the model, so it gets no gradient here).
    """

    @staticmethod
    def forward(ctx, f_cum, table, bucket, flat, head_bucket, causal, out_dtype):
        ctx.save_for_backward(flat, head_bucket, causal)
        ctx.f_dtype = None if f_cum is None else f_cum.dtype
        ctx.t_shape = None if table is None else tuple(table.shape)
        ctx.t_dtype = None if table is None else table.dtype
        return _forward_kernel(f_cum, table, bucket, causal, out_dtype)

    @staticmethod
    def backward(ctx, grad):
        flat, head_bucket, causal = ctx.saved_tensors
        d_f = d_table = None
        if ctx.f_dtype is not None and ctx.needs_input_grad[0]:
            d_f = _backward_kernel(grad, causal, ctx.f_dtype)
        if ctx.t_shape is not None and ctx.needs_input_grad[1]:
            n_heads, n_buckets = ctx.t_shape
            picked = grad.reshape(-1)[flat].to(ctx.t_dtype)
            d_table = torch.zeros(n_heads * n_buckets, dtype=ctx.t_dtype, device=grad.device)
            d_table = d_table.index_add_(0, head_bucket, picked).view(n_heads, n_buckets)
        return d_f, d_table, None, None, None, None, None


def sym_bias(log_f: torch.Tensor | None, table: torch.Tensor | None, sym: SymIndex | None,
             length: int, out_dtype: torch.dtype) -> torch.Tensor:
    """Build the (B, H, L, L) bias. log_f: (B, L, H) log forget gates or None; table: (H, N_BUCKETS) or None."""
    if log_f is None and table is None:
        raise ValueError("sym_bias needs forget gates, a match table, or both")
    if table is not None and sym is None:
        raise ValueError("a match table needs the SymIndex of the batch")
    ref = log_f if log_f is not None else table
    causal = causal_mask(length, ref.device)
    f_cum = None
    if log_f is not None:
        f_cum = log_f.transpose(1, 2).to(torch.promote_types(log_f.dtype, torch.float32)).cumsum(-1)
    bucket = flat = head_bucket = None
    if table is not None:
        bucket, flat, head_bucket = sym.bucket, sym.flat, sym.head_bucket
    return SymBias.apply(f_cum, table, bucket, flat, head_bucket, causal, out_dtype)


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor | None = None,
           impl: str = "sdpa") -> torch.Tensor:
    """Causal attention. q, k, v: (B, H, L, h) with k/v already expanded to H heads; bias: (B, H, L, L)."""
    if impl == "manual":
        dtype = torch.promote_types(q.dtype, torch.float32)
        scores = (q.to(dtype) @ k.to(dtype).transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if bias is None:
            scores = scores.masked_fill(~causal_mask(q.shape[-2], q.device), float("-inf"))
        else:
            scores = scores + bias.to(dtype)
        return (scores.softmax(-1) @ v.to(dtype)).to(q.dtype)
    if impl != "sdpa":
        raise ValueError(f"unknown attention impl {impl!r}")
    if bias is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True)
    if q.is_cuda:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
