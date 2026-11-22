"""Palimpsa language model: one code path for the baseline and every SymA variant / ablation.

Variant  | RoPE | forget gates | match table | attention path
base     |  yes |      -       |      -      | SDPA is_causal (fast)
nope     |   -  |      -       |      -      | SDPA is_causal (tests only)
symA     |   -  |     yes      |     yes     | SDPA + SymBias
fox      |   -  |     yes      |      -      | SDPA + SymBias
symA_rope|  yes |     yes      |     yes     | SDPA + SymBias
ell_rope |  yes |      -       |     yes     | SDPA + SymBias
ell_nope |   -  |      -       |     yes     | SDPA + SymBias (optional ablation)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .symbolic import N_BUCKETS, N_MATCH, SymIndex, attend, match_index, match_len, sym_bias

VARIANTS = {  # name: (rope, forget, ell)
    "base": (True, False, False),
    "nope": (False, False, False),
    "symA": (False, True, True),
    "fox": (False, True, False),
    "symA_rope": (True, True, True),
    "ell_rope": (True, False, True),
    "ell_nope": (False, False, True),
}

LADDER = {"1M": (192, 3), "10M": (384, 6), "30M": (576, 9), "100M": (768, 12)}  # (d_model, n_layers)


@dataclass
class Config:
    vocab: int = 32000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 64
    d_ff: int = 2048
    variant: str = "symA"
    n_match: int = N_MATCH
    rope_base: float = 10000.0
    norm_eps: float = 1e-6
    forget_tau_min: float = 4.0
    forget_tau_max: float = 4096.0
    init_std: float = 0.02
    attn_impl: str = "sdpa"  # "sdpa" or "manual" (fp32 reference for tests)
    qk_norm: bool = True  # False only for diagnostics; the compared models always use QK-norm
    residual: str = "prenorm"  # "prenorm" or "gr" (Gated Residual, candidate A4)
    gr_branches: int = 4
    gr_rank: int = 0  # 0 -> d_model // 8
    attn_gate: str = "none"  # "none", "elementwise" or "headwise" sigmoid gate on the SDPA output (candidate A5)

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}; expected one of {sorted(VARIANTS)}")
        if self.residual not in ("prenorm", "gr"):
            raise ValueError(f"unknown residual {self.residual!r}")
        if self.attn_gate not in ("none", "elementwise", "headwise"):
            raise ValueError(f"unknown attn_gate {self.attn_gate!r}")
        if self.attn_gate != "none" and self.residual == "gr":
            raise ValueError("candidates A4 (GR) and A5 (output gate) are tested separately, not in one model")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        if self.use_rope and self.head_dim % 2:
            raise ValueError("RoPE needs an even head_dim")

    @property
    def use_rope(self) -> bool:
        return VARIANTS[self.variant][0]

    @property
    def use_forget(self) -> bool:
        return VARIANTS[self.variant][1]

    @property
    def use_ell(self) -> bool:
        return VARIANTS[self.variant][2]


def gr_extra_params(d_model: int, n_layers: int, n_branches: int = 4, rank: int = 0) -> int:
    """Parameters GR adds over pre-norm: two read/write pairs per layer and a final read, minus the RMSNorms."""
    rank = rank or d_model // 8
    read = n_branches * d_model + 2 * rank * n_branches * d_model  # gamma, down, up
    write = n_branches * n_branches * d_model
    return n_layers * (2 * (read + write) - 2 * d_model) + (read - d_model)


def gr_matched_d_ff(d_model: int, n_layers: int, d_ff: int, n_branches: int = 4, rank: int = 0) -> int:
    """d_ff (multiple of 32) that keeps a GR model at the parameter count of its pre-norm counterpart."""
    cut = gr_extra_params(d_model, n_layers, n_branches, rank) / (3 * d_model * n_layers)
    return d_ff - 32 * round(cut / 32)


def gate_extra_params(d_model: int, n_layers: int, n_heads: int, head_dim: int, attn_gate: str) -> int:
    """Parameters of the output gate W_g: d x (H*h) per layer (elementwise) or d x H (headwise), no bias."""
    width = {"none": 0, "elementwise": n_heads * head_dim, "headwise": n_heads}[attn_gate]
    return n_layers * d_model * width


def ladder_config(stage: str, variant: str, residual: str = "prenorm", attn_gate: str = "none",
                  **overrides) -> Config:
    """Ladder stage from section 5.2: h = 64, H = d / 64, H_kv = H / 3, d_ff = 8d / 3, d / n_layers = 64.

    residual="gr" (candidate A4) and attn_gate != "none" (candidate A5) shrink d_ff (to a multiple of 32) so the
    model keeps the parameter count of the plain pre-norm one.
    """
    d_model, n_layers = LADDER[stage]
    n_heads = d_model // 64
    d_ff = 8 * d_model // 3
    if residual == "gr":
        d_ff = gr_matched_d_ff(d_model, n_layers, d_ff)
    extra = gate_extra_params(d_model, n_layers, n_heads, 64, attn_gate)
    d_ff -= 32 * round(extra / (3 * d_model * n_layers) / 32)
    return Config(d_model=d_model, n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_heads // 3,
                  head_dim=64, d_ff=d_ff, variant=variant, residual=residual, attn_gate=attn_gate, **overrides)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = torch.promote_types(x.dtype, torch.float32)
        xf = x.to(dtype)
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.to(dtype)
        return y.to(x.dtype)


def rope_cache(length: int, dim: int, base: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float64) / dim)
    angles = torch.outer(torch.arange(length, device=device, dtype=torch.float64), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, L, H, h); rotate the two halves of the head dimension."""
    dtype = torch.promote_types(x.dtype, torch.float32)
    x1, x2 = x.to(dtype).chunk(2, dim=-1)
    c = cos.to(dtype)[None, :, None, :]
    s = sin.to(dtype)[None, :, None, :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x.dtype)


def forget_bias_init(n_heads: int, tau_min: float, tau_max: float) -> torch.Tensor:
    """c_h = logit(2^(-1/tau_h)), half-lives tau_h log-spaced over [tau_min, tau_max] (section 4.5)."""
    tau = torch.logspace(math.log10(tau_min), math.log10(tau_max), n_heads, dtype=torch.float64)
    f = torch.pow(2.0, -1.0 / tau)
    return torch.log(f / (1 - f))


@dataclass
class BatchContext:
    rope: tuple[torch.Tensor, torch.Tensor] | None
    sym: SymIndex | None


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        d, n_h, n_kv, h = cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.wq = nn.Linear(d, n_h * h, bias=False)
        self.wk = nn.Linear(d, n_kv * h, bias=False)
        self.wv = nn.Linear(d, n_kv * h, bias=False)
        self.wo = nn.Linear(n_h * h, d, bias=False)
        self.q_norm = RMSNorm(h, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(h, cfg.norm_eps) if cfg.qk_norm else nn.Identity()
        self.wf = nn.Linear(d, n_h, bias=True) if cfg.use_forget else None
        self.b_free = nn.Parameter(torch.zeros(n_h, N_BUCKETS - 1)) if cfg.use_ell else None
        gate_width = {"none": 0, "elementwise": n_h * h, "headwise": n_h}[cfg.attn_gate]
        self.wg = nn.Linear(d, gate_width, bias=False) if gate_width else None

    def forward(self, a: torch.Tensor, ctx: BatchContext) -> torch.Tensor:
        cfg = self.cfg
        bsz, length, _ = a.shape
        n_h, n_kv, h = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        q = self.q_norm(self.wq(a).view(bsz, length, n_h, h))
        k = self.k_norm(self.wk(a).view(bsz, length, n_kv, h))
        v = self.wv(a).view(bsz, length, n_kv, h)
        if cfg.use_rope:
            q, k = apply_rope(q, *ctx.rope), apply_rope(k, *ctx.rope)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        q = q.contiguous()
        if n_kv != n_h:
            k = k.repeat_interleave(n_h // n_kv, dim=1)
            v = v.repeat_interleave(n_h // n_kv, dim=1)
        bias = None
        if self.wf is not None or self.b_free is not None:
            log_f = None
            if self.wf is not None:
                gate = self.wf(a)
                log_f = F.logsigmoid(gate.to(torch.promote_types(gate.dtype, torch.float32)))
            table = None if self.b_free is None else F.pad(self.b_free, (1, 0))
            bias = sym_bias(log_f, table, ctx.sym, length, q.dtype)
        out = attend(q, k, v, bias, cfg.attn_impl).transpose(1, 2)  # (B, L, H, h)
        if self.wg is not None:  # sigmoid gate on the SDPA output, from the attention input (A5)
            gate = torch.sigmoid(self.wg(a))
            gate = gate.view(bsz, length, n_h, h if cfg.attn_gate == "elementwise" else 1)
            out = out * gate.to(out.dtype)
        return self.wo(out.reshape(bsz, length, n_h * h))


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GRRead(nn.Module):
    """Gated Residual read (arXiv 2608.30320, Eqs. 30-32).

    R_hat_i = RMSNorm(R_i; gamma_i);  G = sigmoid(W_u SiLU(W_d vec(R_hat) / n_r));  x = mean_i G_i * R_hat_i
    """

    def __init__(self, d_model: int, n_branches: int, rank: int, eps: float):
        super().__init__()
        self.n_branches, self.eps = n_branches, eps
        self.gamma = nn.Parameter(torch.ones(n_branches, d_model))
        self.down = nn.Linear(n_branches * d_model, rank, bias=False)
        self.up = nn.Linear(rank, n_branches * d_model, bias=False)

    def forward(self, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = torch.promote_types(r.dtype, torch.float32)
        rf = r.to(dtype)
        r_hat = rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + self.eps) * self.gamma.to(dtype)
        gate = torch.sigmoid(self.up(F.silu(self.down(r_hat.flatten(-2)) / self.n_branches)))
        x = (gate.view(r_hat.shape).to(dtype) * r_hat).mean(-2)
        return x.to(r.dtype), r_hat


class GRWrite(nn.Module):
    """Gated Residual write (Eqs. 33-34): R_i += s_i * y with s = 2 * sigmoid(W_w vec(R_hat) / n_r)."""

    def __init__(self, d_model: int, n_branches: int):
        super().__init__()
        self.n_branches = n_branches
        self.proj = nn.Linear(n_branches * d_model, n_branches, bias=False)

    def forward(self, r: torch.Tensor, r_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        s = 2 * torch.sigmoid(self.proj(r_hat.flatten(-2)).to(r.dtype) / self.n_branches)
        return r + s[..., None] * y[..., None, :].to(r.dtype)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gr = cfg.residual == "gr"
        if self.gr:
            rank = cfg.gr_rank or cfg.d_model // 8
            self.read_attn = GRRead(cfg.d_model, cfg.gr_branches, rank, cfg.norm_eps)
            self.write_attn = GRWrite(cfg.d_model, cfg.gr_branches)
            self.read_mlp = GRRead(cfg.d_model, cfg.gr_branches, rank, cfg.norm_eps)
            self.write_mlp = GRWrite(cfg.d_model, cfg.gr_branches)
        else:
            self.norm1 = RMSNorm(cfg.d_model, cfg.norm_eps)
            self.norm2 = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor, ctx: BatchContext) -> torch.Tensor:
        if self.gr:  # x is the widened stream (B, L, n_r, d); GR replaces pre-norm
            a, r_hat = self.read_attn(x)
            x = self.write_attn(x, r_hat, self.attn(a, ctx))
            m, r_hat = self.read_mlp(x)
            return self.write_mlp(x, r_hat, self.mlp(m))
        x = x + self.attn(self.norm1(x), ctx)
        return x + self.mlp(self.norm2(x))


class PalimpsaLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        if cfg.residual == "gr":  # final gated read replaces the final RMSNorm
            self.read_final = GRRead(cfg.d_model, cfg.gr_branches, cfg.gr_rank or cfg.d_model // 8, cfg.norm_eps)
        else:
            self.norm_f = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        cfg = self.cfg
        out_std = cfg.init_std / math.sqrt(2 * cfg.n_layers)
        c_init = forget_bias_init(cfg.n_heads, cfg.forget_tau_min, cfg.forget_tau_max)
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "w_down.weight")):
                nn.init.normal_(p, 0.0, out_std)
            elif name.endswith("wf.weight") or name.endswith("b_free"):
                nn.init.zeros_(p)
            elif name.endswith("wf.bias"):
                p.copy_(c_init.to(device=p.device, dtype=p.dtype))
            elif name.endswith("gamma"):
                nn.init.ones_(p)
            elif p.dim() == 2:
                nn.init.normal_(p, 0.0, cfg.init_std)
            else:
                nn.init.ones_(p)

    def forward(self, idx: torch.Tensor, doc: torch.Tensor | None = None,
                ell: torch.Tensor | None = None, last_only: bool = False) -> torch.Tensor:
        """idx: (B, L) token ids; doc: optional (B, L) document ids; ell: optional precomputed match lengths.
        last_only: logits of the last position only, (B, 1, V) (sampling: saves the B x L x V output)."""
        cfg = self.cfg
        _, length = idx.shape
        rope = rope_cache(length, cfg.head_dim, cfg.rope_base, idx.device) if cfg.use_rope else None
        sym = None
        if cfg.use_ell:
            if ell is None:
                ell = match_len(idx, doc, cfg.n_match)
            sym = match_index(ell, cfg.n_heads)
        ctx = BatchContext(rope, sym)
        x = self.embed(idx)
        if cfg.residual == "gr":  # every branch starts at the token embedding
            x = x[:, :, None, :].expand(-1, -1, cfg.gr_branches, -1)
        for block in self.blocks:
            x = block(x, ctx)
        out = self.read_final(x)[0] if cfg.residual == "gr" else self.norm_f(x)
        return F.linear(out[:, -1:] if last_only else out, self.embed.weight)


class _ChunkedLMLoss(torch.autograd.Function):
    """mean(lse - x_target) + z_loss * mean(lse^2) over the rows whose target != ignore_index.

    The (N, V) logits are processed `chunk` rows at a time in fp32 (fp64 stays fp64). Only the per-row
    log-normaliser is saved; backward recomputes the softmax chunk by chunk and writes the gradient in the dtype
    of the logits, so no full-size fp32 copy of the logits exists in either pass.
    """

    @staticmethod
    def forward(ctx, logits, targets, z_loss, ignore_index, chunk):
        dtype = torch.promote_types(logits.dtype, torch.float32)
        keep = targets != ignore_index
        safe = torch.where(keep, targets, 0)
        lse = torch.empty(logits.shape[0], dtype=dtype, device=logits.device)
        picked = torch.empty_like(lse)
        for i in range(0, logits.shape[0], chunk):
            part = logits[i:i + chunk].to(dtype)
            lse[i:i + chunk] = torch.logsumexp(part, dim=-1)
            picked[i:i + chunk] = part.gather(1, safe[i:i + chunk, None]).squeeze(1)
        weight = keep.to(dtype)
        n = weight.sum()
        loss = ((lse - picked) * weight).sum() / n
        if z_loss:
            loss = loss + z_loss * (lse.pow(2) * weight).sum() / n
        ctx.save_for_backward(logits, safe, weight, lse, n)
        ctx.z_loss, ctx.chunk = z_loss, chunk
        return loss

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        logits, safe, weight, lse, n = ctx.saved_tensors
        grad = torch.empty_like(logits)
        row_scale = weight * (grad_out.to(lse.dtype) / n)
        coef = row_scale * (1 + 2 * ctx.z_loss * lse) if ctx.z_loss else row_scale
        for i in range(0, logits.shape[0], ctx.chunk):
            rows = slice(i, i + ctx.chunk)
            g = (logits[rows].to(lse.dtype) - lse[rows, None]).exp_().mul_(coef[rows, None])
            g.scatter_add_(1, safe[rows, None], -row_scale[rows, None])
            grad[rows] = g
        return grad, None, None, None, None


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, z_loss: float = 1e-4,
            ignore_index: int = -100, chunk: int = 1024) -> torch.Tensor:
    """Cross-entropy in fp32 plus z-loss on log-normalisers of the scored positions, `chunk` rows at a time."""
    return _ChunkedLMLoss.apply(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), z_loss, ignore_index,
                                chunk)


def count_params(model: nn.Module) -> tuple[int, int]:
    """(total, non-embedding) parameter counts; the tied head is counted once."""
    total = sum(p.numel() for p in model.parameters())
    return total, total - model.embed.weight.numel()
