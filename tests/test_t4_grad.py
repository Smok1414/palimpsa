"""T4: fp64 gradcheck of SymBias and of the forget/match parameters inside the full model."""
import pytest
import torch
import torch.nn.functional as F
from torch.func import functional_call

from palimpsa.model import PalimpsaLM
from palimpsa.symbolic import attend, match_index, match_len, sym_bias

from helpers import tiny_config


def test_symbias_gradcheck_through_attention():
    torch.manual_seed(0)
    bsz, n_heads, length, h = 1, 2, 12, 4
    x = torch.randint(0, 3, (bsz, length))
    sym = match_index(match_len(x), n_heads)
    assert sym.flat.numel() > 0
    q, k, v = (torch.randn(bsz, n_heads, length, h, dtype=torch.float64) for _ in range(3))
    log_f = F.logsigmoid(torch.randn(bsz, length, n_heads, dtype=torch.float64)).requires_grad_()
    b_free = torch.randn(n_heads, 7, dtype=torch.float64, requires_grad=True)

    def fn(lf, bf):
        bias = sym_bias(lf, F.pad(bf, (1, 0)), sym, length, torch.float64)
        return attend(q, k, v, bias, impl="manual")

    assert torch.autograd.gradcheck(fn, (log_f, b_free), eps=1e-6, atol=1e-6)


@pytest.mark.parametrize("variant", ["symA", "fox", "ell_rope"])
@pytest.mark.parametrize("impl", ["manual", "sdpa"])
def test_model_gradcheck_forget_and_table(variant, impl):
    torch.manual_seed(0)
    cfg = tiny_config(variant, vocab=5, d_model=8, n_layers=1, n_heads=2, n_kv_heads=1, head_dim=4,
                      d_ff=16, attn_impl=impl)
    model = PalimpsaLM(cfg).double()
    x = torch.randint(0, 3, (1, 10))
    names = [n for n, _ in model.named_parameters() if n.endswith(("wf.weight", "wf.bias", "b_free"))]
    assert names
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    for n in names:  # move off the zero init so the check is not degenerate
        params[n] = params[n] + 0.5 * torch.randn_like(params[n])
    inputs = tuple(params[n].requires_grad_() for n in names)

    def fn(*tensors):
        return functional_call(model, {**params, **dict(zip(names, tensors))}, (x,))

    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-5)
