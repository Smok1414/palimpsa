"""T7: Gated Residual (candidate A4): parameter formula, equal-parameter matching, causality, fp64 gradcheck."""
import pytest
import torch
from torch.func import functional_call

from palimpsa.model import PalimpsaLM, count_params, gr_extra_params, ladder_config

from helpers import activate_channels, tiny_config


def test_gr_extra_params_formula():
    base = PalimpsaLM(tiny_config("symA"))
    gr = PalimpsaLM(tiny_config("symA", residual="gr"))
    cfg = gr.cfg
    assert count_params(gr)[0] - count_params(base)[0] == gr_extra_params(cfg.d_model, cfg.n_layers)


@pytest.mark.parametrize("stage", ["10M", "30M", "100M"])
@pytest.mark.parametrize("variant", ["base", "symA"])
def test_gr_matched_to_prenorm_params(stage, variant):
    with torch.device("meta"):
        pre = PalimpsaLM(ladder_config(stage, variant))
        gr = PalimpsaLM(ladder_config(stage, variant, residual="gr"))
    p_pre, p_gr = count_params(pre)[0], count_params(gr)[0]
    assert abs(p_gr - p_pre) / p_pre <= 0.005, (p_pre, p_gr, gr.cfg.d_ff)
    assert gr.cfg.d_ff % 32 == 0 and gr.cfg.d_ff < pre.cfg.d_ff


@pytest.mark.parametrize("variant", ["base", "symA"])
def test_gr_causality(variant):
    torch.manual_seed(0)
    model = activate_channels(PalimpsaLM(tiny_config(variant, residual="gr")).double().eval())
    gen = torch.Generator().manual_seed(1)
    x = torch.randint(0, 4, (2, 48), generator=gen)
    x2 = x.clone()
    x2[:, 30] = (x[:, 30] + 1) % 4
    with torch.no_grad():
        y1, y2 = model(x), model(x2)
    assert (y1[:, :30] - y2[:, :30]).abs().max().item() <= 1e-10
    assert (y1[:, 30:] - y2[:, 30:]).abs().max().item() > 1e-6


def test_gr_gradcheck():
    torch.manual_seed(0)
    cfg = tiny_config("symA", vocab=5, d_model=8, n_layers=1, n_heads=2, n_kv_heads=1, head_dim=4, d_ff=16,
                      attn_impl="manual", residual="gr", gr_rank=2)
    model = PalimpsaLM(cfg).double()
    x = torch.randint(0, 3, (1, 8))
    names = [n for n, _ in model.named_parameters() if ".read_" in n or ".write_" in n or "read_final" in n]
    assert len(names) == 3 * 2 + 2 + 3  # two reads (gamma, down, up) + two writes (proj) + final read
    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    inputs = tuple(params[n].requires_grad_() for n in names)

    def fn(*tensors):
        return functional_call(model, {**params, **dict(zip(names, tensors))}, (x,))

    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-5)
