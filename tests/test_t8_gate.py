"""T8: attention output gate (candidate A5): parameter matching, causality, separation from A4."""
import pytest
import torch

from palimpsa.model import Config, PalimpsaLM, count_params, gate_extra_params, ladder_config

from helpers import activate_channels, tiny_config


@pytest.mark.parametrize("gate", ["elementwise", "headwise"])
def test_gate_extra_params_formula(gate):
    plain = PalimpsaLM(tiny_config("symA"))
    gated = PalimpsaLM(tiny_config("symA", attn_gate=gate))
    c = gated.cfg
    assert count_params(gated)[0] - count_params(plain)[0] == gate_extra_params(
        c.d_model, c.n_layers, c.n_heads, c.head_dim, gate)


@pytest.mark.parametrize("stage", ["10M", "30M", "100M"])
@pytest.mark.parametrize("variant", ["base", "symA"])
@pytest.mark.parametrize("gate", ["elementwise", "headwise"])
def test_gate_matched_to_plain_params(stage, variant, gate):
    with torch.device("meta"):
        plain = PalimpsaLM(ladder_config(stage, variant))
        gated = PalimpsaLM(ladder_config(stage, variant, attn_gate=gate))
    p0, p1 = count_params(plain)[0], count_params(gated)[0]
    assert abs(p1 - p0) / p0 <= 0.005, (p0, p1, gated.cfg.d_ff)


@pytest.mark.parametrize("variant", ["base", "symA"])
@pytest.mark.parametrize("gate", ["elementwise", "headwise"])
def test_gate_causality(variant, gate):
    torch.manual_seed(0)
    model = activate_channels(PalimpsaLM(tiny_config(variant, attn_gate=gate)).double().eval())
    gen = torch.Generator().manual_seed(1)
    x = torch.randint(0, 4, (2, 48), generator=gen)
    x2 = x.clone()
    x2[:, 30] = (x[:, 30] + 1) % 4
    with torch.no_grad():
        y1, y2 = model(x), model(x2)
    assert (y1[:, :30] - y2[:, :30]).abs().max().item() <= 1e-10
    assert (y1[:, 30:] - y2[:, 30:]).abs().max().item() > 1e-6


def test_a4_and_a5_are_not_combined():
    with pytest.raises(ValueError):
        Config(residual="gr", attn_gate="elementwise")
