"""T3: equivalences between variants and between the SDPA bias path and a manual fp32 reference."""
import pytest
import torch

from palimpsa.model import PalimpsaLM

from helpers import activate_channels, tiny_config


def _clone_into(src: PalimpsaLM, variant: str, **overrides) -> PalimpsaLM:
    dst = PalimpsaLM(tiny_config(variant, **overrides)).to(dtype=next(src.parameters()).dtype)
    missing, _ = dst.load_state_dict(src.state_dict(), strict=False)
    assert not missing, missing
    return dst.eval()


def _inputs(seed=0, length=48):
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(0, 5, (2, length), generator=gen)


@torch.no_grad()
def test_symA_with_zero_table_equals_fox():
    torch.manual_seed(0)
    sym = activate_channels(PalimpsaLM(tiny_config("symA")).double().eval())
    for block in sym.blocks:
        block.attn.b_free.zero_()
    fox = _clone_into(sym, "fox")
    x = _inputs()
    assert (sym(x) - fox(x)).abs().max().item() <= 1e-12


@torch.no_grad()
def test_fox_without_forgetting_equals_nope():
    torch.manual_seed(0)
    fox = PalimpsaLM(tiny_config("fox")).double().eval()
    for block in fox.blocks:
        block.attn.wf.weight.zero_()
        block.attn.wf.bias.fill_(40.0)  # log sigmoid(40) ~ -4e-18: no forgetting
    nope = _clone_into(fox, "nope")
    x = _inputs()
    assert (fox(x) - nope(x)).abs().max().item() <= 1e-10


@torch.no_grad()
def test_ell_rope_with_zero_table_equals_base():
    torch.manual_seed(0)
    ell = PalimpsaLM(tiny_config("ell_rope")).double().eval()
    base = _clone_into(ell, "base")
    x = _inputs()
    assert (ell(x) - base(x)).abs().max().item() <= 1e-10


@pytest.mark.parametrize("variant", ["symA", "fox", "symA_rope", "ell_rope"])
@torch.no_grad()
def test_sdpa_bias_path_matches_manual_cpu(variant):
    torch.manual_seed(0)
    sdpa = activate_channels(PalimpsaLM(tiny_config(variant)).eval())
    manual = _clone_into(sdpa, variant, attn_impl="manual")
    x = _inputs()
    assert (sdpa(x) - manual(x)).abs().max().item() <= 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("variant", ["symA", "fox", "ell_rope"])
@torch.no_grad()
def test_sdpa_fp16_matches_manual_fp32_cuda(variant):
    torch.manual_seed(0)
    cfg = dict(vocab=64, d_model=128, n_heads=2, n_kv_heads=1, head_dim=64, d_ff=256)
    sdpa = activate_channels(PalimpsaLM(tiny_config(variant, **cfg))).cuda().eval()
    manual = _clone_into(sdpa, variant, attn_impl="manual", **cfg).cuda()
    x = _inputs(length=256).cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        y16 = sdpa(x).float()
    y32 = manual(x).float()
    rel = (y16 - y32).abs().max() / y32.abs().max()
    assert rel.item() <= 1e-2
