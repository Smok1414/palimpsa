"""T2: changing token p must not change logits at positions t < p, for every variant."""
import pytest
import torch

from palimpsa.model import PalimpsaLM

from helpers import activate_channels, random_docs, tiny_config

MAIN_VARIANTS = ["base", "symA", "fox", "symA_rope", "ell_rope", "ell_nope"]


@pytest.mark.parametrize("variant", MAIN_VARIANTS)
@pytest.mark.parametrize("use_doc", [False, True])
def test_causality(variant, use_doc):
    torch.manual_seed(0)
    model = activate_channels(PalimpsaLM(tiny_config(variant)).double().eval())
    gen = torch.Generator().manual_seed(1)
    x = torch.randint(0, 4, (2, 64), generator=gen)
    doc = random_docs(2, 64, 3, gen) if use_doc else None
    p = 40
    x2 = x.clone()
    x2[:, p] = (x[:, p] + 1) % 4
    with torch.no_grad():
        y1, y2 = model(x, doc), model(x2, doc)
    assert (y1[:, :p] - y2[:, :p]).abs().max().item() <= 1e-10
    assert (y1[:, p:] - y2[:, p:]).abs().max().item() > 1e-6  # the change is visible downstream
