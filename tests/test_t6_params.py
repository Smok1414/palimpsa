"""T6: parameter counts of every ladder stage and variant equal table 5.2 to the unit."""
import pytest
import torch

from palimpsa.model import PalimpsaLM, count_params, ladder_config

EXPECTED = {  # stage: {variant: total params}
    "1M": {"base": 7_325_376, "symA": 7_327_176, "fox": 7_327_113, "symA_rope": 7_327_176,
           "ell_rope": 7_325_439},
    "10M": {"base": 21_730_944, "symA": 21_745_056, "fox": 21_744_804, "symA_rope": 21_745_056,
            "ell_rope": 21_731_196},
    "30M": {"base": 50_294_592, "symA": 50_341_896, "fox": 50_341_329, "symA_rope": 50_341_896,
            "ell_rope": 50_295_159},
    "100M": {"base": 100_094_208, "symA": 100_205_952, "fox": 100_204_944, "symA_rope": 100_205_952,
             "ell_rope": 100_095_216},
}
CASES = [(stage, variant) for stage, table in EXPECTED.items() for variant in table]


@pytest.mark.parametrize("stage,variant", CASES)
def test_param_count(stage, variant):
    with torch.device("meta"):
        model = PalimpsaLM(ladder_config(stage, variant))
    total, non_emb = count_params(model)
    assert total == EXPECTED[stage][variant]
    assert non_emb == total - 32000 * model.cfg.d_model


def test_100m_within_two_percent():
    for variant, total in EXPECTED["100M"].items():
        assert abs(total / 1e8 - 1) <= 0.02, variant
