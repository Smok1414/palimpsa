"""T5: leakage canary with a positive control (see palimpsa/canary.py)."""
import pytest

from palimpsa.canary import LN_ALPHABET, leak_canary


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["base", "symA", "ell_rope"])
def test_no_leak_on_uniform_tokens(variant):
    loss = leak_canary(variant)
    assert loss >= LN_ALPHABET - 0.01, f"{variant}: {loss:.4f} < ln(64) - 0.01 = {LN_ALPHABET - 0.01:.4f}"


@pytest.mark.slow
def test_positive_control_detects_a_leak():
    loss = leak_canary("symA", leaky=True)
    assert loss < LN_ALPHABET - 0.5, f"leaky channel not exploited: {loss:.4f} vs ln(64) = {LN_ALPHABET:.4f}"
