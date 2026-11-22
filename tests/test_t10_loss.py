"""T10: chunked lm_loss equals the full fp32 cross-entropy + z-loss (loss and gradients) and needs less memory."""
import pytest
import torch
import torch.nn.functional as F

from palimpsa.model import lm_loss

from helpers import DEVICES


def full_lm_loss(logits, targets, z_loss=1e-4, ignore_index=-100):
    """The previous (full-size fp32) implementation, kept as the reference."""
    flat = logits.reshape(-1, logits.shape[-1]).float()
    tgt = targets.reshape(-1)
    loss = F.cross_entropy(flat, tgt, ignore_index=ignore_index)
    if z_loss:
        keep = tgt != ignore_index
        loss = loss + z_loss * torch.logsumexp(flat[keep], dim=-1).pow(2).mean()
    return loss


def case(device, dtype, shape=(3, 37), vocab=101, ignore=True, seed=0, std=3.0):
    gen = torch.Generator().manual_seed(seed)
    logits = (std * torch.randn(*shape, vocab, generator=gen)).to(device=device, dtype=dtype)
    targets = torch.randint(0, vocab, shape, generator=gen)
    if ignore:
        targets[torch.rand(shape, generator=gen) < 0.2] = -100
    return logits, targets.to(device)


def loss_and_grad(fn, logits, targets, scale=1.0, **kw):
    x = logits.detach().clone().requires_grad_()
    loss = fn(x, targets, **kw)
    (loss * scale).backward()
    return loss.detach(), x.grad


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("chunk", [1, 7, 111, 4096])
@pytest.mark.parametrize("z_loss", [0.0, 1e-4, 0.1])
@pytest.mark.parametrize("ignore", [False, True])
def test_chunked_matches_full_fp32(device, chunk, z_loss, ignore):
    logits, targets = case(device, torch.float32, ignore=ignore)
    l_ref, g_ref = loss_and_grad(full_lm_loss, logits, targets, z_loss=z_loss)
    l_new, g_new = loss_and_grad(lm_loss, logits, targets, z_loss=z_loss, chunk=chunk)
    assert abs(l_new.item() - l_ref.item()) <= 1e-6 * abs(l_ref.item())
    assert (g_new - g_ref).abs().max().item() <= 1e-5 * g_ref.abs().max().item()
    if ignore:  # ignored rows get no gradient
        assert g_new[targets == -100].abs().max().item() == 0.0


@pytest.mark.parametrize("device", DEVICES)
def test_chunked_matches_full_fp16_at_lm_scale(device):
    """fp16 logits as in training (V = 32000, loss scaled like GradScaler): same loss, same fp16 gradient."""
    logits, targets = case(device, torch.float16, shape=(2, 300), vocab=32000, std=2.0)
    l_ref, g_ref = loss_and_grad(full_lm_loss, logits, targets, scale=65536.0)
    l_new, g_new = loss_and_grad(lm_loss, logits, targets, scale=65536.0, chunk=128)
    assert g_new.dtype == torch.float16
    assert abs(l_new.item() - l_ref.item()) <= 1e-6 * abs(l_ref.item())
    diff = (g_new.float() - g_ref.float()).abs()
    assert diff.max().item() <= 2e-3 * g_ref.float().abs().max().item()  # at most fp16 rounding
    assert (diff > 0).float().mean().item() <= 0.01


def test_chunked_gradcheck_fp64():
    logits, targets = case("cpu", torch.float64, shape=(2, 5), vocab=11)
    x = logits.detach().requires_grad_()
    assert torch.autograd.gradcheck(lambda t: lm_loss(t, targets, z_loss=0.1, chunk=3), (x,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_chunked_uses_less_memory_cuda():
    logits, targets = case("cuda", torch.float16, shape=(4, 1024), vocab=32000, ignore=False)
    peaks = {}
    for name, fn in (("full", full_lm_loss), ("chunked", lm_loss)):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        loss_and_grad(fn, logits, targets)
        torch.cuda.synchronize()
        peaks[name] = torch.cuda.max_memory_allocated() - base
    # logits: 4096 x 32000 fp16 = 250 MiB; the full version holds several fp32 copies (>= 1 GiB)
    assert peaks["chunked"] <= 0.4 * peaks["full"], peaks
