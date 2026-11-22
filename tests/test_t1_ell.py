"""T1: vectorised match lengths and the sparse match index against a naive reference."""
import pytest
import torch

from palimpsa.symbolic import BUCKET_OF_LEN, N_BUCKETS, match_index, match_len

from helpers import DEVICES, random_docs


def naive_match_len(x: torch.Tensor, doc: torch.Tensor | None, n_max: int) -> torch.Tensor:
    bsz, length = x.shape
    xs = x.tolist()
    ds = doc.tolist() if doc is not None else [[0] * length for _ in range(bsz)]
    out = torch.zeros(bsz, length, length, dtype=torch.uint8)
    for b in range(bsz):
        for t in range(length):
            for j in range(1, t + 1):
                i, k = j - 1, 0
                while (k < n_max and i - k >= 0 and xs[b][t - k] == xs[b][i - k]
                       and ds[b][t - k] == ds[b][i - k]):
                    k += 1
                out[b, t, j] = k
    return out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("n_max", [1, 3, 16])
def test_match_len_random_small_vocab(device, n_max):
    gen = torch.Generator().manual_seed(n_max)
    x = torch.randint(0, 4, (4, 128), generator=gen)
    doc = random_docs(4, 128, 3, gen)
    expected = naive_match_len(x, doc, n_max)
    got = match_len(x.to(device), doc.to(device), n_max).cpu()
    assert torch.equal(got, expected)
    got_no_doc = match_len(x.to(device), None, n_max).cpu()
    assert torch.equal(got_no_doc, naive_match_len(x, None, n_max))


@pytest.mark.parametrize("device", DEVICES)
def test_match_len_cap_on_long_run(device):
    x = torch.cat([torch.arange(10), torch.full((40,), 7), torch.arange(10)]).view(1, -1)
    got = match_len(x.to(device)).cpu()
    assert int(got.max()) == 16
    assert torch.equal(got, naive_match_len(x, None, 16))


@pytest.mark.parametrize("device", DEVICES)
def test_match_index_consistent(device):
    gen = torch.Generator().manual_seed(5)
    x = torch.randint(0, 3, (3, 64), generator=gen).to(device)
    ell = match_len(x)
    n_heads = 4
    sym = match_index(ell, n_heads)
    lut = torch.tensor(BUCKET_OF_LEN, dtype=torch.uint8)
    assert torch.equal(sym.bucket.cpu(), lut[ell.cpu().long()])
    bsz, length, _ = ell.shape
    flat, hb = sym.flat.cpu(), sym.head_bucket.cpu()
    j = flat % length
    t = (flat // length) % length
    h = (flat // (length * length)) % n_heads
    b = flat // (length * length * n_heads)
    ell_c = ell.cpu()
    assert flat.numel() == int((ell_c > 0).sum()) * n_heads
    assert bool((ell_c[b, t, j] > 0).all())
    assert torch.equal(hb, h * N_BUCKETS + lut[ell_c[b, t, j].long()].long())
    assert bool((j <= t).all())
