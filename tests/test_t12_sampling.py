"""T12: sampling (batched generation, repetition penalty) and repetition metrics. CPU only."""
import numpy as np
import torch

from palimpsa.eval import doc_starts
from palimpsa.loader import EOS_ID
from palimpsa.model import PalimpsaLM, ladder_config
from palimpsa.sampling import _filter, apply_repetition_penalty, distinct_n, generate, repeated_tail


def test_repetition_metrics():
    assert distinct_n([1, 2, 3, 4, 1, 2, 3, 4], 4) == 0.8
    assert distinct_n([1, 2, 3], 4) is None
    assert repeated_tail(list(range(50))) == 0.0
    assert abs(repeated_tail([5, 6, 7] + [1, 2] * 10) - 20 / 23) < 1e-12
    assert repeated_tail([9] * 10) == 1.0
    assert repeated_tail([1, 2, 3, 1, 2]) == 0.0  # one period plus a partial one is not a loop


def test_repetition_penalty_and_filters():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    seen = torch.tensor([[True, True, False]])
    assert torch.equal(apply_repetition_penalty(logits, seen, 2.0), torch.tensor([[1.0, -4.0, 1.0]]))
    assert torch.equal(apply_repetition_penalty(logits, seen, 1.0), logits)
    kept = torch.isfinite(_filter(torch.tensor([[3.0, 2.9, 0.0, -1.0]]), top_k=0, top_p=0.5))
    assert kept.tolist() == [[True, False, False, False]]
    assert torch.isfinite(_filter(torch.tensor([[3.0, 2.0, 1.0]]), top_k=2, top_p=1.0)).tolist() == [[True, True, False]]


def test_batched_greedy_matches_single_and_penalty_changes_output():
    torch.manual_seed(0)
    model = PalimpsaLM(ladder_config("1M", "symA")).eval()
    p1, p2 = [5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15, 16]
    both = generate(model, [p1, p2], 12, temperature=0)
    assert both == [generate(model, [p1], 12, temperature=0)[0], generate(model, [p2], 12, temperature=0)[0]]
    assert all(EOS_ID not in c for c in both)
    a = generate(model, [p1], 12, temperature=0.7, generator=torch.Generator().manual_seed(3))
    b = generate(model, [p1], 12, temperature=0.7, generator=torch.Generator().manual_seed(3))
    assert a == b  # same seed, same sample
    plain = generate(model, [[7] * 6], 20, temperature=0)[0]
    penal = generate(model, [[7] * 6], 20, temperature=0, repetition_penalty=5.0)[0]
    assert len(set(penal)) >= len(set(plain))


def test_last_only_logits_match_full_forward():
    torch.manual_seed(0)
    for variant in ("base", "symA"):
        model = PalimpsaLM(ladder_config("1M", variant)).eval()
        x = torch.randint(1, 200, (2, 24))
        with torch.no_grad():
            full, last = model(x), model(x, last_only=True)
        assert last.shape == (2, 1, full.shape[-1]) and torch.allclose(full[:, -1:], last, atol=1e-5)


def test_doc_starts_picks_long_documents(tmp_path):
    docs = [[3] * 10, [4] * 200, [5] * 50, [6] * 300, [7] * 160]
    stream = np.array(sum((d + [EOS_ID] for d in docs), []), dtype=np.uint16)
    (tmp_path / "web").mkdir()
    stream.tofile(tmp_path / "web" / "val.bin")
    offsets = np.cumsum([0] + [len(d) + 1 for d in docs[:-1]]).astype(np.int64)
    np.save(tmp_path / "web" / "val_offsets.npy", offsets)
    got = doc_starts(tmp_path, "web", 32, 120, 10)
    assert [d[0] for d in got] == [4, 6, 7] and all(EOS_ID not in d for d in got)
