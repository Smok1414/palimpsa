"""T9: data loader determinism, resume equivalence of train.py (section 9 'T7 resume'), eval smoke. CPU only."""
import numpy as np
import pytest
import torch

from palimpsa.eval import val_loss
from palimpsa.loader import DOMAINS, EOS_ID, TrainStream, doc_ids
from palimpsa.model import PalimpsaLM, ladder_config
from palimpsa.train import main as train_main


def make_data(root, n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    for d in DOMAINS:
        (root / d).mkdir(parents=True, exist_ok=True)
        for name, size in (("train.bin", n), ("val.bin", n // 4)):
            arr = rng.integers(2, 500, size).astype(np.uint16)
            arr[rng.random(size) < 0.02] = EOS_ID
            arr.tofile(root / d / name)
    return root


def test_loader_is_deterministic_and_doc_ids_follow_eos(tmp_path):
    data = TrainStream(make_data(tmp_path), seq_len=64, seed=3)
    a = data.batch(step=5, micro=1, batch_size=4)
    b = data.batch(step=5, micro=1, batch_size=4)
    c = data.batch(step=5, micro=2, batch_size=4)
    assert torch.equal(a[0], b[0]) and torch.equal(a[2], b[2])
    assert not torch.equal(a[0], c[0])
    assert torch.equal(a[1][:, :-1], a[0][:, 1:])  # targets are inputs shifted by one
    x = np.array([[5, EOS_ID, 7, 8, EOS_ID, 9]])
    assert doc_ids(x).tolist() == [[0, 0, 1, 1, 1, 2]]


def test_resume_reproduces_an_uninterrupted_run(tmp_path):
    root = make_data(tmp_path / "data")
    common = ["--stage", "1M", "--variant", "symA", "--data", str(root), "--steps", "6", "--seq-len", "64",
              "--micro-batch", "2", "--global-batch-tokens", "256", "--warmup", "2", "--eval-every", "1000",
              "--eval-windows", "2", "--log-every", "1000", "--device", "cpu", "--seed", "7"]
    torch.use_deterministic_algorithms(True)
    try:
        train_main(common + ["--out", str(tmp_path / "full")])
        train_main(common + ["--out", str(tmp_path / "split"), "--stop-at", "3"])
        summary = train_main(common + ["--out", str(tmp_path / "split"), "--resume", "auto"])
    finally:
        torch.use_deterministic_algorithms(False)
    assert summary["step"] == 6
    full = torch.load(tmp_path / "full" / "ckpt.pt", weights_only=False)["model"]
    split = torch.load(tmp_path / "split" / "ckpt.pt", weights_only=False)["model"]
    worst = max((full[k].float() - split[k].float()).abs().max().item() for k in full)
    assert worst <= 1e-6, worst


def test_failure_saves_error_checkpoint_and_raises(tmp_path, monkeypatch):
    import json

    import palimpsa.train as train_mod

    root = make_data(tmp_path / "data")
    calls = {"n": 0}
    real_loss = train_mod.lm_loss

    def failing_loss(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 3:  # third optimizer step (one micro-batch per step)
            raise RuntimeError("CUDA out of memory (simulated)")
        return real_loss(*a, **kw)

    monkeypatch.setattr(train_mod, "lm_loss", failing_loss)
    out = tmp_path / "run"
    args = ["--stage", "1M", "--variant", "symA", "--data", str(root), "--steps", "6", "--seq-len", "64",
            "--micro-batch", "2", "--global-batch-tokens", "128", "--warmup", "2", "--eval-every", "1000",
            "--eval-windows", "2", "--log-every", "1", "--device", "cpu", "--out", str(out)]
    with pytest.raises(RuntimeError, match="simulated"):
        train_main(args)
    saved = torch.load(out / "ckpt_error.pt", weights_only=False)
    assert saved["step"] == 2 and not (out / "ckpt.pt").exists()
    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["event"] == "error" and rows[-1]["step"] == 2


def test_stop_file_ends_the_run_cleanly(tmp_path):
    root = make_data(tmp_path / "data")
    stop = tmp_path / "STOP"
    stop.write_text("")
    out = tmp_path / "run"
    summary = train_main(["--stage", "1M", "--variant", "symA", "--data", str(root), "--steps", "6", "--seq-len", "64",
                          "--micro-batch", "2", "--global-batch-tokens", "128", "--warmup", "2", "--eval-every",
                          "1000", "--eval-windows", "2", "--log-every", "1000", "--device", "cpu", "--out", str(out),
                          "--stop-file", str(stop)])
    assert summary["reason"] == "stop file" and summary["step"] == 1
    assert torch.load(out / "ckpt.pt", weights_only=False)["step"] == 1


def test_val_loss_smoke(tmp_path):
    root = make_data(tmp_path)
    model = PalimpsaLM(ladder_config("1M", "symA"))
    res = val_loss(model, root, seq_len=64, batch=4, max_windows=3, device="cpu", amp=False, breakdown=True)
    assert all(np.isfinite(res[d]) for d in DOMAINS) and np.isfinite(res["macro"])
    assert abs(sum(res["web_by_lmax_bucket"]["share"]) - 1) < 1e-9
    assert len(res["web_by_position"]) == 1
