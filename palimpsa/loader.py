"""Token data for training and evaluation.

Training batches are random contiguous windows drawn from memory-mapped uint16 streams. Every window is a pure
function of (seed, step, micro_step, rank), so a resumed run sees exactly the batches it would have seen.
Validation uses fixed, non-overlapping windows over the deduplicated validation streams.

Two layouts are supported: local `data/tokens/<domain>/<name>` and the flat Kaggle dataset `<domain>_<name>`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

EOS_ID = 0
DOMAINS = ("web", "code", "math")
MIX = {"web": 0.60, "code": 0.25, "math": 0.15}


def token_path(root: Path | str, domain: str, name: str) -> Path:
    root = Path(root)
    flat = root / f"{domain}_{name}"
    return flat if flat.exists() else root / domain / name


def doc_ids(x: np.ndarray) -> np.ndarray:
    """Document index of every position in (B, L) windows: it increases right after each EOS."""
    ids = np.zeros(x.shape, dtype=np.int64)
    ids[:, 1:] = np.cumsum(x[:, :-1] == EOS_ID, axis=1)
    return ids


class TrainStream:
    def __init__(self, root: Path | str, seq_len: int, mix: dict[str, float] | None = None, seed: int = 0):
        mix = mix or MIX
        self.seq_len, self.seed = seq_len, seed
        self.domains = [d for d in DOMAINS if mix.get(d, 0) > 0]
        weights = np.array([mix[d] for d in self.domains], dtype=np.float64)
        self.weights = weights / weights.sum()
        self.streams = [np.memmap(token_path(root, d, "train.bin"), dtype=np.uint16, mode="r")
                        for d in self.domains]
        for d, s in zip(self.domains, self.streams):
            if len(s) <= seq_len + 1:
                raise ValueError(f"{d} train stream is shorter than one window")

    def batch(self, step: int, micro: int, batch_size: int, rank: int = 0):
        """(x, y, doc, domain_index) for one micro-batch; x, y, doc are int64 tensors of shape (B, L)."""
        rng = np.random.default_rng([self.seed, step, micro, rank])
        which = rng.choice(len(self.domains), size=batch_size, p=self.weights)
        length = self.seq_len
        out = np.empty((batch_size, length + 1), dtype=np.int64)
        for i, k in enumerate(which):
            stream = self.streams[k]
            start = int(rng.integers(0, len(stream) - length - 1))
            out[i] = stream[start:start + length + 1]
        x, y = out[:, :-1], out[:, 1:]
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y)),
                torch.from_numpy(doc_ids(x)), which)


def val_windows(root: Path | str, domain: str, seq_len: int, max_windows: int | None = None):
    """Non-overlapping (x, y) windows over the deduplicated validation stream of a domain."""
    stream = np.fromfile(token_path(root, domain, "val.bin"), dtype=np.uint16).astype(np.int64)
    n = (len(stream) - 1) // seq_len
    if max_windows:
        n = min(n, max_windows)
    x = stream[: n * seq_len].reshape(n, seq_len)
    y = stream[1: n * seq_len + 1].reshape(n, seq_len)
    return x, y


def find_data_root(hint: str = "auto") -> Path:
    """`auto`: the Kaggle input folder that holds SHA256SUMS.txt, else the local data/tokens."""
    if hint != "auto":
        return Path(hint)
    for sums in sorted(Path("/kaggle/input").glob("**/SHA256SUMS.txt")):
        return sums.parent
    return Path(__file__).resolve().parents[1] / "data" / "tokens"
