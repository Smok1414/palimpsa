"""Deduplicate the held-out validation documents against all training data (every domain).

A validation document is dropped if
  (a) its normalised-text hash equals that of any training document or of an earlier validation document, or
  (b) at least OVERLAP_FRAC of its 13-gram token hashes occur anywhere in the training token streams.
Training data is left untouched. Writes data/tokens/<domain>/val.bin, val_offsets.npy and dedup.json.

Usage: python scripts/prep_dedup_val.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from palimpsa.data import DOMAINS, TOKENS  # noqa: E402

N = 13
OVERLAP_FRAC = 0.5
CHUNK = 8_000_000  # train tokens per scan step; peak RAM stays around 0.5 GB
PREFILTER_BITS = 28  # 32 MB bitmap over the top hash bits: cheap rejection before the binary search
_P = np.uint64(0x9E3779B97F4A7C15)
_ONE = np.uint64(1)
_SHIFT = np.uint64(64 - PREFILTER_BITS)


def ngram_hashes(tokens: np.ndarray) -> np.ndarray:
    m = len(tokens) - N + 1
    if m <= 0:
        return np.empty(0, dtype=np.uint64)
    t = tokens.astype(np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(N):  # in place: no temporaries of chunk size
            np.multiply(h, _P, out=h)
            np.add(h, t[k:k + m], out=h)
            np.add(h, _ONE, out=h)
    return h


def build_prefilter(table: np.ndarray) -> np.ndarray:
    bitmap = np.zeros(1 << (PREFILTER_BITS - 3), dtype=np.uint8)
    idx = table >> _SHIFT
    np.bitwise_or.at(bitmap, (idx >> np.uint64(3)).astype(np.int64),
                     (np.uint8(1) << (idx & np.uint64(7)).astype(np.uint8)))
    return bitmap


def maybe_in(bitmap: np.ndarray, h: np.ndarray) -> np.ndarray:
    idx = h >> _SHIFT
    return ((bitmap[(idx >> np.uint64(3)).astype(np.int64)] >> (idx & np.uint64(7)).astype(np.uint8)) & 1) == 1


def main() -> None:
    start = time.time()
    train_hash = np.unique(np.concatenate([np.load(TOKENS / d / "train_hashes.npy") for d in DOMAINS]))
    val = {}
    for d in DOMAINS:
        tokens = np.fromfile(TOKENS / d / "val_raw.bin", dtype=np.uint16)
        offsets = np.load(TOKENS / d / "val_raw_offsets.npy")
        hashes = np.load(TOKENS / d / "val_raw_hashes.npy")
        val[d] = {"tokens": tokens, "offsets": offsets, "hashes": hashes,
                  "keep": np.ones(len(hashes), dtype=bool), "drop_exact": 0, "drop_overlap": 0}

    seen: set[int] = set()  # exact: against train and earlier val documents
    for d in DOMAINS:
        v = val[d]
        in_train = np.isin(v["hashes"], train_hash)
        for i, h in enumerate(v["hashes"].tolist()):
            if in_train[i] or h in seen:
                v["keep"][i] = False
                v["drop_exact"] += 1
            seen.add(h)

    ngram, owner = [], []  # 13-grams of kept val documents (without the trailing EOS)
    doc_ids = []
    for d in DOMAINS:
        v = val[d]
        for i in np.flatnonzero(v["keep"]):
            h = ngram_hashes(v["tokens"][v["offsets"][i]:v["offsets"][i + 1] - 1])
            ngram.append(h)
            owner.append(np.full(len(h), len(doc_ids), dtype=np.int64))
            doc_ids.append((d, int(i)))
    ngram_all = np.concatenate(ngram) if ngram else np.empty(0, np.uint64)
    owner_all = np.concatenate(owner) if owner else np.empty(0, np.int64)
    table = np.unique(ngram_all)
    found = np.zeros(len(table), dtype=bool)
    bitmap = build_prefilter(table)
    print(f"val 13-grams: {len(ngram_all)} ({len(table)} unique) over {len(doc_ids)} docs", flush=True)

    scanned = 0
    for d in DOMAINS:
        train = np.memmap(TOKENS / d / "train.bin", dtype=np.uint16, mode="r")
        for lo in range(0, len(train), CHUNK):
            h = ngram_hashes(np.asarray(train[lo:min(len(train), lo + CHUNK + N - 1)]))
            scanned += len(h)
            h = h[maybe_in(bitmap, h)]
            pos = np.searchsorted(table, h)
            ok = pos < len(table)
            hit = pos[ok][table[pos[ok]] == h[ok]]
            found[hit] = True
            if (lo // CHUNK) % 25 == 0:
                print(f"  {d}: {scanned / 1e9:.2f}B train 13-grams, {time.time() - start:.0f}s", flush=True)
        print(f"  scanned {d}: total {scanned / 1e9:.2f}B train 13-grams, {time.time() - start:.0f}s", flush=True)

    frac = np.zeros(len(doc_ids))
    if len(ngram_all):
        hit_per_gram = found[np.searchsorted(table, ngram_all)].astype(np.float64)
        n_per_doc = np.bincount(owner_all, minlength=len(doc_ids))
        frac = np.bincount(owner_all, weights=hit_per_gram, minlength=len(doc_ids)) / np.maximum(n_per_doc, 1)
    for k, (d, i) in enumerate(doc_ids):
        if frac[k] >= OVERLAP_FRAC:
            val[d]["keep"][i] = False
            val[d]["drop_overlap"] += 1

    report = {}
    for d in DOMAINS:
        v = val[d]
        idx = np.flatnonzero(v["keep"])
        pieces = [v["tokens"][v["offsets"][i]:v["offsets"][i + 1]] for i in idx]
        out_tokens = np.concatenate(pieces) if pieces else np.empty(0, np.uint16)
        out_offsets = np.concatenate([[0], np.cumsum([len(p) for p in pieces])]).astype(np.int64)
        out_tokens.tofile(TOKENS / d / "val.bin")
        np.save(TOKENS / d / "val_offsets.npy", out_offsets)
        report[d] = {"val_raw_docs": len(v["hashes"]), "val_raw_tokens": int(len(v["tokens"])),
                     "dropped_exact": v["drop_exact"], "dropped_overlap_13gram": v["drop_overlap"],
                     "val_docs": int(len(idx)), "val_tokens": int(len(out_tokens))}
    report["settings"] = {"ngram": N, "overlap_frac": OVERLAP_FRAC, "seconds": round(time.time() - start)}
    (TOKENS / "dedup.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    main()
