"""Tokenize raw documents into uint16 token files with an EOS after every document.

Per domain, in data/tokens/<domain>/:
  train.bin          uint16 tokens of training documents
  train_hashes.npy   uint64 normalised-text hash of every training document (for exact dedup)
  val_raw.bin        uint16 tokens of held-out documents (before dedup)
  val_raw_offsets.npy int64 document boundaries into val_raw.bin (n_docs + 1)
  val_raw_hashes.npy uint64 normalised-text hash of every held-out document
  stats.json

Usage: python scripts/prep_tokenize.py --domains web,code,math
       python scripts/prep_tokenize.py --count-only --domains code   # tokens per raw file, nothing written
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from palimpsa.data import EOS, PAD, TOKENIZER, TOKENS, doc_hash, is_val, iter_docs, norm_hash, raw_files  # noqa: E402

BATCH_DOCS = 2048


def batches(path: Path):
    buf = []
    for text in iter_docs(path):
        buf.append(text)
        if len(buf) == BATCH_DOCS:
            yield buf
            buf = []
    if buf:
        yield buf


def count_only(domain: str, tok: Tokenizer) -> None:
    for path in raw_files(domain):
        n_tok = n_docs = n_bytes = 0
        for texts in batches(path):
            n_tok += sum(len(e.ids) + 1 for e in tok.encode_batch(texts))
            n_docs += len(texts)
            n_bytes += sum(len(t.encode("utf-8")) for t in texts)
        print(f"{domain} {path.name}: {n_docs} docs, {n_bytes / 1e6:.1f} MB text, {n_tok / 1e6:.2f}M tokens",
              flush=True)


def tokenize_domain(domain: str, tok: Tokenizer, eos_id: int, pad_id: int) -> dict:
    out = TOKENS / domain
    out.mkdir(parents=True, exist_ok=True)
    train_hashes: list[int] = []
    val_hashes: list[int] = []
    val_offsets = [0]
    n_train = n_val = 0
    start = time.time()
    with (out / "train.bin").open("wb") as f_train, (out / "val_raw.bin").open("wb") as f_val:
        for path in raw_files(domain):
            for texts in batches(path):
                encoded = tok.encode_batch(texts)
                train_chunks, val_chunks = [], []
                for text, enc in zip(texts, encoded):
                    ids = enc.ids + [eos_id]
                    if is_val(doc_hash(text), domain):
                        val_chunks.append(ids)
                        val_hashes.append(norm_hash(text))
                        n_val += len(ids)
                        val_offsets.append(n_val)
                    else:
                        train_chunks.append(ids)
                        train_hashes.append(norm_hash(text))
                        n_train += len(ids)
                for chunks, fh in ((train_chunks, f_train), (val_chunks, f_val)):
                    if not chunks:
                        continue
                    arr = np.concatenate([np.asarray(c, dtype=np.uint16) for c in chunks])
                    # EOS only where we put it, never from text; PAD never appears in data
                    if int((arr == eos_id).sum()) != len(chunks) or int((arr == pad_id).sum()) != 0:
                        raise SystemExit(f"special token leaked from text in {path.name}")
                    fh.write(arr.tobytes())
            print(f"  {domain} {path.name}: train {n_train / 1e6:.1f}M, val {n_val / 1e6:.2f}M tokens, "
                  f"{time.time() - start:.0f}s", flush=True)
    np.save(out / "train_hashes.npy", np.asarray(train_hashes, dtype=np.uint64))
    np.save(out / "val_raw_hashes.npy", np.asarray(val_hashes, dtype=np.uint64))
    np.save(out / "val_raw_offsets.npy", np.asarray(val_offsets, dtype=np.int64))
    stats = {"domain": domain, "train_tokens": n_train, "val_raw_tokens": n_val,
             "train_docs": len(train_hashes), "val_raw_docs": len(val_hashes),
             "files": [p.name for p in raw_files(domain)], "seconds": round(time.time() - start)}
    (out / "stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", default="web,code,math")
    ap.add_argument("--count-only", action="store_true")
    args = ap.parse_args()
    tok = Tokenizer.from_file(str(TOKENIZER))
    tok.encode_special_tokens = True  # literal "<|endoftext|>" / "<|pad|>" in documents are plain text
    eos_id, pad_id = tok.token_to_id(EOS), tok.token_to_id(PAD)
    if tok.get_vocab_size() > np.iinfo(np.uint16).max:
        raise SystemExit("vocabulary does not fit uint16")
    for domain in args.domains.split(","):
        if args.count_only:
            count_only(domain, tok)
        else:
            print(json.dumps(tokenize_domain(domain, tok, eos_id, pad_id)), flush=True)


if __name__ == "__main__":
    main()
