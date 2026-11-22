"""Raw-document iteration, deterministic train/val split and hashing shared by the data scripts."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TOKENS = ROOT / "data" / "tokens"
TOKENIZER = ROOT / "data" / "tokenizer" / "tokenizer.json"
DOMAINS = ("web", "code", "math")
# Held-out fraction of documents per domain, chosen for ~2M validation tokens each.
VAL_FRACTION = {"web": 0.0014, "code": 0.0045, "math": 0.007}
EOS = "<|endoftext|>"
PAD = "<|pad|>"


def raw_files(domain: str) -> list[Path]:
    manifest = json.loads((RAW / "manifest.json").read_text(encoding="utf-8"))
    files = [ROOT / f["local"] for f in manifest["files"] if f["domain"] == domain and f["ok"]]
    return sorted(files)


def iter_docs(path: Path, batch_size: int = 2048) -> Iterator[str]:
    if path.suffix == ".parquet":
        for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=batch_size):
            yield from (t for t in batch.column(0).to_pylist() if t)
    elif path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                text = json.loads(line).get("text")
                if text:
                    yield text
    else:
        raise ValueError(f"unsupported raw file {path}")


def _h64(data: bytes) -> int:
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def doc_hash(text: str) -> int:
    """Hash of the raw text; decides the train/val split."""
    return _h64(text.encode("utf-8"))


def norm_hash(text: str) -> int:
    """Hash after lowercasing and collapsing whitespace; used for exact deduplication."""
    return _h64(" ".join(text.lower().split()).encode("utf-8"))


def is_val(text_hash: int, domain: str) -> bool:
    return text_hash < VAL_FRACTION[domain] * 2**64
