"""Pre-upload data report: vocab, bytes/token, random documents, non-English share in web, exact duplicates.

Writes data/tokens/pre_upload_report.json. Prints UTF-8 (run with PYTHONIOENCODING=utf-8 on Windows).
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from palimpsa.data import DOMAINS, TOKENIZER, TOKENS, raw_files  # noqa: E402

EOS_ID = 0
CHUNK = 100_000_000
EN_FUNCTION_WORDS = {"the", "and", "of", "to", "in", "is", "that", "for", "it", "with", "as", "are", "this", "be", "on"}


def eos_positions(path: Path) -> np.ndarray:
    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    parts = [np.flatnonzero(np.asarray(tokens[lo:lo + CHUNK]) == EOS_ID) + lo for lo in range(0, len(tokens), CHUNK)]
    return np.concatenate(parts)


def decode_doc(tok: Tokenizer, tokens: np.ndarray, eos: np.ndarray, i: int) -> str:
    start = 0 if i == 0 else int(eos[i - 1]) + 1
    return tok.decode(tokens[start:int(eos[i])].tolist())


def is_latin_letter(ch: str) -> bool:
    return "LATIN" in unicodedata.name(ch, "")


def language_proxies(texts: list[str]) -> dict:
    non_latin_docs = low_en_docs = 0
    letters = non_latin = 0
    for text in texts:
        alpha = [c for c in text if c.isalpha()]
        nl = sum(1 for c in alpha if not is_latin_letter(c))
        letters += len(alpha)
        non_latin += nl
        if alpha and nl / len(alpha) > 0.10:
            non_latin_docs += 1
        words = [w.strip(".,;:!?()[]\"'").lower() for w in text.split()]
        words = [w for w in words if w]
        if len(words) >= 50 and sum(w in EN_FUNCTION_WORDS for w in words) / len(words) < 0.05:
            low_en_docs += 1
    n = max(len(texts), 1)
    return {"sample_docs": len(texts), "share_docs_non_latin_gt10pct": non_latin_docs / n,
            "share_letters_non_latin": non_latin / max(letters, 1),
            "share_docs_few_english_function_words": low_en_docs / n}


def main() -> None:
    tok = Tokenizer.from_file(str(TOKENIZER))
    rng = np.random.default_rng(2026)
    report: dict = {"vocab": tok.get_vocab_size(), "bytes_per_token": {}, "samples": {}, "duplicates": {}}
    held = json.loads((TOKENIZER.parent / "report.json").read_text(encoding="utf-8"))["bytes_per_token"]
    all_hashes, all_lengths = [], []
    for d in DOMAINS:
        val = np.fromfile(TOKENS / d / "val.bin", dtype=np.uint16)
        off = np.load(TOKENS / d / "val_offsets.npy")
        n_bytes = sum(len(tok.decode(val[off[i]:off[i + 1] - 1].tolist()).encode("utf-8")) for i in range(len(off) - 1))
        report["bytes_per_token"][d] = {"held_out_5MB": held[d], "val_dedup": round(n_bytes / (len(val) - (len(off) - 1)), 3)}

        train = np.memmap(TOKENS / d / "train.bin", dtype=np.uint16, mode="r")
        eos = eos_positions(TOKENS / d / "train.bin")
        hashes = np.load(TOKENS / d / "train_hashes.npy")
        lengths = np.diff(np.concatenate([[-1], eos]))  # tokens per document including its EOS
        if len(hashes) != len(eos):
            raise SystemExit(f"{d}: {len(eos)} EOS tokens but {len(hashes)} documents (literal EOS inside text?)")
        picks = sorted(rng.choice(len(eos), size=5, replace=False).tolist())
        report["samples"][d] = [decode_doc(tok, train, eos, i)[:300] for i in picks]
        _, first = np.unique(hashes, return_index=True)
        dup = np.ones(len(hashes), dtype=bool)
        dup[first] = False
        report["duplicates"][d] = {"docs": int(len(hashes)), "dup_docs": int(dup.sum()),
                                   "dup_doc_share": float(dup.mean()),
                                   "dup_token_share": float(lengths[dup].sum() / lengths.sum())}
        all_hashes.append(hashes)
        all_lengths.append(lengths)
        if d == "web":
            idx = rng.choice(len(eos), size=5000, replace=False)
            report["web_language_proxy_sample"] = language_proxies([decode_doc(tok, train, eos, int(i)) for i in idx])
    hashes, lengths = np.concatenate(all_hashes), np.concatenate(all_lengths)
    _, first = np.unique(hashes, return_index=True)
    dup = np.ones(len(hashes), dtype=bool)
    dup[first] = False
    report["duplicates"]["all_domains"] = {"docs": int(len(hashes)), "dup_docs": int(dup.sum()),
                                           "dup_doc_share": float(dup.mean()),
                                           "dup_token_share": float(lengths[dup].sum() / lengths.sum())}
    meta = {"docs": 0, "non_en": 0, "score_lt_0_65": 0, "score_lt_0_8": 0}
    for path in raw_files("web"):
        table = pq.read_table(path, columns=["language", "language_score"])
        lang = np.asarray(table.column("language").to_pylist())
        score = table.column("language_score").to_numpy()
        meta["docs"] += len(lang)
        meta["non_en"] += int((lang != "en").sum())
        meta["score_lt_0_65"] += int((score < 0.65).sum())
        meta["score_lt_0_8"] += int((score < 0.8).sum())
    report["web_language_metadata"] = meta
    out = TOKENS / "pre_upload_report.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
