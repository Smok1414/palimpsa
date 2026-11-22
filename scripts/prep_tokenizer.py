"""Train the shared 32k byte-level BPE tokenizer on a 60/25/15 web/code/math sample.

Digits are split one per token; whitespace runs are left to BPE (code indentation gets merged tokens).
Validation documents are excluded from the training sample. Reports bytes/token per domain on held-out text.

Usage: python scripts/prep_tokenizer.py [--sample-mb 300]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from palimpsa.data import DOMAINS, EOS, PAD, TOKENIZER, doc_hash, is_val, iter_docs, raw_files  # noqa: E402

MIX = {"web": 0.60, "code": 0.25, "math": 0.15}
VOCAB = 32000


def sample_docs(domain: str, budget: int, skip: int = 0) -> list[str]:
    """Train-split documents of a domain until `budget` UTF-8 bytes, after skipping `skip` bytes."""
    out, used, skipped = [], 0, 0
    for path in raw_files(domain):
        for text in iter_docs(path):
            if is_val(doc_hash(text), domain):
                continue
            n = len(text.encode("utf-8"))
            if skipped < skip:
                skipped += n
                continue
            out.append(text)
            used += n
            if used >= budget:
                return out
    return out


def build_tokenizer() -> Tokenizer:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])
    tok.decoder = decoders.ByteLevel()
    return tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-mb", type=float, default=300.0)
    args = ap.parse_args()
    start = time.time()
    docs: dict[str, list[str]] = {}
    for domain in DOMAINS:
        docs[domain] = sample_docs(domain, int(args.sample_mb * 1e6 * MIX[domain]))
        print(f"sample {domain}: {len(docs[domain])} docs, "
              f"{sum(len(t.encode('utf-8')) for t in docs[domain]) / 1e6:.1f} MB", flush=True)
    tok = build_tokenizer()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB, min_frequency=2, special_tokens=[EOS, PAD],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    corpus = [t for d in DOMAINS for t in docs[d]]
    tok.train_from_iterator(corpus, trainer=trainer, length=len(corpus))
    TOKENIZER.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(TOKENIZER))
    print(f"trained vocab {tok.get_vocab_size()} in {time.time() - start:.0f}s -> {TOKENIZER}", flush=True)

    report = {"vocab": tok.get_vocab_size(), "eos_id": tok.token_to_id(EOS), "pad_id": tok.token_to_id(PAD),
              "sample_mb": args.sample_mb, "bytes_per_token": {}}
    for domain in DOMAINS:  # held-out text: the next 5 MB of train-split documents after the sample
        held = sample_docs(domain, 5_000_000, skip=int(args.sample_mb * 1e6 * MIX[domain]))
        n_bytes = sum(len(t.encode("utf-8")) for t in held)
        n_tok = sum(len(e.ids) for e in tok.encode_batch(held))
        report["bytes_per_token"][domain] = round(n_bytes / max(n_tok, 1), 3)
    probe = "def f(x):\n        return x + 12345  # indent 8\n"
    report["probe_tokens"] = tok.encode(probe).tokens
    out = TOKENIZER.parent / "report.json"
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1), flush=True)  # ASCII escapes: the console may not be UTF-8


if __name__ == "__main__":
    main()
