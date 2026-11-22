"""Synthetic in-context memory tasks for stage S1 (gate G1, section 7.3).

Every generator returns (x, y): x (B, L) token ids, y (B, L) targets where y[b, t] is the expected token
x[b, t + 1] at scored positions and IGNORE elsewhere. Only answers are scored.

mqar1  - 32 single-token keys (alphabet 64) -> values; 16 queries. Sanity task.
mqar3  - 32 distinct 3-token keys over an alphabet of 4, so 1- and 2-token suffixes are ambiguous and
         only the full key identifies the value; 16 queries.
fuzzy3 - mqar3 where one token of every query key is replaced by its synonym (a fixed bijection learned
         from data): exact matching is broken or misleading, content addressing has to do the work.
trap   - flag COPY: answer = value of the queried key; flag PREV: answer = value of the pair *before* it.
         Exact continuation copying gives the wrong answer for PREV.
copy   - a random 48-token string (alphabet 64) repeated after SEP; every continuation is scored.
"""
from __future__ import annotations

import random

import torch

PAD, BOS, SEP, FLAG_COPY, FLAG_PREV = 0, 1, 2, 3, 4
KEY0, N_KEY = 16, 64
SYN0 = KEY0 + N_KEY  # synonym of KEY0 + i is SYN0 + i
VAL0, N_VAL = SYN0 + N_KEY, 112
VOCAB = VAL0 + N_VAL  # 256
IGNORE = -100


def _pack(seqs: list[list[int]], targets: list[list[tuple[int, int]]], length: int):
    x = torch.full((len(seqs), length), PAD, dtype=torch.long)
    y = torch.full((len(seqs), length), IGNORE, dtype=torch.long)
    for b, (seq, tgt) in enumerate(zip(seqs, targets)):
        if len(seq) > length:
            raise ValueError(f"sequence of {len(seq)} tokens does not fit length {length}")
        x[b, : len(seq)] = torch.tensor(seq)
        for pos, tok in tgt:
            if x[b, pos + 1] != tok:
                raise AssertionError("target must equal the next input token")
            y[b, pos] = tok
    return x, y


def mqar(batch: int, rng: random.Random, n_pairs: int, n_queries: int, key_len: int, alphabet: int,
         fuzzy: bool = False, length: int = 256):
    seqs, targets = [], []
    for _ in range(batch):
        codes = rng.sample(range(alphabet ** key_len), n_pairs)
        keys = [[KEY0 + (c // alphabet ** p) % alphabet for p in reversed(range(key_len))] for c in codes]
        vals = [VAL0 + rng.randrange(N_VAL) for _ in range(n_pairs)]
        seq = [BOS]
        for key, val in zip(keys, vals):
            seq += key + [val]
        seq.append(SEP)
        tgt = []
        for i in rng.sample(range(n_pairs), n_queries):
            key = list(keys[i])
            if fuzzy:
                key[rng.randrange(key_len)] += N_KEY
            seq += key
            tgt.append((len(seq) - 1, vals[i]))
            seq.append(vals[i])
        seqs.append(seq)
        targets.append(tgt)
    return _pack(seqs, targets, length)


def trap(batch: int, rng: random.Random, n_pairs: int = 24, n_queries: int = 12, length: int = 128):
    seqs, targets = [], []
    for _ in range(batch):
        flag = rng.choice((FLAG_COPY, FLAG_PREV))
        keys = [KEY0 + k for k in rng.sample(range(N_KEY), n_pairs)]
        vals = [VAL0 + rng.randrange(N_VAL) for _ in range(n_pairs)]
        seq = [BOS, flag]
        for key, val in zip(keys, vals):
            seq += [key, val]
        seq.append(SEP)
        tgt = []
        for i in rng.sample(range(1, n_pairs), n_queries):
            answer = vals[i] if flag == FLAG_COPY else vals[i - 1]
            seq.append(keys[i])
            tgt.append((len(seq) - 1, answer))
            seq.append(answer)
        seqs.append(seq)
        targets.append(tgt)
    return _pack(seqs, targets, length)


def copy_task(batch: int, rng: random.Random, n_tokens: int = 48, alphabet: int = 64, length: int = 128):
    seqs, targets = [], []
    for _ in range(batch):
        s = [KEY0 + rng.randrange(alphabet) for _ in range(n_tokens)]
        seq = [BOS] + s + [SEP] + s
        second = n_tokens + 2  # index of s[0] in the second copy
        seqs.append(seq)
        targets.append([(second + i, s[i + 1]) for i in range(n_tokens - 1)])
    return _pack(seqs, targets, length)


TASKS = {
    "mqar1_easy": lambda b, rng: mqar(b, rng, n_pairs=8, n_queries=8, key_len=1, alphabet=64, length=64),
    "mqar1": lambda b, rng: mqar(b, rng, n_pairs=32, n_queries=16, key_len=1, alphabet=64, length=128),
    "mqar3": lambda b, rng: mqar(b, rng, n_pairs=32, n_queries=16, key_len=3, alphabet=4, length=256),
    "fuzzy3": lambda b, rng: mqar(b, rng, n_pairs=32, n_queries=16, key_len=3, alphabet=4, fuzzy=True,
                                  length=256),
    "trap": lambda b, rng: trap(b, rng),
    "copy": lambda b, rng: copy_task(b, rng),
}
