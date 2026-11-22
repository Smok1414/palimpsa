"""Exact paired McNemar test from the discordant counts stored in the downstream-benchmark suite file.

For every benchmark and metric, `paired_vs_base` holds two counts: questions only the variant got right and questions
only Base got right. Under the null hypothesis both discordant outcomes are equally likely, so the two-sided exact
p-value is min(1, 2 * P(X <= min(a, b))) with X ~ Binomial(a + b, 1/2).

The p-values are NOT corrected for multiple comparisons, and they only describe sampling noise of the test set:
every model is a single training run (one seed), so seed-to-seed variation is not covered.

Usage: python scripts/paired_mcnemar.py [--suite results/benchmarks_user_measured.json]
                                         [--out results/benchmarks_paired_mcnemar.json]
"""
from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def mcnemar_exact(a: int, b: int) -> float:
    n, k = a + b, min(a, b)
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def build(suite: dict) -> dict:
    out: dict = {"method": "exact two-sided McNemar on paired discordant counts, vs Base, uncorrected", "results": {}}
    for bench, entries in suite["paired_vs_base"].items():
        out["results"][bench] = {}
        for key, c in entries.items():
            a, b = c["variant_only_correct"], c["base_only_correct"]
            out["results"][bench][key] = {"variant_only_correct": a, "base_only_correct": b,
                                          "p": mcnemar_exact(a, b)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(ROOT / "results" / "benchmarks_user_measured.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "benchmarks_paired_mcnemar.json"))
    args = ap.parse_args()
    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    res = build(suite)
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8", newline="\n")
    for bench, entries in res["results"].items():
        for key, r in entries.items():
            print(f"{bench:11s} {key:18s} {r['variant_only_correct']:4d} vs {r['base_only_correct']:4d}  p={r['p']:.4f}")


if __name__ == "__main__":
    main()
