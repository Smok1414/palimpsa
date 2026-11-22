# Palimpsa

Research code for **SymA**: softmax attention with no positional encoding (NoPE) whose logits get an
exact-match-length (ℓ) bias and a FoX-style forget-gate bias. It is compared with a Llama-style baseline (RoPE,
SwiGLU, RMSNorm, QK-norm, grouped-query attention). "Palimpsa" is a working name.

**Status: research in progress.** Snapshot of 2026-11-22. Stage S2 (10M-class models) is still running: the SymA + RoPE
ablation (A2) and the long runs for gate G2c are not finished. No 30M or 100M result exists yet. Model weights are not
part of this repository; they will be published separately (for example on Hugging Face) once that is decided.
License: **TBD by owner** (`LICENSE` is a placeholder, see the end of this file).

## What SymA is

For a query at position `t` and an earlier key `j`, the attention logit is

    q·k / 8  +  b[h, bucket(ℓ(t, j))]  +  (F[h, t] - F[h, j])

- `ℓ(t, j)` is the length of the longest common suffix of the token histories `x[..t]` and `x[..j-1]`, capped at 16 and
  never crossing a document boundary. Key `j` is the position right after an earlier occurrence of the current suffix.
  It is mapped to 8 buckets (0, 1, 2, 3, 4, 5-7, 8-15, 16); each head learns one scalar for each bucket except bucket 0
  ("no match"), which is fixed at 0 (`palimpsa/symbolic.py`, `palimpsa/model.py`).
- `F` is the cumulative sum of log-sigmoid forget gates, one per head and token (FoX).
- No RoPE. QK-norm is used in every model that is compared here.

One code path (`palimpsa/model.py`) builds every variant:

| Variant (`--variant`) | RoPE | Forget gate | ℓ bias | Role | Result in this snapshot |
|---|:---:|:---:|:---:|---|---|
| `base` | yes | no | no | baseline | yes |
| `symA` | no | yes | yes | proposed model | yes |
| `fox` | no | yes | no | A1: SymA without ℓ | yes |
| `ell_rope` | yes | no | yes | A3: baseline plus ℓ | yes |
| `symA_rope` | yes | yes | yes | A2: SymA plus RoPE | **not finished** (run in progress) |
| `nope`, `ell_nope` | no | no | no / yes | tests and optional ablation | none |

The code also contains two further candidates, a gated residual stream (`--residual gr`) and a sigmoid gate on the
attention output (`--attn-gate`). They have unit tests but **no training results**, and they are not part of the
claims below.

## Measured results (single seed)

Every number in this section comes from a file in [`results/`](results/) (see
[`docs/PROVENANCE.md`](docs/PROVENANCE.md) for the number-to-file table). **Every training result is a single run
(one seed), so all differences between models are tentative.** The project assumes a one-seed noise level of up to
0.012 nats/token in validation loss. That is a planning assumption: **the real seed-to-seed noise has not been
measured yet.** It can only be estimated after a second seed has been run (proposed, not started).

Setup of the 10M-class runs: d_model 384, 6 layers, 6 heads with 2 KV heads, d_ff 1024, tied embeddings, 32,000-token
byte-level BPE. 21,730,944 parameters for Base and 21,745,056 for SymA, of which 12,288,000 are the
embedding table, so the non-embedding count is 9,442,944 (Base) and 9,457,056 (SymA); "10M" is the
name of the stage, not the exact count. Sequence length 1,024, global batch 65,536 tokens, AdamW (0.9, 0.95), weight
decay 0.1, gradient clipping 1.0, z-loss 1e-4, fp16 autocast with a gradient scaler, 200 warmup steps, constant LR, then
linear decay over the last 15% of the run. Seed 1; runs that share a seed see identical batches. Validation: 1,953
windows of 1,024 tokens per domain from held-out, deduplicated documents; "macro" is the unweighted mean over web, code
and math, in nats per token.

### G0: speed and memory (SymA relative to Base)

| Model size | Setting | Base tok/s | SymA tok/s | SymA / Base | Peak memory, GiB (Base / SymA) |
|---|---|---:|---:|---:|---:|
| 100M | 1 T4, micro-batch 4, 40 steps | 12,702 | 10,418 | 0.820 | 4.91 / 6.12 |
| 10M | 1 T4, micro-batch 16, 50 steps | 39,039 | 32,563 | 0.834 | 5.09 / 6.30 |
| 30M | one process per T4, micro-batch 16, 20 steps | 23,199 | 18,679 | 0.805 | 8.50 / 11.22 |
| 10M | S2-1 LR-sweep runs (lr 3e-3, 1,526 steps), one process per T4 | 40,346 | 32,891 | 0.815 | 6.12 / 6.34 |
| 10M | S2-2 long runs (5,642 steps), one process per T4 | 37,743 | 32,680 | 0.866 | 6.12 / 6.34 |

SymA is slower and uses more memory than Base in every measurement. The ratio depends on the model size and on the
session (0.805 to 0.866 here). The first three rows are short probes (20 to 50 steps); the last two
are medians over full runs. In the S2-3a session A1 (FoX, no ℓ) ran at 32,345 tok/s and A3 (ℓ + RoPE) at
31,920 tok/s, against 32,680 tok/s for SymA in S2-2 (different sessions). FoX alone is about as slow as
SymA, so the cost comes from the additive-bias attention path rather than from ℓ itself. Hardware: Tesla T4, fp16.

### G1: synthetic in-context memory (seed 0, 3,000 steps, ~886,400 parameters)

Final accuracy in %, 4-layer models with d_model 128. mqar1/mqar3: multi-query associative recall with 1-token and
3-token keys; fuzzy3: mqar3 with one key token replaced by a synonym; trap: the answer is the value of the queried key or
of the pair before it, by a flag; copy: repeat a 48-token string. Run locally on one CUDA GPU with torch 2.11.0+cu128
(first line of `results/g1_synthetic.log`).

| Task | Base | SymA | FoX |
|---|---:|---:|---:|
| mqar1 | 6.2 | 100.0 | 4.6 |
| mqar3 | 6.8 | 100.0 | 5.1 |
| fuzzy3 | 5.8 | 23.7 | 3.8 |
| trap | 7.2 | 100.0 | 7.7 |
| copy | 100.0 | 99.9 | 4.4 |

Gate G1 result recorded by the script: `PASS (b, c uninformative: Base did not learn mqar1)`. The caveat is real: Base reached only 6.2% on
mqar1, so it never learned in-context recall at this budget, and the "no harm" criteria (trap, fuzzy3) say nothing about
real language. Positional copy works for Base and SymA (not for FoX). The leakage canary (loss on i.i.d. uniform tokens,
ln 64 = 4.1589) gave 4.1603 (Base) and 4.1602 (SymA), and 1.9261 for the deliberately
leaky positive control, so the canary can detect a leak.

### LR sweep, 0.1B tokens (1,526 steps, seed 1)

Final validation macro loss. Both architectures have their minimum at 3e-3 (chosen: Base 3e-3, SymA
3e-3).

| Peak LR | Base | SymA | SymA - Base |
|---:|---:|---:|---:|
| 1.5e-3 | 3.0702 | 3.0336 | -0.0366 |
| 3e-3 | **3.0495** | **3.0151** | -0.0344 |
| 6e-3 | 3.0593 | 3.0296 | -0.0297 |

### G2a: SymA vs Base at 0.435B tokens (D/N = 20, seed 1)

| Val loss, nats/token | Base | SymA | SymA - Base |
|---|---:|---:|---:|
| macro | 2.7309 | 2.7081 | -0.0228 |
| web | 3.6531 | 3.6337 | -0.0195 |
| code | 1.9202 | 1.8976 | -0.0227 |
| math | 2.6193 | 2.5930 | -0.0263 |

Gate verdict from the script: `continue to G2b` (the gate stops only if the macro difference is ≥ 0 or any domain is worse
by more than 0.010). The Base and SymA numbers are decay branches resumed from the stable checkpoint of a longer run
(step 5,642, then 6,638 steps in total).

### G2b: where does the gain come from?

Standalone runs with the same schedule and seed (A1 = FoX without ℓ, A3 = ℓ with RoPE and no forget gate):

| Val loss, nats/token | Base | A1 FoX (no RoPE, no ℓ) | A3 ℓ + RoPE | SymA (FoX + ℓ, no RoPE) |
|---|---:|---:|---:|---:|
| macro | 2.7309 | 2.7255 | 2.7124 | 2.7081 |
| web | 3.6531 | 3.6463 | 3.6375 | 3.6337 |
| code | 1.9202 | 1.9195 | 1.9006 | 1.8976 |
| math | 2.6193 | 2.6108 | 2.5992 | 2.5930 |

Differences of validation loss (negative = better):

| Contrast | macro | web | code | math |
|---|---:|---:|---:|---:|
| ℓ, without RoPE (SymA - A1) | -0.0175 | -0.0126 | -0.0220 | -0.0178 |
| ℓ, with RoPE (A3 - Base) | -0.0184 | -0.0156 | -0.0196 | -0.0201 |
| forget gate + no RoPE vs RoPE, no ℓ (A1 - Base) | -0.0054 | -0.0068 | -0.0007 | -0.0085 |
| forget gate + no RoPE vs RoPE, with ℓ (SymA - A3) | -0.0044 | -0.0038 | -0.0030 | -0.0062 |
| everything together (SymA - Base) | -0.0228 | -0.0195 | -0.0227 | -0.0263 |

Gate verdict from the script: `continue`. **Tentative reading (one seed):** most of the SymA-vs-Base difference
appears to come from the ℓ bias (about -0.0175 without RoPE and -0.0184 with RoPE). The
forget gate together with removing RoPE adds only -0.0054 and -0.0044 in the two
settings, which is at the edge of the assumed (not measured) one-seed noise. These two effects are not separated yet:
that needs A2 (SymA + RoPE), which is not finished.

## What is NOT established yet

- **G2c: does the advantage shrink with more data?** The three available points of the macro gap (SymA - Base) are
  -0.0344 (0.1B-token sweep), -0.0279 (step 5,500 of the long runs, before the LR decay) and
  -0.0228 (after decay at 0.435B). They come from different schedules, and the gap is shrinking along them. We do
  not consider this settled in either direction. The runs to D/N = 60 that gate G2c needs have not been done.
- **S3 (30M models, three seeds, gate G3) and the 100M model:** not started. Only speed probes exist for these sizes.
- **A2 (SymA + RoPE):** not finished. Until then the forget gate and the removal of RoPE cannot be told apart.
- **Seed variance:** every training result above is one run. No confidence interval over seeds exists.
- **Training-data overlap with the downstream benchmarks** was not checked (`training_data_overlap_checked: false` in the
  benchmark files).
- BLiMP and LAMBADA evaluation code is included (`palimpsa/eval.py`); no BLiMP or LAMBADA result on the trained 10M
  models is reported here.

## Benchmarks (user-measured; run by the owner, harness not included)

Zero-shot accuracies of the four 10M-class models at step 6,638, run by the project owner on a local GPU
(float32, 0-shot, log-likelihood scoring; each cell is `acc / acc_norm` in %). **The evaluation harness is not
included in this repository, so these numbers cannot be reproduced from it.** The prompt format, scoring rule, dataset
URL and dataset sha256 of every benchmark are recorded in `results/benchmarks/`, so the protocol can be re-implemented.

| Model | ARC-Easy | HellaSwag | PIQA | MathQA |
|---|---:|---:|---:|---:|
| Base | 40.9 / 37.2 | 26.7 / 27.1 | 56.2 / 55.3 | 20.3 / 20.2 |
| SymA | 40.3 / 36.1 | 26.8 / 27.5 | 56.1 / 55.2 | 20.8 / 21.2 |
| FoX (A1) | 40.1 / 36.0 | 26.4 / 27.0 | 56.6 / 55.3 | 21.0 / 22.0 |
| ℓ+RoPE (A3) | 40.4 / 36.7 | 26.7 / 27.2 | 57.5 / 56.0 | 20.9 / 20.8 |
| uniform chance | 25.0 | 25.0 | 50.0 | 20.0 |
| test items | 2,376 | 10,042 | 1,838 | 4,475 |

ARC-Easy is clearly above chance for all four models; HellaSwag and MathQA are at or near chance and PIQA is not far above
it. No variant differs from Base by more than 1.9 percentage points in any cell. Paired exact McNemar p-values on the per-question discordant counts
(`scripts/paired_mcnemar.py`, uncorrected, and covering only test-set sampling noise, not seed variance) are in
`results/benchmarks_paired_mcnemar.json`. Of the 24 comparisons with Base, the ones below nominal p = 0.05 are:
MathQA acc_norm, SymA vs Base: +1.0 pp (p = 0.0323); MathQA acc_norm, FoX vs Base: +1.9 pp (p = 0.0009). The smallest p-value on ARC-Easy is 0.121, on HellaSwag 0.166 and on PIQA
0.091. A single seed cannot tell whether the MathQA differences are an architecture effect; they are
small and close to the 20% chance level, so we do not interpret them.

**ArithMark-3 is not used.** The raw scores are in the results file (32.1% to 36.5% for the four models)
but are not interpreted. The owner considers the benchmark non-diagnostic and reports that a trivial untrained baseline
scores above the trained models on it. **That check is not backed by any result file in this repository yet, so it is an
unverified statement and not part of the measured results.**

### Sampling and repetition (local check)

Share of generated texts that repeat (in %, 100 texts per model: 50 web and 50 code prompts, 32-token
prompts from validation documents, 120 new tokens, pure temperature sampling at T = 0.3 and 0.7 without
top-k/top-p, seed 0). A text counts as repeating if a periodic tail covers at least 25% of it or its distinct-4-gram
ratio is below 0.5. Real continuations of the same documents score 0.0% (distinct-4: 0.99 web,
0.92 code). Greedy decoding was **not** measured. The last two columns give the difference in percentage
points and an unpaired two-proportion p-value, uncorrected.

| Penalty | T | Base | ℓ+RoPE (A3) | FoX (A1) | SymA | A3 - Base, pp (p) | SymA - A1, pp (p) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.3 | 84 | 87 | 85 | 90 | +3 (0.547) | +5 (0.285) |
| 1.0 | 0.7 | 23 | 32 | 19 | 25 | +9 (0.154) | +6 (0.306) |
| 1.2 | 0.3 | 39 | 57 | 43 | 45 | +18 (0.011) | +2 (0.776) |
| 1.2 | 0.7 | 1 | 2 | 1 | 3 | +1 (0.561) | +2 (0.312) |

The model with ℓ repeats more often than its counterpart without ℓ in 8 of 8 comparisons, but the differences are
small and mostly not significant (only A3 - Base at penalty 1.2, T 0.3 is below 0.05, and it does not survive a correction
for 8 tests). With a repetition penalty of 1.2 and T = 0.7 all four models repeat in at most 3% of the texts.
The direction is consistent with the idea that the ℓ bias rewards copying after a matching suffix; the evidence is weak.

## Limitations

- One seed for every trained model; small scale (21,730,944 parameters in total, 9,442,944 non-embedding) and
  a 0.435B-token budget; the code data is Python only.
- The SymA - Base gap at D/N = 20 (-0.0228) is only 1.9 times the assumed one-seed noise level of
  0.012. That level is an assumption: the real noise has not been measured. The ℓ and forget-gate effects are only
  partly separated.
- Speed and memory are measured on short probes and on one GPU type (T4). SymA is slower and uses more memory than Base.
- Resuming on CUDA is not bit-exact. Two identical uninterrupted runs differ by up to 0.084 in loss within
  25 steps, so the planned resume check with tolerance 0.01 could not be confirmed on CUDA (after a resume the
  difference was 0.024). Resume equivalence is covered by the deterministic CPU test in
  `tests/test_t9_train.py`.
- Generated text can repeat (see above); no quality evaluation of free-form generation exists.

## Data

Documents are tokenized with a 32k byte-level BPE tokenizer (EOS id 0, PAD id 1, digits split) trained on a
300 MB sample (60% web, 25% code, 15% math); bytes per token on held-out text: 4.414 web,
3.377 code, 3.018 math. Training windows are random 1,025-token windows; each row of a batch draws
its domain with the probabilities below. Validation documents are held out by document hash, then removed if they match a
training document exactly or share at least 50% of their 13-gram hashes with training data.

| Domain | Source | Mixture weight | Train tokens | Train docs | Val tokens (deduplicated) | Val docs |
|---|---|---:|---:|---:|---:|---:|
| web | FineWeb-Edu `sample/10BT` (2 of 14 shards) | 60% | 1,585,818,532 | 1,452,946 | 2,185,289 | 1,991 |
| code | StarCoder Python from `olmo-mix-1124` (5 files) | 25% | 527,949,558 | 332,386 | 2,444,229 | 1,494 |
| math | FineMath `finemath-4plus` (2 of 64 shards) | 15% | 333,731,463 | 207,887 | 2,058,640 | 1,366 |

Sources (public, not gated at the time of download; license metadata as listed on the dataset cards on 2026-11-21):

- [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), `sample/10BT`, ODC-By 1.0
  and the CommonCrawl terms of use.
- [`allenai/olmo-mix-1124`](https://huggingface.co/datasets/allenai/olmo-mix-1124), StarCoder Python subset, ODC-By 1.0
  (the StarCoder part is code from permissively licensed GitHub repositories).
- [`HuggingFaceTB/finemath`](https://huggingface.co/datasets/HuggingFaceTB/finemath), `finemath-4plus`, ODC-By 1.0 and
  the CommonCrawl terms of use.
- Evaluation: [`nyu-mll/blimp`](https://huggingface.co/datasets/nyu-mll/blimp) (CC BY 4.0) and
  [`EleutherAI/lambada_openai`](https://huggingface.co/datasets/EleutherAI/lambada_openai) (MIT per its card).

**The code license is stated at the dataset level only. We did not check licenses at the level of individual files, so
this repository does not distribute any data, tokens, `train.bin` or `val.bin` files.** `configs/raw_files.json` lists the
exact files with sizes and sha256; `configs/hf_revisions_pin.json` pins the dataset revisions that were used.

## Reproduce

Environment of the checks done for this snapshot: Python 3.14.3, torch 2.11.0 (CPU), numpy 2.4.3, pytest 9.0.3 on
Windows 11. The 10M runs above were produced on Kaggle (two Tesla T4 per session, Python 3.13 image, fp16); the torch
version of those sessions is not recorded in the result files. G1 and the benchmark and repetition checks ran locally. The tests need only `torch`, `numpy` and `pytest`
(`pip install -r requirements.txt`); the data scripts also need `pyarrow`, `tokenizers` and `huggingface_hub`.

```bash
# tests (CPU is enough): T1-T12, one file per topic
python -m pytest -q

# tiny CPU training run on real or synthetic token files (layout: <root>/<web|code|math>/{train,val}.bin)
python -m palimpsa.train --stage 1M --variant symA --steps 20 --micro-batch 2 --global-batch-tokens 2048 \
    --warmup 5 --eval-every 20 --eval-windows 4 --log-every 5 --data <token-root> --out runs/smoke --device cpu

# synthetic suite S1 (G1); --smoke runs 30 steps per run
python scripts/run_s1.py --smoke

# the S2 run table and the exact train.py command of every run (nothing is executed)
python scripts/plan_s2.py --commands
```

The G2a/G2b runs use `--stage 10M --seed 1 --lr 3e-3 --warmup 200 --micro-batch 16 --global-batch-tokens 65536`
(see the printed commands for the step counts and decay windows). `scripts/kaggle_s2.py` is the Kaggle orchestration
script of those sessions; its gate rules (`gate_g2a`, `gate_g2b`) are covered by `tests/test_t11_s2_rules.py`.

Data preparation, in this order (it downloads about 5.4 GB; this was done once in the original working copy and was not
repeated for this snapshot):

```bash
mkdir -p data/raw && cp configs/hf_revisions_pin.json data/raw/manifest.json   # pin the dataset revisions
python scripts/prep_download.py --domains web,math,code,eval --code-files 5
python scripts/prep_tokenizer.py
python scripts/prep_tokenize.py --domains web,code,math
python scripts/prep_dedup_val.py
```

`prep_tokenizer.py` writes `data/tokenizer/tokenizer.json`, which `palimpsa.eval` and `scripts/generate.py` expect. The
tokenizer file is not included in the repository.

## Repository layout

    palimpsa/    model, symbolic (ℓ and the fused bias), loader, train, eval, sampling, synthetic tasks, data helpers
    scripts/     data preparation, synthetic suite S1, S2 plan and Kaggle runner, generation, paired McNemar test
    tests/       T1-T12
    results/     result files behind every number in this README (no paths, no private identifiers)
    configs/     dataset revisions and the sha256 list of the raw files
    docs/        PROVENANCE.md: number -> file table

## License

**License: TBD by owner.** `LICENSE` currently holds the unmodified Apache-2.0 text as a placeholder so that the file
exists; it is not a decision. The final license and the copyright holder are for the owner to choose.
