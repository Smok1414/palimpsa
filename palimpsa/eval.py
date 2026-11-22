"""Evaluation: per-domain validation loss, loss by match-length stratum and by position, BLiMP, LAMBADA,
degeneration (repetition) check of sampled continuations.

Usage: python -m palimpsa.eval --ckpt runs/x/ckpt.pt [--data auto] [--max-windows 200] [--breakdown]
                               [--blimp data/raw/eval/...] [--lambada data/raw/eval/en/test/en.parquet]
       python -m palimpsa.eval --ckpt runs/x/ckpt.pt --no-val --repetition [--rep-penalty 1.2] [--out-json f]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .loader import DOMAINS, EOS_ID, doc_ids, find_data_root, token_path, val_windows
from .model import Config, PalimpsaLM
from .sampling import distinct_n, generate, repeated_tail
from .symbolic import BUCKET_OF_LEN, N_BUCKETS, match_len

POSITION_BIN = 64


def _autocast(device: str, amp: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device == "cuda")


@torch.no_grad()
def val_loss(model: PalimpsaLM, root, seq_len: int = 1024, batch: int = 16, max_windows: int | None = None,
             device: str = "cuda", amp: bool = True, breakdown: bool = False) -> dict:
    """Mean next-token loss (nats/token) per domain and their unweighted mean ("macro").

    breakdown=True adds the loss per bucket of L_max (the longest suffix match available to the query) and per
    position bin, which tell where a difference between models comes from.
    """
    was_training = model.training
    model.eval()
    lut = torch.tensor(BUCKET_OF_LEN, device=device)
    out: dict = {}
    for domain in DOMAINS:
        x_all, y_all = val_windows(root, domain, seq_len, max_windows)
        total, count = 0.0, 0
        b_sum = torch.zeros(N_BUCKETS, dtype=torch.float64, device=device)
        b_cnt = torch.zeros(N_BUCKETS, dtype=torch.float64, device=device)
        p_sum = torch.zeros(seq_len // POSITION_BIN, dtype=torch.float64, device=device)
        for i in range(0, len(x_all), batch):
            x_np = x_all[i:i + batch]
            x = torch.from_numpy(x_np).to(device)
            y = torch.from_numpy(y_all[i:i + batch]).to(device)
            doc = torch.from_numpy(doc_ids(x_np)).to(device)
            with _autocast(device, amp):
                logits = model(x, doc)
            nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                                  reduction="none").view_as(y).double()
            total += nll.sum().item()
            count += nll.numel()
            if breakdown:
                bucket = lut[match_len(x, doc).max(-1).values.long()]
                b_sum.index_add_(0, bucket.reshape(-1), nll.reshape(-1))
                b_cnt.index_add_(0, bucket.reshape(-1), torch.ones_like(nll).reshape(-1))
                p_sum += nll[:, : (seq_len // POSITION_BIN) * POSITION_BIN].reshape(
                    nll.shape[0], -1, POSITION_BIN).sum((0, 2))
        out[domain] = total / max(count, 1)
        if breakdown:
            out[f"{domain}_by_lmax_bucket"] = {"loss": (b_sum / b_cnt.clamp_min(1)).tolist(),
                                               "share": (b_cnt / b_cnt.sum()).tolist()}
            out[f"{domain}_by_position"] = (p_sum / (len(x_all) * POSITION_BIN)).tolist()
    out["macro"] = float(np.mean([out[d] for d in DOMAINS]))
    model.train(was_training)
    return out


def _encode(tok, text: str) -> list[int]:
    return tok.encode(text).ids


@torch.no_grad()
def _sequence_logprobs(model, seqs: list[list[int]], device: str, amp: bool, score_from: list[int]) -> list[float]:
    """Sum of log p over tokens seqs[i][score_from[i]:], each sequence scored causally from its start."""
    width = max(len(s) for s in seqs)
    x = torch.full((len(seqs), width), EOS_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s)
    x = x.to(device)
    with _autocast(device, amp):
        logp = model(x).float().log_softmax(-1)
    scores = []
    for i, s in enumerate(seqs):
        tgt = torch.tensor(s[score_from[i]:], device=device)
        pos = torch.arange(score_from[i] - 1, len(s) - 1, device=device)
        scores.append(logp[i, pos, tgt].sum().item())
    return scores


@torch.no_grad()
def blimp(model, tok, blimp_dir, device: str = "cuda", amp: bool = True, batch: int = 64,
          max_tasks: int = 0, max_per_task: int = 0) -> dict:
    """BLiMP accuracy: log p(good) > log p(bad), macro-averaged over the 67 phenomena.

    blimp_dir holds either flat `blimp_<task>.parquet` files (Kaggle dataset) or `<task>/*.parquet` (HF layout).
    """
    import pyarrow.parquet as pq

    model.eval()
    per_task = {}
    base = Path(blimp_dir)
    paths = sorted(base.glob("blimp_*.parquet")) or sorted(base.glob("*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no BLiMP parquet files under {base}")
    for path in paths[:max_tasks] if max_tasks else paths:
        task = path.stem[len("blimp_"):] if path.name.startswith("blimp_") else path.parent.name
        rows = pq.read_table(path, columns=["sentence_good", "sentence_bad"]).to_pylist()
        rows = rows[:max_per_task] if max_per_task else rows
        good = [[EOS_ID] + _encode(tok, r["sentence_good"]) for r in rows]
        bad = [[EOS_ID] + _encode(tok, r["sentence_bad"]) for r in rows]
        hits = 0
        for i in range(0, len(rows), batch):
            g = _sequence_logprobs(model, good[i:i + batch], device, amp, [1] * len(good[i:i + batch]))
            b = _sequence_logprobs(model, bad[i:i + batch], device, amp, [1] * len(bad[i:i + batch]))
            hits += sum(gi > bi for gi, bi in zip(g, b))
        per_task[task] = hits / len(rows)
    return {"blimp_macro": float(np.mean(list(per_task.values()))), "blimp_tasks": per_task}


@torch.no_grad()
def lambada(model, tok, path, device: str = "cuda", amp: bool = True, batch: int = 32, max_items: int = 0) -> dict:
    """LAMBADA (OpenAI version): greedy accuracy on the whole last word and its mean log-likelihood."""
    import pyarrow.parquet as pq

    model.eval()
    texts = pq.read_table(path, columns=["text"]).column(0).to_pylist()
    texts = texts[:max_items] if max_items else texts
    hits, logp_total = 0, 0.0
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        ctx = [[EOS_ID] + _encode(tok, t.rsplit(" ", 1)[0]) for t in chunk]
        tgt = [_encode(tok, " " + t.rsplit(" ", 1)[1]) for t in chunk]
        seqs = [c + g for c, g in zip(ctx, tgt)]
        width = max(len(s) for s in seqs)
        x = torch.full((len(seqs), width), EOS_ID, dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, : len(s)] = torch.tensor(s)
        x = x.to(device)
        with _autocast(device, amp):
            logp = model(x).float().log_softmax(-1)
        for j, (c, g) in enumerate(zip(ctx, tgt)):
            pos = torch.arange(len(c) - 1, len(c) + len(g) - 1, device=device)
            target = torch.tensor(g, device=device)
            hits += int(torch.equal(logp[j, pos].argmax(-1), target))
            logp_total += logp[j, pos, target].sum().item()
    return {"lambada_acc": hits / len(texts), "lambada_mean_logp": logp_total / len(texts)}


def doc_starts(root, domain: str, prompt_len: int, cont_len: int, n: int) -> list[list[int]]:
    """n validation documents (evenly spaced over those with >= prompt_len + cont_len tokens), as token lists."""
    stream = np.fromfile(token_path(root, domain, "val.bin"), dtype=np.uint16).astype(np.int64)
    offsets = np.load(token_path(root, domain, "val_offsets.npy")).tolist() + [len(stream)]
    docs = []
    for a, b in zip(offsets[:-1], offsets[1:]):
        d = stream[a:b].tolist()
        while d and d[-1] == EOS_ID:
            d.pop()
        if len(d) >= prompt_len + cont_len:
            docs.append(d)
    pick = np.unique(np.linspace(0, len(docs) - 1, min(n, len(docs))).round().astype(int))
    return [docs[i] for i in pick]


def _rep_stats(conts: list[list[int]], full: int, d4_max: float, tail_min: float) -> dict:
    d4 = [distinct_n(c, 4) for c in conts]
    tail = [repeated_tail(c) for c in conts]
    loop = [t >= tail_min for t in tail]
    low = [v is not None and v < d4_max for v in d4]
    return {"n": len(conts), "distinct4_mean": float(np.mean([v for v in d4 if v is not None] or [float("nan")])),
            "tail_share_mean": float(np.mean(tail)), "share_tail_loop": float(np.mean(loop)),
            "share_low_distinct4": float(np.mean(low)), "share_repeats": float(np.mean([a or b for a, b in zip(loop, low)])),
            "share_eos_early": float(np.mean([len(c) < full for c in conts])),
            "mean_len": float(np.mean([len(c) for c in conts]))}


@torch.no_grad()
def repetition_check(model, tok, root, device: str = "cuda", domains=("web", "code"), per_domain: int = 50,
                     prompt_len: int = 32, new_tokens: int = 120, temperatures=(0.3, 0.7),
                     repetition_penalty: float = 1.0, batch: int = 25, seed: int = 0,
                     d4_max: float = 0.5, tail_min: float = 0.25, examples: int = 2) -> dict:
    """Degeneration check: continue `per_domain` validation document starts (first prompt_len tokens) by new_tokens
    tokens at each temperature (pure temperature sampling, no top-k/top-p) and measure repetition per continuation:
    distinct-4 (unique 4-grams / all 4-grams) and the share of the continuation covered by a periodic tail repeated
    at least twice. A text counts as repeating if its tail share >= tail_min or its distinct-4 < d4_max. The real
    continuations of the same documents give the reference level (code repeats more than prose by nature)."""
    model.eval()
    out: dict = {"settings": {"domains": list(domains), "per_domain": per_domain, "prompt_len": prompt_len,
                              "new_tokens": new_tokens, "temperatures": list(temperatures), "top_k": 0, "top_p": 1.0,
                              "repetition_penalty": repetition_penalty, "seed": seed,
                              "repeating_if": f"tail_share >= {tail_min} or distinct4 < {d4_max}"}}
    starts = {d: doc_starts(root, d, prompt_len, new_tokens, per_domain) for d in domains}
    out["reference"] = {d: _rep_stats([s[prompt_len:prompt_len + new_tokens] for s in starts[d]], new_tokens,
                                      d4_max, tail_min) for d in domains}
    for temp in temperatures:
        gen = torch.Generator().manual_seed(seed)
        res, worst = {}, {}
        for d in domains:
            prompts = [s[:prompt_len] for s in starts[d]]
            conts, t0 = [], time.time()
            for i in range(0, len(prompts), batch):
                conts += generate(model, prompts[i:i + batch], new_tokens, temperature=temp,
                                  repetition_penalty=repetition_penalty, generator=gen)
            seconds = time.time() - t0  # generate() returns lists, so the GPU work is done
            res[d] = _rep_stats(conts, new_tokens, d4_max, tail_min)
            n_tok = sum(len(c) for c in conts)
            res[d].update({"gen_tokens": n_tok, "gen_seconds": round(seconds, 2),
                           "gen_tok_s": round(n_tok / max(seconds, 1e-9), 1)})
            print(f"T={temp} {d}: {n_tok} tokens in {seconds:.1f} s = {res[d]['gen_tok_s']:.0f} tok/s "
                  f"(batch {batch}, no KV cache)", flush=True)
            order = sorted(range(len(conts)), key=lambda k: -repeated_tail(conts[k]))[:examples]
            worst[d] = [{"tail_share": round(repeated_tail(conts[k]), 3), "distinct4": distinct_n(conts[k], 4),
                         "prompt": tok.decode(prompts[k])[-80:], "continuation": tok.decode(conts[k])[:300]}
                        for k in order]
        res["all"] = {k: float(np.mean([res[d][k] for d in domains])) for k in res[domains[0]]}
        out[f"T={temp}"] = res
        out[f"T={temp}_worst"] = worst
    return out


def load_checkpoint(path, device: str) -> PalimpsaLM:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)  # tensors and plain values only
    model = PalimpsaLM(Config(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device)


def main() -> None:
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="auto")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--breakdown", action="store_true")
    ap.add_argument("--blimp", default="")
    ap.add_argument("--lambada", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--blimp-max-per-task", type=int, default=0)
    ap.add_argument("--lambada-max", type=int, default=0)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--no-val", action="store_true", help="skip the validation loss")
    ap.add_argument("--repetition", action="store_true", help="degeneration check (see repetition_check)")
    ap.add_argument("--rep-per-domain", type=int, default=50)
    ap.add_argument("--rep-domains", default="web,code")
    ap.add_argument("--rep-temps", default="0.3,0.7")
    ap.add_argument("--rep-new", type=int, default=120)
    ap.add_argument("--rep-penalty", type=float, default=1.0)
    ap.add_argument("--rep-batch", type=int, default=25)
    args = ap.parse_args()
    root = find_data_root(args.data)
    model = load_checkpoint(args.ckpt, args.device)
    res = {} if args.no_val else val_loss(model, root, args.seq_len, max_windows=args.max_windows or None,
                                          device=args.device, breakdown=args.breakdown)
    if args.blimp or args.lambada or args.repetition:
        tok_path = root / "tokenizer.json" if (root / "tokenizer.json").exists() else root.parent / "tokenizer" / "tokenizer.json"
        tok = Tokenizer.from_file(str(tok_path))
        tok.encode_special_tokens = True
        if args.blimp:
            res.update(blimp(model, tok, args.blimp, args.device, max_per_task=args.blimp_max_per_task))
        if args.lambada:
            res.update(lambada(model, tok, args.lambada, args.device, max_items=args.lambada_max))
        if args.repetition:
            res["repetition"] = repetition_check(
                model, tok, root, args.device, domains=tuple(args.rep_domains.split(",")),
                per_domain=args.rep_per_domain, new_tokens=args.rep_new,
                temperatures=tuple(float(t) for t in args.rep_temps.split(",")),
                repetition_penalty=args.rep_penalty, batch=args.rep_batch)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
