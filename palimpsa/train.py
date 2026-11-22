"""Training loop: fp16 AMP + GradScaler, WSD schedule with decay branches, z-loss, clipping, atomic checkpoints.

One process per GPU (the minimal program runs independent seeds per T4; no DDP).

  python -m palimpsa.train --stage 10M --variant symA --steps 6600 --out runs/10M/symA_s1
  # WSD decay branch from a stable checkpoint (same out dir layout, new folder):
  python -m palimpsa.train ... --resume runs/10M/symA_s1/stable_step3145.pt --decay-start 3145 --steps 3700 --out ...

Batches depend only on (seed, step, micro_step), so resuming reproduces the data order exactly.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .eval import val_loss
from .loader import TrainStream, find_data_root
from .model import Config, PalimpsaLM, count_params, ladder_config, lm_loss


def build_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="10M")
    ap.add_argument("--variant", default="symA")
    ap.add_argument("--residual", default="prenorm")
    ap.add_argument("--attn-gate", default="none")
    ap.add_argument("--data", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=None, help="default: the checkpoint's seed when resuming, else 1")
    ap.add_argument("--steps", type=int, required=True, help="total optimizer steps of this run")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--global-batch-tokens", type=int, default=65_536)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=0, help="0 -> max(100, 1%% of steps)")
    ap.add_argument("--decay-frac", type=float, default=0.15)
    ap.add_argument("--decay-start", type=int, default=-1, help="-1 -> steps * (1 - decay_frac)")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--z-loss", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-windows", type=int, default=200, help="per domain; 0 = full validation")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-minutes", type=float, default=30.0)
    ap.add_argument("--keep-steps", default="", help="comma list: stable checkpoints kept for decay branches")
    ap.add_argument("--max-minutes", type=float, default=0.0, help="stop cleanly (with checkpoint) after this")
    ap.add_argument("--stop-at", type=int, default=0, help="stop cleanly (with checkpoint) at this step")
    ap.add_argument("--stop-file", default="", help="stop cleanly (with checkpoint) once this file exists")
    ap.add_argument("--resume", default="", help="checkpoint path, or 'auto' = <out>/ckpt.pt if present")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args(argv)


def lr_at(step: int, args) -> float:
    warmup = args.warmup or max(100, args.steps // 100)
    decay_start = args.decay_start if args.decay_start >= 0 else int(args.steps * (1 - args.decay_frac))
    if step < warmup:
        return args.lr * (step + 1) / warmup
    if step < decay_start:
        return args.lr
    return args.lr * max(0.0, (args.steps - step) / max(1, args.steps - decay_start))


def param_groups(model: PalimpsaLM, weight_decay: float):
    decay, keep = [], []
    for name, p in model.named_parameters():
        (decay if p.dim() >= 2 and not name.endswith(("b_free", "gamma")) else keep).append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": keep, "weight_decay": 0.0}]


def save_atomic(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main(argv=None) -> dict:
    args = build_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "cuda":  # read by the caching allocator at its first use, which comes later in this function
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    amp = device == "cuda"
    root = find_data_root(args.data)
    resume_path = out / "ckpt.pt" if args.resume == "auto" else (Path(args.resume) if args.resume else None)
    ckpt = torch.load(resume_path, map_location="cpu", weights_only=False) if resume_path and resume_path.exists() else None

    cfg = Config(**ckpt["config"]) if ckpt else ladder_config(args.stage, args.variant, residual=args.residual,
                                                             attn_gate=args.attn_gate)
    if args.seed is None:  # a resumed run or decay branch keeps the data order of its parent
        args.seed = ckpt["args"]["seed"] if ckpt else 1
    torch.manual_seed(args.seed)
    model = PalimpsaLM(cfg).to(device)
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step = 0
    if ckpt:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        scaler.load_state_dict(ckpt["scaler"])
        step = ckpt["step"]
    data = TrainStream(root, args.seq_len, seed=args.seed)
    tokens_per_micro = args.micro_batch * args.seq_len
    if args.global_batch_tokens % tokens_per_micro:
        raise SystemExit("global batch must be a multiple of micro_batch * seq_len")
    accum = args.global_batch_tokens // tokens_per_micro
    keep_steps = {int(s) for s in args.keep_steps.split(",") if s}
    total_params = count_params(model)[0]
    log = (out / "metrics.jsonl").open("a", encoding="utf-8")

    def record(row: dict) -> None:
        log.write(json.dumps(row) + "\n")
        log.flush()

    def checkpoint(path: Path) -> None:
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                     "step": step, "config": asdict(cfg), "args": vars(args)}, path)

    record({"event": "start", "step": step, "config": asdict(cfg), "params": total_params, "accum": accum,
            "data_root": str(root), "resumed_from": str(resume_path) if ckpt else None,
            "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")})
    start, last_ckpt, skipped, window_start, window_tokens = time.time(), time.time(), 0, time.time(), 0
    reason = "finished"
    model.train()
    try:
        while step < args.steps:
            lr = lr_at(step, args)
            for group in opt.param_groups:
                group["lr"] = lr
            loss_sum = 0.0
            for micro in range(accum):
                x, y, doc, _ = data.batch(step, micro, args.micro_batch)
                x, y, doc = x.to(device, non_blocking=True), y.to(device, non_blocking=True), doc.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                    logits = model(x, doc)
                loss = lm_loss(logits, y, z_loss=args.z_loss) / accum
                scaler.scale(loss).backward()
                loss_sum += loss.item()
            scaler.unscale_(opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip).item()
            scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            skipped += int(scaler.get_scale() < scale)
            step += 1
            window_tokens += accum * tokens_per_micro
            if not math.isfinite(loss_sum):
                reason = "non-finite loss"
                break
            if step % args.log_every == 0 or step == args.steps:
                now = time.time()
                row = {"step": step, "loss": round(loss_sum, 5), "lr": lr, "grad_norm": round(grad_norm, 4),
                       "scale": scaler.get_scale(), "skipped": skipped,
                       "tok_s": round(window_tokens / max(now - window_start, 1e-9)),
                       "elapsed_min": round((now - start) / 60, 2)}
                if device == "cuda":
                    row["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                record(row)
                print(json.dumps(row), flush=True)
                window_start, window_tokens = now, 0
            if step % args.eval_every == 0 or step == args.steps:
                ev = val_loss(model, root, args.seq_len, batch=args.micro_batch,
                              max_windows=args.eval_windows or None, device=device, amp=amp)
                record({"step": step, "eval": ev})
                print(json.dumps({"step": step, "eval": ev}), flush=True)
            if step in keep_steps:
                checkpoint(out / f"stable_step{step}.pt")
            if time.time() - last_ckpt > args.ckpt_minutes * 60:
                checkpoint(out / "ckpt.pt")
                last_ckpt = time.time()
            if args.stop_at and step >= args.stop_at:
                reason = f"stop_at {args.stop_at}"
                break
            if args.max_minutes and time.time() - start > args.max_minutes * 60:
                reason = "time limit"
                break
            if args.stop_file and os.path.exists(args.stop_file):
                reason = "stop file"
                break
    except Exception as exc:  # e.g. CUDA OOM: keep the state of the last completed step, then fail
        record({"event": "error", "step": step, "error": repr(exc)[:500]})
        try:
            checkpoint(out / "ckpt_error.pt")  # the periodic ckpt.pt stays untouched
        finally:
            log.close()
        raise
    checkpoint(out / "ckpt.pt")
    summary = {"event": "end", "step": step, "reason": reason, "skipped": skipped,
               "minutes": round((time.time() - start) / 60, 2)}
    record(summary)
    log.close()
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    main()
