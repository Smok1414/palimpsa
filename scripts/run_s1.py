"""Stage S1: synthetic in-context memory suite and gate G1 (section 7.3).

G1 passes when all of the following hold:
  a) on mqar3 there is an eval step where SymA >= 95% while Base < 80% (exact multi-token addressing
     makes SymA learn the task clearly faster);
  b) trap:   final SymA accuracy >= final Base accuracy - 5 pp (the match bias does not trap the model);
  c) fuzzy3: final SymA accuracy >= final Base accuracy - 5 pp (no harm when exact matching misleads);
  d) leakage canary: honest Base/SymA >= ln(64) - 0.01 and the leaky positive control < ln(64) - 0.5.

Usage: python scripts/run_s1.py [--smoke] [--steps 3000] [--tasks mqar3,trap] [--variants base,symA,fox]
Results are written incrementally to runs/s1/results.json; reruns skip finished (task, variant) pairs.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from palimpsa import synth  # noqa: E402
from palimpsa.canary import LN_ALPHABET, leak_canary  # noqa: E402
from palimpsa.model import Config, PalimpsaLM, count_params, lm_loss  # noqa: E402


QK_NORM = True


def model_config(variant: str) -> Config:
    return Config(vocab=synth.VOCAB, d_model=128, n_layers=4, n_heads=2, n_kv_heads=2, head_dim=64,
                  d_ff=384, variant=variant, qk_norm=QK_NORM)


def lr_at(step: int, steps: int, peak: float, warmup: int) -> float:
    """WSD: linear warmup, constant, linear decay to zero over the last 20% of steps."""
    if step < warmup:
        return peak * (step + 1) / warmup
    decay_start = int(0.8 * steps)
    if step < decay_start:
        return peak
    return peak * (steps - step) / (steps - decay_start)


def eval_set(task: str, n_seq: int, batch: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    rng = random.Random(90_000 + list(synth.TASKS).index(task))  # fixed, shared by all variants
    return [synth.TASKS[task](batch, rng) for _ in range(n_seq // batch)]


@torch.no_grad()
def accuracy(model: PalimpsaLM, batches, device: str, amp: bool) -> float:
    model.eval()
    hit = total = 0
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            pred = model(x).argmax(-1)
        scored = y != synth.IGNORE
        hit += (pred[scored] == y[scored]).sum().item()
        total += scored.sum().item()
    model.train()
    return hit / total


def train_run(task: str, variant: str, args, device: str, batches) -> dict:
    torch.manual_seed(args.seed)
    model = PalimpsaLM(model_config(variant)).to(device)
    decay = [p for n, p in model.named_parameters() if p.dim() == 2 and not n.endswith("b_free")]
    keep = [p for n, p in model.named_parameters() if not (p.dim() == 2 and not n.endswith("b_free"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": keep, "weight_decay": 0.0}],
                            lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rng = random.Random(args.seed * 1000 + 17)  # identical training stream for every variant
    curve, skipped, start = [], 0, time.time()
    for step in range(args.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, args.steps, args.lr, args.warmup)
        x, y = synth.TASKS[task](args.batch, rng)
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            logits = model(x)
        loss = lm_loss(logits, y, z_loss=1e-4, ignore_index=synth.IGNORE)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped += int(scaler.get_scale() < scale)
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            acc = accuracy(model, batches, device, amp)
            curve.append({"step": step + 1, "acc": acc, "loss": round(loss.item(), 5)})
            print(f"  {task:6s} {variant:5s} step {step + 1:5d}  loss {loss.item():.4f}  acc {acc:.3f}"
                  f"  {time.time() - start:6.1f}s", flush=True)
    return {"curve": curve, "final_acc": curve[-1]["acc"], "seconds": round(time.time() - start, 1),
            "skipped_steps": skipped, "params": count_params(model)[0]}


def gate_g1(results: dict, leak: dict) -> dict:
    out = {}
    s = results.get("mqar3", {}).get("symA")
    b = results.get("mqar3", {}).get("base")
    if s and b:
        steps = [p["step"] for p, q in zip(s["curve"], b["curve"]) if p["acc"] >= 0.95 and q["acc"] < 0.80]
        out["a_mqar3_symA_faster"] = {"pass": bool(steps), "steps": steps}
    for task, key in (("trap", "b_trap_no_harm"), ("fuzzy3", "c_fuzzy_no_harm")):
        s = results.get(task, {}).get("symA")
        b = results.get(task, {}).get("base")
        if s and b:
            out[key] = {"pass": s["final_acc"] >= b["final_acc"] - 0.05,
                        "symA": s["final_acc"], "base": b["final_acc"]}
    if leak:
        ok = (leak["base"] >= LN_ALPHABET - 0.01 and leak["symA"] >= LN_ALPHABET - 0.01
              and leak["symA_leaky"] < LN_ALPHABET - 0.5)
        out["d_leak_canary"] = {"pass": ok, **leak}
    base_sanity = results.get("mqar1", {}).get("base")
    if base_sanity:
        # b) and c) compare against Base; they only mean "no harm" if Base itself learned recall.
        out["validity_base_learns_mqar1"] = {"informative": base_sanity["final_acc"] >= 0.5,
                                             "base_mqar1": base_sanity["final_acc"]}
    complete = all(k in out for k in ("a_mqar3_symA_faster", "b_trap_no_harm", "c_fuzzy_no_harm",
                                      "d_leak_canary"))
    passed = complete and all(v["pass"] for v in out.values() if isinstance(v, dict) and "pass" in v)
    out["G1"] = "PASS" if passed else ("FAIL" if complete else "INCOMPLETE")
    if passed and base_sanity and not out["validity_base_learns_mqar1"]["informative"]:
        out["G1"] = "PASS (b, c uninformative: Base did not learn mqar1)"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(synth.TASKS))
    ap.add_argument("--variants", default="base,symA,fox")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-seqs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/s1")
    ap.add_argument("--smoke", action="store_true", help="30 steps per run, no files written")
    ap.add_argument("--no-qk-norm", action="store_true", help="diagnostics only")
    args = ap.parse_args()
    global QK_NORM
    QK_NORM = not args.no_qk_norm
    if args.smoke:
        args.steps, args.eval_every, args.eval_seqs, args.warmup = 30, 15, 128, 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    path = out_dir / "results.json"
    state = {"config": vars(args), "runs": {}, "leak": {}}
    if path.exists() and not args.smoke:
        state = json.loads(path.read_text(encoding="utf-8"))
        if {k: state["config"].get(k) for k in ("steps", "batch", "lr", "seed")} != \
                {k: getattr(args, k) for k in ("steps", "batch", "lr", "seed")}:
            raise SystemExit(f"{path} was produced with different settings; use another --out")

    def save():
        if args.smoke:
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        state["gate"] = gate_g1(state["runs"], state["leak"])
        path.write_text(json.dumps(state, indent=1), encoding="utf-8")

    print(f"device {device} | torch {torch.__version__}", flush=True)
    for task in args.tasks.split(","):
        batches = eval_set(task, args.eval_seqs, 128)
        for variant in args.variants.split(","):
            if variant in state["runs"].get(task, {}):
                print(f"skip {task}/{variant}: done", flush=True)
                continue
            state["runs"].setdefault(task, {})[variant] = train_run(task, variant, args, device, batches)
            save()
    if not state["leak"] and not args.smoke:
        state["leak"] = {"base": leak_canary("base"), "symA": leak_canary("symA"),
                         "symA_leaky": leak_canary("symA", leaky=True)}
        save()
    print("\nfinal accuracy")
    for task, by_variant in state["runs"].items():
        print(f"  {task:6s} " + "  ".join(f"{v}={r['final_acc']:.3f}" for v, r in by_variant.items()))
    gate = gate_g1(state["runs"], state["leak"])
    print("\n" + json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
