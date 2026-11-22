"""Stage S2 session runner for committed Kaggle notebook versions ("Save & Run All").

Guarantees: the process always exits with code 0; <work>/status.json is rewritten at every state change;
<work>/summary_<session>.json is written at the end; train.py writes checkpoints every 30 min, at the end of every
run (also on a clean stop at the session deadline) and on failure (ckpt_error.pt).

Chain between versions: every version writes <work>/palimpsa_chain.json (sequence number, session, file list with
sizes, sha256 of small text files and of every checkpoint). The previous version's output is attached as input; its
marker is found under <input>, the files it lists are verified, and its files are carried over except the code and
checkpoints; the checkpoints in CARRY_PT are carried too, each copy verified by sha256 before it is renamed.

Sessions:
  probe  CPU only: chain check, carry (incl. CARRY_PT checkpoints, loaded on CPU), a tiny file.
  s2-1   dataset sha256, CUDA tests, 30M speed probe (Base on GPU0 || SymA on GPU1), LR sweep (Base on GPU0,
         SymA on GPU1), LR choice by the approved rule.
  s2-2   dataset sha256, CUDA tests, B_long_a -> B_d435 on GPU0 || S_long_a -> S_d435 on GPU1 with the LRs given
         by --lr-base / --lr-symA, then gate G2a (SymA - Base at D/N = 20).
  s2-3   dataset sha256, CUDA tests, A1_fox -> A3_ell_rope on GPU0 || A2_symA_rope -> S_long_b on GPU1 (S_long_b
         resumes the carried S_long_a/ckpt.pt and stops cleanly once GPU0 is done), then gate G2b.
  s2-3a  split S2-3 (chosen 2026-11-22): A1_fox on GPU0 || A3_ell_rope on GPU1, then gate G2b.
  s2-3b  A2_symA_rope on GPU0 || S_long_b on GPU1 (stops once A2 is done); G2b again with A2 - SymA.

Usage: python -u scripts/kaggle_s2.py --session s2-1 --max-hours 4 [--work /kaggle/working] [--dry-run]
       python -u scripts/kaggle_s2.py --session s2-2 --max-hours 5.2 --lr-base 3e-3 --lr-symA 3e-3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from plan_s2 import WARMUP, plan  # noqa: E402

CHAIN = "palimpsa_chain.json"
SPEED30_STEPS = 20
# Checkpoints that later sessions resume from; each version copies them forward (verified) when its input has them:
# B_long_a/S_long_a (S2-2) feed B_long_b in S2-4 and S_long_b in S2-3; S_long_b (S2-3) feeds S2-4.
CARRY_PT = ("s2/B_long_a/ckpt.pt", "s2/S_long_a/ckpt.pt", "s2/S_long_b/ckpt.pt", "s2/S_long_b/stable_step11283.pt")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 23), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()) + " UTC"


class Status:
    def __init__(self, work: Path, session: str):
        self.path, self.lock = work / "status.json", threading.Lock()
        self.data = {"session": session, "state": "running", "started": now(), "updated": now(), "events": [],
                     "runs": {}}
        self.save()

    def event(self, msg: str) -> None:
        print(f"[{now()}] {msg}", flush=True)
        with self.lock:
            self.data["events"].append(f"{now()} {msg}")
        self.save()

    def run(self, rid: str, **fields) -> None:
        with self.lock:
            self.data["runs"].setdefault(rid, {}).update(fields)
        self.save()

    def save(self) -> None:
        with self.lock:
            self.data["updated"] = now()
            write_json(self.path, self.data)


# ---------------------------------------------------------------- chain
def chain_in(input_root: Path, status: Status) -> tuple[dict, Path | None, dict]:
    markers = sorted(input_root.glob(f"**/{CHAIN}")) if input_root.exists() else []
    mounts = sorted(str(p.relative_to(input_root)) for p in input_root.glob("*/*/*")) if input_root.exists() else []
    info: dict = {"input_mounts": mounts[:20], "markers_found": [str(m) for m in markers]}
    if not markers:
        status.event("chain: no previous marker under the input")
        return info, None, {}
    best = max(markers, key=lambda m: json.loads(m.read_text(encoding="utf-8")).get("seq", 0))
    prev = json.loads(best.read_text(encoding="utf-8"))
    base = best.parent
    checks = {rel: (base / rel).exists() and sha256_of(base / rel) == want for rel, want in prev["checks"].items()}
    info.update({"previous_seq": prev["seq"], "previous_session": prev["session"], "previous_written": prev["time"],
                 "previous_dir": str(base), "previous_files": len(prev["files"]), "verified": checks,
                 "ok": all(checks.values())})
    status.event(f"chain: found seq {prev['seq']} ({prev['session']}) at {base}, verified={info['ok']}")
    return info, base, prev["checks"]


def copy_verified(src: Path, dst: Path, want: str | None) -> dict:
    """Stream-copy src to dst.tmp, hash both ends, rename only if the hashes agree (and match `want` if given)."""
    t0, digest, tmp = time.time(), hashlib.sha256(), dst.with_suffix(dst.suffix + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fi, tmp.open("wb") as fo:
        for chunk in iter(lambda: fi.read(1 << 23), b""):
            digest.update(chunk)
            fo.write(chunk)
    src_hash, dst_hash = digest.hexdigest(), sha256_of(tmp)
    ok = src_hash == dst_hash and (want is None or want == src_hash)
    if ok:
        os.replace(tmp, dst)
    else:
        tmp.unlink()
    return {"bytes": src.stat().st_size, "sha256": src_hash, "marker_sha256": want, "ok": ok,
            "seconds": round(time.time() - t0, 1)}


def carry(prev: Path, work: Path, status: Status, keep_pt: tuple[str, ...] = CARRY_PT,
          prev_checks: dict | None = None) -> tuple[list[str], dict]:
    """Copy the previous version's files into this version's output: everything except the code, status, marker,
    Kaggle's own "__" files and checkpoints; checkpoints listed in `keep_pt` are copied with sha256 verification."""
    copied, pts = [], {}
    for src in prev.rglob("*"):
        rel = src.relative_to(prev)
        name = rel.as_posix()
        if (src.is_dir() or rel.parts[0] == "palimpsa_code" or src.suffix == ".tmp"
                or rel.name in (CHAIN, "status.json") or rel.name.startswith("__")):  # "__" = Kaggle's own files
            continue
        dst = work / rel
        if src.suffix == ".pt":
            if name in keep_pt and not dst.exists():
                pts[name] = copy_verified(src, dst, (prev_checks or {}).get(name))
                status.event(f"carry checkpoint {name}: {pts[name]}")
            continue
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(name)
    missing = [p for p in keep_pt if p not in pts and not (work / p).exists()]
    status.event(f"carry: {len(copied)} files, checkpoints {sorted(pts)}; not in the previous output: {missing}")
    return copied, pts


def chain_out(work: Path, seq: int, session: str) -> dict:
    files, checks = {}, {}
    for p in sorted(work.rglob("*")):
        rel = p.relative_to(work)
        if p.is_dir() or rel.parts[0] == "palimpsa_code" or rel.name == CHAIN:
            continue
        files[str(rel).replace("\\", "/")] = p.stat().st_size
        small_text = p.stat().st_size <= 1 << 20 and p.suffix in (".txt", ".json", ".jsonl")
        if (small_text and rel.name != "status.json") or p.suffix == ".pt":  # checkpoints: verified by the next hop
            checks[str(rel).replace("\\", "/")] = sha256_of(p)
    marker = {"seq": seq, "session": session, "time": now(), "files": files, "checks": checks}
    write_json(work / CHAIN, marker)
    return marker


# ---------------------------------------------------------------- runs
def read_rows(metrics: Path) -> list[dict]:
    if not metrics.exists():
        return []
    rows = []
    for line in metrics.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def spike_check(rows: list[dict], warmup: int, threshold: float = 0.3, window: int = 100) -> dict:
    """After warmup: largest excess of the step loss over the running minimum, and whether an excess above
    `threshold` lasted `window` steps or more (the agreed stop criterion)."""
    run_min, worst, worst_step, streak, unrecovered = math.inf, 0.0, None, None, False
    for r in rows:
        if "tok_s" not in r or r["step"] <= warmup:
            continue
        excess = r["loss"] - run_min if run_min < math.inf else 0.0
        if excess > worst:
            worst, worst_step = excess, r["step"]
        if excess > threshold:
            streak = streak if streak is not None else r["step"]
            unrecovered = unrecovered or r["step"] - streak >= window
        else:
            streak = None
        run_min = min(run_min, r["loss"])
    return {"max_excess_over_running_min": round(worst, 4), "at_step": worst_step, "unrecovered_spike": unrecovered}


def run_summary(run_dir: Path, warmup: int, steady_after: int = 0) -> dict:
    rows = read_rows(run_dir / "metrics.jsonl")
    train = [r for r in rows if "tok_s" in r]
    evals = [r for r in rows if "eval" in r]
    end = next((r for r in reversed(rows) if r.get("event") in ("end", "error")), None)
    steady = [r for r in train if r["step"] > steady_after] if steady_after else train[len(train) // 2:]
    tok = sorted(r["tok_s"] for r in steady)
    last_step = train[-1]["step"] if train else 0
    skipped = train[-1]["skipped"] if train else None
    return {"steps_done": last_step, "first_loss": train[0]["loss"] if train else None,
            "last_loss": train[-1]["loss"] if train else None,
            "tok_s_median": tok[len(tok) // 2] if tok else None,
            "peak_gib": max((r.get("peak_gib", 0) for r in train), default=None), "skipped": skipped,
            "skipped_frac": round(skipped / last_step, 4) if train and last_step else None,
            "final_eval": evals[-1]["eval"] if evals else None, "final_eval_step": evals[-1]["step"] if evals else None,
            "end": end, "spike": spike_check(rows, warmup)}


def choose_lr(results: dict[str, dict], steps: int) -> dict:
    """Approved rule: lowest final macro after decay; within 0.005 nat take the smaller LR; runs that are
    non-finite, unfinished or have > 1% skipped steps are excluded; an extra grid point is proposed only when the
    strict optimum is at an edge and remains the choice."""
    valid, excluded = {}, {}
    for lr, r in results.items():
        ev, frac = r.get("final_eval"), r.get("skipped_frac")
        macro = ev.get("macro") if ev else None
        if r.get("final_eval_step") != steps or macro is None or not math.isfinite(macro):
            excluded[lr] = "unfinished or non-finite"
        elif frac is None or frac > 0.01:
            excluded[lr] = f"skipped fraction {frac}"
        else:
            valid[lr] = macro
    if not valid:
        return {"choice": None, "valid": valid, "excluded": excluded, "reason": "no valid run"}
    best_lr = min(valid, key=valid.get)
    near = [lr for lr in valid if valid[lr] - valid[best_lr] <= 0.005]
    choice = min(near, key=float)
    grid = sorted(results, key=float)
    extra = None
    if choice == best_lr and choice == grid[-1]:
        extra = f"{float(choice) * 2:g}"
    elif choice == best_lr and choice == grid[0]:
        extra = f"{float(choice) / 2:g}"
    return {"choice": choice, "argmin": best_lr, "within_0.005": sorted(near, key=float), "valid": valid,
            "excluded": excluded, "extra_point_needed": extra}


def gate_g2a(base: dict, sym: dict, steps: int) -> dict:
    """Approved G2a (D/N = 20): STOP only if delta_macro >= 0 or any domain delta > +0.010, where delta is
    SymA - Base on the final validation after decay. Incomplete runs give no verdict."""
    for name, r in (("Base", base), ("SymA", sym)):
        ev = r.get("final_eval") or {}
        if r.get("final_eval_step") != steps or not all(
                isinstance(ev.get(k), (int, float)) and math.isfinite(ev[k]) for k in ("web", "code", "math", "macro")):
            return {"verdict": None, "reason": f"{name} has no finite final eval at step {steps}"}
    delta = {k: sym["final_eval"][k] - base["final_eval"][k] for k in ("macro", "web", "code", "math")}
    stop = [f"delta_macro {delta['macro']:+.4f} >= 0"] if delta["macro"] >= 0 else []
    stop += [f"{d} delta {delta[d]:+.4f} > +0.010" for d in ("web", "code", "math") if delta[d] > 0.010]
    return {"delta": delta, "base": base["final_eval"], "symA": sym["final_eval"],
            "verdict": "STOP" if stop else "continue to G2b", "stop_reasons": stop,
            "within_noise": -0.005 < delta["macro"] < 0}


def gate_g2b(base: dict, sym: dict, a1: dict, a3: dict, steps: int, a2: dict | None = None) -> dict:
    """Approved G2b (D/N = 20, macro): STOP if l helps in neither setting, i.e. (SymA - A1) > 0 and (A3 - Base) > 0.
    SymA - A1: effect of l without RoPE (with forgetting); A3 - Base: effect of l with RoPE. A2 - SymA (RoPE added
    to SymA) is reported for information only."""
    named = {"Base": base, "SymA": sym, "A1_fox": a1, "A3_ell_rope": a3}
    for name, r in named.items():
        ev = r.get("final_eval") or {}
        if r.get("final_eval_step") != steps or not all(
                isinstance(ev.get(k), (int, float)) and math.isfinite(ev[k]) for k in ("web", "code", "math", "macro")):
            return {"verdict": None, "reason": f"{name} has no finite final eval at step {steps}"}
    keys = ("macro", "web", "code", "math")
    d_l_nope = {k: sym["final_eval"][k] - a1["final_eval"][k] for k in keys}
    d_l_rope = {k: a3["final_eval"][k] - base["final_eval"][k] for k in keys}
    out = {"symA_minus_A1": d_l_nope, "A3_minus_base": d_l_rope,
           "evals": {name: r["final_eval"] for name, r in named.items()},
           "verdict": "STOP" if d_l_nope["macro"] > 0 and d_l_rope["macro"] > 0 else "continue",
           "l_helps": {"without_rope": d_l_nope["macro"] < 0, "with_rope": d_l_rope["macro"] < 0}}
    if a2 and a2.get("final_eval_step") == steps and a2.get("final_eval"):
        out["A2_minus_symA"] = {k: a2["final_eval"][k] - sym["final_eval"][k] for k in keys}
        out["evals"]["A2_symA_rope"] = a2["final_eval"]
    return out


class Runner:
    def __init__(self, args, status: Status, data_root: str):
        self.args, self.status, self.data_root = args, status, data_root
        self.deadline = time.time() + args.max_hours * 3600
        self.out = Path(args.work) / "s2"
        self.out.mkdir(parents=True, exist_ok=True)

    def shrink(self, argv: list[str]) -> list[str]:
        """Dry run: same structure, tiny 1M CPU runs."""
        a, small = list(argv), {"--stage": "1M", "--steps": "8", "--decay-start": "6", "--warmup": "2",
                               "--micro-batch": "2", "--global-batch-tokens": "256", "--eval-windows": "2"}
        for i, tok in enumerate(a[:-1]):
            if tok in small:
                a[i + 1] = small[tok]
            elif tok == "--eval-every" and int(a[i + 1]) > 4:
                a[i + 1] = "4"
            elif tok in ("--stop-at", "--keep-steps"):
                a[i + 1] = "4"
            elif tok == "--resume":
                a[i + 1] = re.sub(r"stable_step\d+", "stable_step4", a[i + 1])
        return a + ["--seq-len", "64", "--device", "cpu"]

    def train(self, rid: str, argv: list[str], gpu: int) -> int:
        left_min = (self.deadline - time.time()) / 60 - 8  # margin for the last eval and checkpoint
        if left_min < 2:
            self.status.run(rid, state="skipped: session time", gpu=gpu)
            return -1
        argv = self.shrink(argv) if self.args.dry_run else argv
        if "--resume" in argv and not Path(argv[argv.index("--resume") + 1]).exists():
            self.status.run(rid, state="skipped: resume checkpoint missing", gpu=gpu)
            self.status.event(f"{rid}: skipped, {argv[argv.index('--resume') + 1]} does not exist")
            return -1
        if self.args.inject_fail == rid:
            argv = argv + ["--variant", "no_such_variant"]
        log = self.out / f"{rid}.log"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        cmd = [sys.executable, "-u", "-m", "palimpsa.train", *argv, "--max-minutes", f"{left_min:.1f}"]
        self.status.run(rid, state="running", gpu=gpu, started=now(), log=str(log.relative_to(self.args.work)),
                        cmd=" ".join(cmd[3:]))
        self.status.event(f"start {rid} on GPU{gpu}")
        t0 = time.time()
        try:
            with log.open("w", encoding="utf-8") as fh:
                rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
        except Exception as exc:  # the runner itself must not fail
            rc = -2
            self.status.event(f"{rid}: launch error {exc!r}")
        state = "done" if rc == 0 else f"failed rc={rc}"
        self.status.run(rid, state=state, rc=rc, seconds=round(time.time() - t0, 1), ended=now())
        self.status.event(f"end {rid}: {state} in {time.time() - t0:.0f}s")
        return rc

    def queue(self, gpu: int, jobs: list[tuple[str, list[str], list[str] | None]]) -> None:
        for rid, argv, fallback in jobs:
            try:
                rc = self.train(rid, argv, gpu)
                log = self.out / f"{rid}.log"
                if rc not in (0, -1) and fallback and log.exists() and "out of memory" in log.read_text(
                        encoding="utf-8", errors="replace").lower():
                    self.status.event(f"{rid}: OOM, retrying with the fallback micro-batch")
                    self.train(rid + "_fallback", fallback, gpu)
            except Exception as exc:
                self.status.event(f"{rid}: runner error {exc!r}")

    def run_queues(self, queues: dict[int, list], stop_file: Path | None = None, primary: int = 0) -> None:
        """Run one queue per GPU. If `stop_file` is given, it is created once the `primary` queue is done, so a
        filler run on the other GPU (started with --stop-file) stops cleanly instead of burning quota alone."""
        threads = {g: threading.Thread(target=self.queue, args=(g, q), daemon=True) for g, q in queues.items()}
        for t in threads.values():
            t.start()
        last_beat = time.time()
        while any(t.is_alive() for t in threads.values()):
            time.sleep(5)
            if stop_file and primary in threads and not threads[primary].is_alive() and not stop_file.exists():
                stop_file.write_text(now(), encoding="utf-8")
                self.status.event(f"GPU{primary} queue done: wrote {stop_file.name}, filler runs stop cleanly")
            if time.time() - last_beat > 600:
                last_beat = time.time()
                running = {rid: r for rid, r in self.status.data["runs"].items() if r.get("state") == "running"}
                steps = {rid: run_summary(self.out / rid, WARMUP).get("steps_done") for rid in running}
                self.status.event(f"heartbeat: running {steps}")


# ---------------------------------------------------------------- sessions
def speed30_jobs(out: Path, data: str) -> dict[int, tuple]:
    def argv(variant, micro, rid):
        return ["--stage", "30M", "--variant", variant, "--steps", str(SPEED30_STEPS), "--micro-batch", str(micro),
                "--global-batch-tokens", "65536", "--warmup", "10", "--log-every", "2", "--eval-every", "1000000",
                "--eval-windows", "8", "--ckpt-minutes", "30", "--data", data, "--out", str(out / rid)]
    return {gpu: (f"speed30_{v}", argv(v, 16, f"speed30_{v}"), argv(v, 8, f"speed30_{v}_fallback"))
            for gpu, v in ((0, "base"), (1, "symA"))}


def preflight(runner: Runner, status: Status, summary: dict) -> int:
    """GPU names, dataset sha256 and the CUDA unit tests; raises (no training) if the data or the tests fail."""
    import torch

    args, out = runner.args, runner.out
    n_gpu = torch.cuda.device_count()
    summary["gpus"] = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
    status.event(f"GPUs: {summary['gpus']}")
    if not args.dry_run:
        root = Path(runner.data_root)
        sums = [ln.split() for ln in (root / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines() if ln]
        bad = [name for want, name in sums if not (root / name).exists() or sha256_of(root / name) != want]
        summary["sha256"] = {"files_checked": len(sums), "mismatched": bad}
        status.event(f"sha256: {len(sums)} files, mismatched {bad}")
        if bad:
            raise RuntimeError("dataset does not match SHA256SUMS.txt")
        t0 = time.time()
        with (out / "tests.log").open("w", encoding="utf-8") as fh:
            rc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "not slow", "-p", "no:cacheprovider"],
                                cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        summary["tests"] = {"returncode": rc, "seconds": round(time.time() - t0, 1), "log": "s2/tests.log"}
        status.event(f"tests: rc={rc}")
        if rc != 0:
            raise RuntimeError("unit tests failed on this machine; training not started")
    return n_gpu


def session_s2_1(runner: Runner, status: Status, summary: dict) -> None:
    args, out = runner.args, runner.out
    n_gpu = preflight(runner, status, summary)
    sweep = [r for r in plan() if r.rid.startswith("sweep_")]
    speed = speed30_jobs(out, runner.data_root)
    queues = {0: [speed[0]] + [(r.rid, r.argv(str(out), runner.data_root), None) for r in sweep if "_B_" in r.rid],
              1: [speed[1]] + [(r.rid, r.argv(str(out), runner.data_root), None) for r in sweep if "_S_" in r.rid]}
    if n_gpu < 2 and not args.dry_run:
        status.event("fewer than 2 GPUs: both queues run one after the other on GPU0")
        queues = {0: queues[0] + queues[1]}
    runner.run_queues(queues)
    runs = {}
    for rid in ["speed30_base", "speed30_base_fallback", "speed30_symA", "speed30_symA_fallback"] + [
            r.rid for r in sweep]:
        if (out / rid).exists():
            runs[rid] = run_summary(out / rid, 10 if rid.startswith("speed30") else WARMUP,
                                    steady_after=SPEED30_STEPS // 2 if rid.startswith("speed30") else 0)
            runs[rid].update({k: status.data["runs"].get(rid, {}).get(k) for k in ("state", "rc", "seconds", "gpu")})
    summary["runs"] = runs
    steps = 8 if args.dry_run else sweep[0].steps
    summary["lr_choice"] = {
        tag: choose_lr({r.lr: runs.get(r.rid, {}) for r in sweep if f"_{tag[0]}_" in r.rid}, steps)
        for tag in ("Base", "SymA")}
    status.event(f"LR choice: { {k: v['choice'] for k, v in summary['lr_choice'].items()} }")


def session_s2_2(runner: Runner, status: Status, summary: dict) -> None:
    args, out = runner.args, runner.out
    if not (args.lr_base and args.lr_symA):
        raise ValueError("s2-2 needs --lr-base and --lr-symA (the LRs approved after S2-1)")
    lrs = {"eta_B": args.lr_base, "eta_S": args.lr_symA}
    summary["lr"] = {"Base": args.lr_base, "SymA": args.lr_symA}
    prev = Path(args.work) / "summary_s2-1.json"  # carried from the S2-1 version
    if prev.exists():
        chosen = {k: v.get("choice") for k, v in json.loads(prev.read_text(encoding="utf-8"))["lr_choice"].items()}
        summary["lr_choice_s2_1"] = chosen
        summary["lr_matches_s2_1"] = chosen == summary["lr"]
        status.event(f"LR {summary['lr']}, S2-1 choice {chosen}, match={summary['lr_matches_s2_1']}")
    n_gpu = preflight(runner, status, summary)
    runs = {r.rid: r for r in plan()}
    for r in runs.values():
        r.lr = lrs.get(r.lr, r.lr)

    def jobs(ids):
        return [(rid, runs[rid].argv(str(out), runner.data_root), None) for rid in ids]

    queues = {0: jobs(["B_long_a", "B_d435"]), 1: jobs(["S_long_a", "S_d435"])}
    if n_gpu < 2 and not args.dry_run:
        status.event("fewer than 2 GPUs: both queues run one after the other on GPU0")
        queues = {0: queues[0] + queues[1]}
    runner.run_queues(queues)
    summary["runs"] = {}
    for rid in ("B_long_a", "B_d435", "S_long_a", "S_d435"):
        if (out / rid).exists():
            summary["runs"][rid] = run_summary(out / rid, WARMUP)
            summary["runs"][rid].update(
                {k: status.data["runs"].get(rid, {}).get(k) for k in ("state", "rc", "seconds", "gpu")})
    steps = 8 if args.dry_run else runs["B_d435"].steps
    summary["g2a"] = gate_g2a(summary["runs"].get("B_d435", {}), summary["runs"].get("S_d435", {}), steps)
    status.event(f"G2a: {summary['g2a'].get('verdict')} delta={summary['g2a'].get('delta')}")


def session_s2_3(runner: Runner, status: Status, summary: dict) -> None:
    """One session (plan): GPU0 A1_fox -> A3_ell_rope || GPU1 A2_symA_rope -> S_long_b (filler), then G2b."""
    _s2_3_part(runner, status, summary, ["A1_fox", "A3_ell_rope"], ["A2_symA_rope", "S_long_b"], filler="S_long_b")


def session_s2_3a(runner: Runner, status: Status, summary: dict) -> None:
    """Split S2-3, part a (user's choice 2026-11-22): A1_fox on GPU0 || A3_ell_rope on GPU1, then G2b."""
    _s2_3_part(runner, status, summary, ["A1_fox"], ["A3_ell_rope"])


def session_s2_3b(runner: Runner, status: Status, summary: dict) -> None:
    """Split S2-3, part b: A2_symA_rope on GPU0 || S_long_b on GPU1 (resumes the carried S_long_a/ckpt.pt and stops
    cleanly once A2 is done). G2b is recomputed with A1/A3 from the carried summary_s2-3a.json, plus A2 - SymA."""
    _s2_3_part(runner, status, summary, ["A2_symA_rope"], ["S_long_b"], filler="S_long_b")


def _s2_3_part(runner: Runner, status: Status, summary: dict, gpu0: list[str], gpu1: list[str],
               filler: str | None = None) -> None:
    """Runs gpu0 || gpu1 (the `filler` run gets --stop-file and stops once the GPU0 queue is done), then gate G2b
    from A1/A3 (this session or the carried summary_s2-3a.json) and the S2-2 decay branches (summary_s2-2.json)."""
    args, out = runner.args, runner.out
    if not (args.lr_base and args.lr_symA):
        raise ValueError("s2-3 needs --lr-base and --lr-symA")
    lrs = {"eta_B": args.lr_base, "eta_S": args.lr_symA}
    summary["lr"] = {"Base": args.lr_base, "SymA": args.lr_symA}
    prev = Path(args.work) / "summary_s2-2.json"
    s22 = json.loads(prev.read_text(encoding="utf-8")) if prev.exists() else {}
    summary["lr_matches_s2_2"] = s22.get("lr") == summary["lr"] if s22 else None
    n_gpu = preflight(runner, status, summary)
    runs = {r.rid: r for r in plan()}
    for r in runs.values():
        r.lr = lrs.get(r.lr, r.lr)
    stop = out / "STOP_FILLER"
    stop.unlink(missing_ok=True)

    def job(rid, extra=()):
        return (rid, runs[rid].argv(str(out), runner.data_root) + list(extra), None)

    queues = {0: [job(rid) for rid in gpu0],
              1: [job(rid, ("--stop-file", str(stop)) if rid == filler else ()) for rid in gpu1]}
    if n_gpu < 2 and not args.dry_run:
        status.event(f"fewer than 2 GPUs: one queue on GPU0, filler {filler} dropped")
        queues = {0: queues[0] + [j for j in queues[1] if j[0] != filler]}
    runner.run_queues(queues, stop_file=stop if filler and len(queues) > 1 else None, primary=0)
    summary["filler_stop_written"] = stop.read_text(encoding="utf-8") if stop.exists() else None
    stop.unlink(missing_ok=True)  # must not be carried into the next version
    summary["runs"] = {}
    for rid in gpu0 + gpu1:
        if (out / rid).exists():
            summary["runs"][rid] = run_summary(out / rid, WARMUP)
            summary["runs"][rid].update(
                {k: status.data["runs"].get(rid, {}).get(k) for k in ("state", "rc", "seconds", "gpu")})
    s_long_b = out / "S_long_b"
    summary["s_long_b_checkpoints"] = sorted(p.name for p in s_long_b.glob("*.pt")) if s_long_b.exists() else []
    steps = 8 if args.dry_run else runs["A1_fox"].steps
    r22 = s22.get("runs", {})
    part_a = Path(args.work) / "summary_s2-3a.json"  # carried from S2-3a when this is S2-3b
    prior = json.loads(part_a.read_text(encoding="utf-8")).get("runs", {}) if part_a.exists() else {}
    have = {**prior, **summary["runs"]}
    summary["g2b"] = gate_g2b(r22.get("B_d435", {}), r22.get("S_d435", {}), have.get("A1_fox", {}),
                              have.get("A3_ell_rope", {}), steps, have.get("A2_symA_rope"))
    status.event(f"G2b: {summary['g2b'].get('verdict')} SymA-A1={summary['g2b'].get('symA_minus_A1')} "
                 f"A3-Base={summary['g2b'].get('A3_minus_base')}")


def session_probe(runner: Runner, status: Status, summary: dict) -> None:
    probe = Path(runner.args.work) / "probe"
    probe.mkdir(exist_ok=True)
    text = f"palimpsa chain probe, written {now()}\n"
    (probe / "hello.txt").write_text(text, encoding="utf-8")
    summary["probe_file"] = {"path": "probe/hello.txt", "bytes": len(text.encode()), "content": text.strip()}
    import torch

    loaded = {}
    for rel in CARRY_PT:  # the carried checkpoints must load on CPU (proves the files are complete and usable)
        path = Path(runner.args.work) / rel
        if path.exists():
            ck = torch.load(path, map_location="cpu", weights_only=False)
            loaded[rel] = {"bytes": path.stat().st_size, "step": ck["step"], "variant": ck["config"].get("variant"),
                           "tensors": len(ck["model"]), "lr": ck["args"].get("lr"), "seed": ck["args"].get("seed")}
            status.event(f"loaded {rel}: {loaded[rel]}")
    summary["checkpoints_loaded"] = loaded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, choices=["probe", "s2-1", "s2-2", "s2-3", "s2-3a", "s2-3b"])
    ap.add_argument("--max-hours", type=float, required=True)
    ap.add_argument("--lr-base", default="", help="s2-2: Base LR chosen after S2-1")
    ap.add_argument("--lr-symA", default="", help="s2-2: SymA LR chosen after S2-1")
    ap.add_argument("--work", default="/kaggle/working")
    ap.add_argument("--input", default="/kaggle/input")
    ap.add_argument("--data", default="auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--inject-fail", default="", help="dry-run test: make this run fail")
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    status = Status(work, args.session)
    summary: dict = {"session": args.session, "started": now(), "dry_run": args.dry_run}
    try:
        info, prev, prev_checks = chain_in(Path(args.input), status)
        summary["chain_in"] = info
        if prev is not None:
            copied, pts = carry(prev, work, status, prev_checks=prev_checks)
            summary["carried"], summary["carried_pt"] = len(copied), pts
        from palimpsa.loader import find_data_root

        try:
            data_root = str(find_data_root(args.data))
            summary["data_root"] = data_root
        except Exception as exc:
            data_root = ""
            summary["data_root"] = f"not found: {exc!r}"
        status.event(f"data root: {summary['data_root']}")
        runner = Runner(args, status, data_root)
        sessions = {"probe": session_probe, "s2-1": session_s2_1, "s2-2": session_s2_2, "s2-3": session_s2_3,
                    "s2-3a": session_s2_3a, "s2-3b": session_s2_3b}
        sessions[args.session](runner, status, summary)
        status.data["state"] = "done"
    except BaseException as exc:  # never propagate: the version must finish and keep its output
        status.data["state"] = "failed"
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()[-3000:]
        status.event(f"session error: {exc!r}")
    finally:
        summary["ended"] = now()
        write_json(work / f"summary_{args.session}.json", summary)
        try:
            prev_seq = summary.get("chain_in", {}).get("previous_seq", 0) or 0
            marker = chain_out(work, prev_seq + 1, args.session)
            status.event(f"chain: wrote seq {marker['seq']} with {len(marker['files'])} files")
        except Exception as exc:
            status.event(f"chain: marker not written {exc!r}")
        status.save()
        print("== summary", flush=True)
        print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.exit(0)
