"""Stage S2 (10M) run plan: run table, session schedule on 2x T4, quota estimate, train.py arguments.

Speeds are the session-2 real-loop measurements on one T4 (micro-batch 16): Base 39,039 tok/s, SymA 32,559.
FoX (A1), SymA+RoPE (A2) and l+RoPE (A3) use the same SDPA + SymBias path and are assumed to run at the SymA
speed (not measured). Validation: 3 domains x 1953 windows x 1024 tokens (~2M tokens per domain); a forward-only
token is assumed to cost 1/3 of a training token. 30 s start-up per process, 0.1 h per session for start-up.

Usage: python scripts/plan_s2.py [--commands] [--lr-base X --lr-symA Y]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

BATCH, MICRO, WARMUP, DECAY = 65_536, 16, 200, 0.15
TPS = {"base": 39_039, "sym": 32_559}
EVAL_TOKENS = 3 * 1953 * 1024
STARTUP_S, SESSION_OVERHEAD_H = 30, 0.1
STEPS = {"0.1B": 1526, "0.435B": 6638, "0.87B": 13275, "1.3B": 19836}  # round(tokens / 65,536)


def decay_start(steps: int) -> int:
    return int(steps * (1 - DECAY))


@dataclass
class Run:
    rid: str
    variant: str
    lr: str  # "1.5e-3" etc., or "eta_B" / "eta_S" = the LR chosen by the sweep
    start: int  # first step (0, or the step of the checkpoint it resumes from)
    stop: int  # last step done in this run (stop-at or steps)
    steps: int  # --steps (sets the schedule)
    dstart: int  # decay start
    eval_every: int
    note: str = ""
    keep: str = ""
    resume: str = ""
    seed: int = 1

    @property
    def tokens(self) -> float:
        return (self.stop - self.start) * BATCH

    @property
    def speed(self) -> float:
        return TPS["base" if self.variant == "base" else "sym"]

    def n_evals(self) -> int:
        n = sum(1 for s in range(self.start + 1, self.stop + 1) if s % self.eval_every == 0)
        return n + (1 if self.stop == self.steps and self.steps % self.eval_every else 0)

    def hours(self) -> float:
        train = self.tokens / self.speed
        evals = self.n_evals() * EVAL_TOKENS / (3 * self.speed)
        return (train + evals + STARTUP_S) / 3600

    def argv(self, out_root: str = "runs/s2", data: str = "auto") -> list[str]:
        """train.py arguments; resume paths are relative to out_root's parent layout (runs/s2/<rid>/...)."""
        a = ["--stage", "10M", "--variant", self.variant, "--seed", str(self.seed), "--lr", self.lr,
             "--steps", str(self.steps), "--warmup", str(WARMUP), "--decay-start", str(self.dstart),
             "--micro-batch", str(MICRO), "--global-batch-tokens", str(BATCH), "--eval-every", str(self.eval_every),
             "--eval-windows", "1953", "--log-every", "1", "--ckpt-minutes", "30", "--data", data]
        if self.stop != self.steps:
            a += ["--stop-at", str(self.stop)]
        if self.keep:
            a += ["--keep-steps", self.keep]
        if self.resume:
            a += ["--resume", self.resume.replace("runs/s2", out_root, 1)]
        return a + ["--out", f"{out_root}/{self.rid}"]

    def args(self) -> str:
        return " ".join(self.argv())


def plan() -> list[Run]:
    s01, s435, s87, s13 = STEPS["0.1B"], STEPS["0.435B"], STEPS["0.87B"], STEPS["1.3B"]
    k1, k2 = decay_start(s435), decay_start(s87)  # stable checkpoints for the decay branches: 5642, 11283
    runs = []
    for v, tag in (("base", "B"), ("symA", "S")):
        for lr in ("1.5e-3", "3e-3", "6e-3"):
            runs.append(Run(f"sweep_{tag}_{lr}", v, lr, 0, s01, s01, decay_start(s01), 1000, "LR sweep"))
    for v, tag, lr in (("base", "B", "eta_B"), ("symA", "S", "eta_S")):
        runs += [
            Run(f"{tag}_long_a", v, lr, 0, k1, s13, decay_start(s13), 250, "long run to the 0.435B stable point",
                keep=f"{k1},{k2}"),
            Run(f"{tag}_d435", v, lr, k1, s435, s435, k1, 10**6, "decay branch, D/N=20",
                resume=f"runs/s2/{tag}_long_a/stable_step{k1}.pt"),
            Run(f"{tag}_long_b", v, lr, k1, s13, s13, decay_start(s13), 250, "long run continued to 1.3B (D/N=60)",
                keep=f"{k1},{k2}", resume=f"runs/s2/{tag}_long_a/ckpt.pt"),
            Run(f"{tag}_d870", v, lr, k2, s87, s87, k2, 10**6, "decay branch, D/N=40",
                resume=f"runs/s2/{tag}_long_b/stable_step{k2}.pt"),
        ]
    runs += [Run("A1_fox", "fox", "eta_S", 0, s435, s435, k1, 1000, "A1: FoX (SymA without l)"),
             Run("A2_symA_rope", "symA_rope", "eta_S", 0, s435, s435, k1, 1000, "A2: SymA + RoPE (never cut)"),
             Run("A3_ell_rope", "ell_rope", "eta_B", 0, s435, s435, k1, 1000, "A3: l + RoPE (Base + l)")]
    return runs


def sessions(r: dict[str, Run]) -> list[tuple[str, list[str], list[str], str]]:
    return [
        ("S2-1 sweep + 30M speed", ["speed30_base"] + [f"sweep_B_{x}" for x in ("1.5e-3", "3e-3", "6e-3")],
         ["speed30_symA"] + [f"sweep_S_{x}" for x in ("1.5e-3", "3e-3", "6e-3")], "LR choice"),
        ("S2-2 Base/SymA to D/N=20", ["B_long_a", "B_d435"], ["S_long_a", "S_d435"], "G2a"),
        ("S2-3 ablations", ["A1_fox", "A3_ell_rope"], ["A2_symA_rope", "S_long_b*"], "G2b"),
        ("S2-4 long runs to D/N=60", ["B_long_b"], ["S_long_b*", "S_d870", "B_d870"], "G2c -> G2"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--commands", action="store_true")
    ap.add_argument("--lr-base", default="eta_B")
    ap.add_argument("--lr-symA", default="eta_S")
    args = ap.parse_args()
    runs = plan()
    for run in runs:
        run.lr = {"eta_B": args.lr_base, "eta_S": args.lr_symA}.get(run.lr, run.lr)
    by_id = {run.rid: run for run in runs}
    print(f"{'run':16s} {'variant':10s} seed {'LR':7s} {'tokens':>8s} {'steps':>13s} {'decay':>6s} "
          f"{'evals':>5s} {'T4-h':>5s}  note")
    total = 0.0
    for run in runs:
        total += run.hours()
        print(f"{run.rid:16s} {run.variant:10s} {run.seed:4d} {run.lr:7s} {run.tokens / 1e9:7.3f}B "
              f"{run.start:6d}-{run.stop:<6d} {run.dstart:6d} {run.n_evals():5d} {run.hours():5.2f}  {run.note}")
    print(f"{'total':16s} {'':10s} {'':4s} {'':7s} {sum(r.tokens for r in runs) / 1e9:7.3f}B "
          f"{'':13s} {'':6s} {'':5s} {total:5.2f}")
    print()
    speed30 = 1.5 / 60  # 30M speed probe: ~1.5 min per GPU, both GPUs in parallel
    split = by_id["S_long_b"].hours()
    quota = 0.0
    gpu0_s23 = by_id["A1_fox"].hours() + by_id["A3_ell_rope"].hours()
    first_part = gpu0_s23 - by_id["A2_symA_rope"].hours()  # S_long_b runs on GPU1 until GPU0 finishes in S2-3
    for name, q0, q1, gate in sessions(by_id):
        def cost(q):
            h = 0.0
            for rid in q:
                if rid.startswith("speed30"):
                    h += speed30
                elif rid == "S_long_b*":
                    h += first_part if q is q1 and "A2_symA_rope" in q else split - first_part
                else:
                    h += by_id[rid].hours()
            return h
        h0, h1 = cost(q0), cost(q1)
        wall = max(h0, h1) + SESSION_OVERHEAD_H
        quota += wall
        print(f"{name:26s} GPU0 {h0:4.1f} h: {', '.join(q0)}")
        print(f"{'':26s} GPU1 {h1:4.1f} h: {', '.join(q1)}")
        print(f"{'':26s} session {wall:4.1f} quota-h, gate after it: {gate}")
    print(f"S2 total: {quota:.1f} quota-h ({total:.1f} T4-h of runs on 2 GPUs)")
    print("* S_long_b is split: it fills GPU1 in S2-3 after A2 and stops cleanly (--max-minutes), "
          "then resumes from its ckpt.pt in S2-4.")
    if args.commands:
        print()
        for run in runs:
            print(f"{run.rid}: python -u -m palimpsa.train {run.args()}")


if __name__ == "__main__":
    main()
