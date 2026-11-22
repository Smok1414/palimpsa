"""T11: S2 decision rules of scripts/kaggle_s2.py (LR choice, spike check)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from kaggle_s2 import Status, carry, chain_out, choose_lr, gate_g2a, gate_g2b, sha256_of, spike_check  # noqa: E402


def res(macro, step=1526, frac=0.0):
    return {"final_eval": {"macro": macro}, "final_eval_step": step, "skipped_frac": frac}


def ev(web, code, math, step=6638):
    return {"final_eval": {"web": web, "code": code, "math": math, "macro": (web + code + math) / 3},
            "final_eval_step": step}


def test_g2a_rule():
    base = ev(3.70, 2.00, 2.60)
    out = gate_g2a(base, ev(3.68, 1.99, 2.58), 6638)
    assert out["verdict"] == "continue to G2b" and not out["stop_reasons"] and not out["within_noise"]
    out = gate_g2a(base, ev(3.699, 1.999, 2.599), 6638)  # -0.005 < delta < 0: weak, still not a STOP
    assert out["verdict"] == "continue to G2b" and out["within_noise"]
    out = gate_g2a(base, ev(3.60, 1.90, 2.62), 6638)  # math +0.020 > +0.010 although the macro improves
    assert out["verdict"] == "STOP" and out["stop_reasons"] == ["math delta +0.0200 > +0.010"]
    assert gate_g2a(base, ev(3.71, 2.00, 2.60), 6638)["verdict"] == "STOP"  # delta_macro > 0
    assert gate_g2a(base, ev(3.6, 1.9, 2.5, step=5642), 6638)["verdict"] is None  # unfinished decay branch
    assert gate_g2a(base, ev(float("nan"), 1.9, 2.5), 6638)["verdict"] is None


def test_g2b_rule():
    base, sym = ev(3.65, 1.92, 2.62), ev(3.63, 1.90, 2.59)
    out = gate_g2b(base, sym, ev(3.64, 1.91, 2.60), ev(3.64, 1.91, 2.61), 6638)  # l helps in both settings
    assert out["verdict"] == "continue" and out["l_helps"] == {"without_rope": True, "with_rope": True}
    out = gate_g2b(base, sym, ev(3.62, 1.89, 2.58), ev(3.64, 1.91, 2.61), 6638)  # helps only with RoPE
    assert out["verdict"] == "continue" and out["l_helps"] == {"without_rope": False, "with_rope": True}
    out = gate_g2b(base, sym, ev(3.62, 1.89, 2.58), ev(3.66, 1.93, 2.63), 6638)  # helps in neither
    assert out["verdict"] == "STOP"
    assert gate_g2b(base, sym, ev(3.6, 1.9, 2.6, step=3000), ev(3.6, 1.9, 2.6), 6638)["verdict"] is None
    out = gate_g2b(base, sym, ev(3.64, 1.91, 2.60), ev(3.64, 1.91, 2.61), 6638, a2=ev(3.60, 1.88, 2.57))
    assert abs(out["A2_minus_symA"]["web"] + 0.03) < 1e-9


def test_carry_copies_listed_checkpoints_with_verification(tmp_path):
    prev, work = tmp_path / "prev", tmp_path / "work"
    for rel, data in {"s2/B_long_a/ckpt.pt": b"B" * 3_000_000, "s2/S_long_a/ckpt.pt": b"S" * 1000,
                      "s2/B_d435/ckpt.pt": b"x", "s2/B_long_a/metrics.jsonl": b"{}\n", "__output__.json": b"{}",
                      "status.json": b"{}", "palimpsa_chain.json": b"{}"}.items():
        (prev / rel).parent.mkdir(parents=True, exist_ok=True)
        (prev / rel).write_bytes(data)
    work.mkdir()
    good = sha256_of(prev / "s2/B_long_a/ckpt.pt")
    copied, pts = carry(prev, work, Status(work, "test"), keep_pt=("s2/B_long_a/ckpt.pt", "s2/S_long_a/ckpt.pt"),
                        prev_checks={"s2/B_long_a/ckpt.pt": good, "s2/S_long_a/ckpt.pt": "0" * 64})
    assert copied == ["s2/B_long_a/metrics.jsonl"]
    assert pts["s2/B_long_a/ckpt.pt"]["ok"] and sha256_of(work / "s2/B_long_a/ckpt.pt") == good
    assert not pts["s2/S_long_a/ckpt.pt"]["ok"] and not (work / "s2/S_long_a/ckpt.pt").exists()  # bad marker hash
    assert not (work / "s2/B_d435/ckpt.pt").exists() and not list(work.rglob("*.tmp"))
    marker = chain_out(work, 5, "test")
    assert marker["checks"]["s2/B_long_a/ckpt.pt"] == good  # the next hop can verify the checkpoint


def test_lr_tie_takes_smaller_and_edge_only_for_strict_optimum():
    out = choose_lr({"1.5e-3": res(4.30), "3e-3": res(4.200), "6e-3": res(4.196)}, 1526)
    assert out["choice"] == "3e-3" and out["argmin"] == "6e-3" and out["extra_point_needed"] is None
    out = choose_lr({"1.5e-3": res(4.30), "3e-3": res(4.25), "6e-3": res(4.20)}, 1526)
    assert out["choice"] == "6e-3" and out["extra_point_needed"] == "0.012"
    out = choose_lr({"1.5e-3": res(4.10), "3e-3": res(4.25), "6e-3": res(4.30)}, 1526)
    assert out["choice"] == "1.5e-3" and out["extra_point_needed"] == "0.00075"


def test_lr_exclusions():
    out = choose_lr({"1.5e-3": res(4.30), "3e-3": res(4.10, frac=0.02), "6e-3": res(4.0, step=900)}, 1526)
    assert out["choice"] == "1.5e-3" and set(out["excluded"]) == {"3e-3", "6e-3"}
    assert choose_lr({"3e-3": res(float("nan"))}, 1526)["choice"] is None


def test_spike_check():
    rows = [{"step": s, "loss": 5.0 - 0.001 * s, "tok_s": 1} for s in range(1, 400)]
    assert not spike_check(rows, warmup=200)["unrecovered_spike"]
    short = [dict(r, loss=r["loss"] + (0.5 if 250 <= r["step"] < 300 else 0)) for r in rows]
    out = spike_check(short, warmup=200)
    assert not out["unrecovered_spike"] and out["at_step"] == 250 and out["max_excess_over_running_min"] > 0.45
    long = [dict(r, loss=r["loss"] + (0.5 if r["step"] >= 250 else 0)) for r in rows]
    assert spike_check(long, warmup=200)["unrecovered_spike"]
    early = [dict(r, loss=r["loss"] + (1.0 if r["step"] < 150 else 0)) for r in rows]  # inside warmup: ignored
    assert spike_check(early, warmup=200)["max_excess_over_running_min"] < 0.01
