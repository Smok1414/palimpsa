# Provenance of the numbers in README.md

Each row: the value as printed, how it was computed (blank = copied as is, rounded), and where it comes from in `results/` (JSON pointer after `#`).

| README section | key | value | calculation | source |
|---|---|---|---|---|
| G0 speed | g0.100m.base | 12,702 |  | `results/g0_speed_smoke.json#probe_100M_micro4/base/tok_s_steady_median` |
| G0 speed | g0.100m.symA | 10,418 |  | `results/g0_speed_smoke.json#probe_100M_micro4/symA/tok_s_steady_median` |
| G0 speed | g0.100m.ratio | 0.820 | SymA tok/s / Base tok/s | `results/g0_speed_smoke.json#probe_100M_micro4/symA/tok_s_steady_median`<br>`results/g0_speed_smoke.json#probe_100M_micro4/base/tok_s_steady_median` |
| G0 speed | g0.100m.peak | 4.91 / 6.12 |  | `results/g0_speed_smoke.json#probe_100M_micro4/base/peak_gib`<br>`results/g0_speed_smoke.json#probe_100M_micro4/symA/peak_gib` |
| G0 speed | g0.10m.base | 39,039 |  | `results/g0_speed_smoke.json#smoke_10M_micro16/base/tok_s_steady_median` |
| G0 speed | g0.10m.symA | 32,563 |  | `results/g0_speed_smoke.json#smoke_10M_micro16/symA/tok_s_steady_median` |
| G0 speed | g0.10m.ratio | 0.834 | SymA tok/s / Base tok/s | `results/g0_speed_smoke.json#smoke_10M_micro16/symA/tok_s_steady_median`<br>`results/g0_speed_smoke.json#smoke_10M_micro16/base/tok_s_steady_median` |
| G0 speed | g0.10m.peak | 5.09 / 6.30 |  | `results/g0_speed_smoke.json#smoke_10M_micro16/base/peak_gib`<br>`results/g0_speed_smoke.json#smoke_10M_micro16/symA/peak_gib` |
| G0 speed | g0.30m.base | 23,199 |  | `results/s2_1_lr_sweep.json#runs/speed30_base/tok_s_median` |
| G0 speed | g0.30m.symA | 18,679 |  | `results/s2_1_lr_sweep.json#runs/speed30_symA/tok_s_median` |
| G0 speed | g0.30m.ratio | 0.805 | SymA tok/s / Base tok/s | `results/s2_1_lr_sweep.json#runs/speed30_symA/tok_s_median`<br>`results/s2_1_lr_sweep.json#runs/speed30_base/tok_s_median` |
| G0 speed | g0.30m.peak | 8.50 / 11.22 |  | `results/s2_1_lr_sweep.json#runs/speed30_base/peak_gib`<br>`results/s2_1_lr_sweep.json#runs/speed30_symA/peak_gib` |
| G0 speed | g0.sweep.base | 40,346 |  | `results/s2_1_lr_sweep.json#runs/sweep_B_3e-3/tok_s_median` |
| G0 speed | g0.sweep.symA | 32,891 |  | `results/s2_1_lr_sweep.json#runs/sweep_S_3e-3/tok_s_median` |
| G0 speed | g0.sweep.ratio | 0.815 | SymA tok/s / Base tok/s | `results/s2_1_lr_sweep.json#runs/sweep_S_3e-3/tok_s_median`<br>`results/s2_1_lr_sweep.json#runs/sweep_B_3e-3/tok_s_median` |
| G0 speed | g0.sweep.peak | 6.12 / 6.34 |  | `results/s2_1_lr_sweep.json#runs/sweep_B_3e-3/peak_gib`<br>`results/s2_1_lr_sweep.json#runs/sweep_S_3e-3/peak_gib` |
| G0 speed | g0.long.base | 37,743 |  | `results/s2_2_g2a.json#runs/B_long_a/tok_s_median` |
| G0 speed | g0.long.symA | 32,680 |  | `results/s2_2_g2a.json#runs/S_long_a/tok_s_median` |
| G0 speed | g0.long.ratio | 0.866 | SymA tok/s / Base tok/s | `results/s2_2_g2a.json#runs/S_long_a/tok_s_median`<br>`results/s2_2_g2a.json#runs/B_long_a/tok_s_median` |
| G0 speed | g0.long.peak | 6.12 / 6.34 |  | `results/s2_2_g2a.json#runs/B_long_a/peak_gib`<br>`results/s2_2_g2a.json#runs/S_long_a/peak_gib` |
| G0 speed | g0.ratio_min | 0.805 | smallest SymA / Base ratio in the table |  |
| G0 speed | g0.ratio_max | 0.866 | largest SymA / Base ratio in the table |  |
| G0 speed | g0.a1 | 32,345 |  | `results/s2_3a_g2b.json#runs/A1_fox/tok_s_median` |
| G0 speed | g0.a3 | 31,920 |  | `results/s2_3a_g2b.json#runs/A3_ell_rope/tok_s_median` |
| G0 speed | g0.resume_noise | 0.084 |  | `results/g0_speed_smoke.json#resume_check_cuda/max_abs_dloss_steps_1_25_noise_floor` |
| G0 speed | g0.resume_after | 0.024 |  | `results/g0_speed_smoke.json#resume_check_cuda/max_abs_dloss_steps_26_35_after_resume` |
| G0 speed | g0.resume_tol | 0.01 |  | `results/g0_speed_smoke.json#resume_check_cuda/tolerance` |
| G1 synthetic | g1.mqar1.base | 6.2 |  | `results/g1_synthetic.json#runs/mqar1/base/final_acc` |
| G1 synthetic | g1.mqar1.symA | 100.0 |  | `results/g1_synthetic.json#runs/mqar1/symA/final_acc` |
| G1 synthetic | g1.mqar1.fox | 4.6 |  | `results/g1_synthetic.json#runs/mqar1/fox/final_acc` |
| G1 synthetic | g1.mqar3.base | 6.8 |  | `results/g1_synthetic.json#runs/mqar3/base/final_acc` |
| G1 synthetic | g1.mqar3.symA | 100.0 |  | `results/g1_synthetic.json#runs/mqar3/symA/final_acc` |
| G1 synthetic | g1.mqar3.fox | 5.1 |  | `results/g1_synthetic.json#runs/mqar3/fox/final_acc` |
| G1 synthetic | g1.fuzzy3.base | 5.8 |  | `results/g1_synthetic.json#runs/fuzzy3/base/final_acc` |
| G1 synthetic | g1.fuzzy3.symA | 23.7 |  | `results/g1_synthetic.json#runs/fuzzy3/symA/final_acc` |
| G1 synthetic | g1.fuzzy3.fox | 3.8 |  | `results/g1_synthetic.json#runs/fuzzy3/fox/final_acc` |
| G1 synthetic | g1.trap.base | 7.2 |  | `results/g1_synthetic.json#runs/trap/base/final_acc` |
| G1 synthetic | g1.trap.symA | 100.0 |  | `results/g1_synthetic.json#runs/trap/symA/final_acc` |
| G1 synthetic | g1.trap.fox | 7.7 |  | `results/g1_synthetic.json#runs/trap/fox/final_acc` |
| G1 synthetic | g1.copy.base | 100.0 |  | `results/g1_synthetic.json#runs/copy/base/final_acc` |
| G1 synthetic | g1.copy.symA | 99.9 |  | `results/g1_synthetic.json#runs/copy/symA/final_acc` |
| G1 synthetic | g1.copy.fox | 4.4 |  | `results/g1_synthetic.json#runs/copy/fox/final_acc` |
| G1 synthetic | g1.verdict | PASS (b, c uninformative: Base did not learn mqar1) |  | `results/g1_synthetic.json#gate/G1` |
| G1 synthetic | g1.steps | 3,000 |  | `results/g1_synthetic.json#config/steps` |
| G1 synthetic | g1.seed | 0 |  | `results/g1_synthetic.json#config/seed` |
| G1 synthetic | g1.params_base | 886,400 |  | `results/g1_synthetic.json#runs/mqar1/base/params` |
| G1 synthetic | g1.params_symA | 887,488 |  | `results/g1_synthetic.json#runs/mqar1/symA/params` |
| G1 synthetic | g1.canary_base | 4.1603 |  | `results/g1_synthetic.json#gate/d_leak_canary/base` |
| G1 synthetic | g1.canary_symA | 4.1602 |  | `results/g1_synthetic.json#gate/d_leak_canary/symA` |
| G1 synthetic | g1.canary_leaky | 1.9261 |  | `results/g1_synthetic.json#gate/d_leak_canary/symA_leaky` |
| G1 synthetic | g1.base_mqar1 | 6.2 |  | `results/g1_synthetic.json#gate/validity_base_learns_mqar1/base_mqar1` |
| LR sweep 0.1B | lr.1.5e-3.base | 3.0702 |  | `results/s2_1_lr_sweep.json#runs/sweep_B_1.5e-3/final_eval/macro` |
| LR sweep 0.1B | lr.1.5e-3.symA | 3.0336 |  | `results/s2_1_lr_sweep.json#runs/sweep_S_1.5e-3/final_eval/macro` |
| LR sweep 0.1B | lr.1.5e-3.delta | -0.0366 | SymA - Base | `results/s2_1_lr_sweep.json#runs/sweep_S_1.5e-3/final_eval/macro`<br>`results/s2_1_lr_sweep.json#runs/sweep_B_1.5e-3/final_eval/macro` |
| LR sweep 0.1B | lr.3e-3.base | 3.0495 |  | `results/s2_1_lr_sweep.json#runs/sweep_B_3e-3/final_eval/macro` |
| LR sweep 0.1B | lr.3e-3.symA | 3.0151 |  | `results/s2_1_lr_sweep.json#runs/sweep_S_3e-3/final_eval/macro` |
| LR sweep 0.1B | lr.3e-3.delta | -0.0344 | SymA - Base | `results/s2_1_lr_sweep.json#runs/sweep_S_3e-3/final_eval/macro`<br>`results/s2_1_lr_sweep.json#runs/sweep_B_3e-3/final_eval/macro` |
| LR sweep 0.1B | lr.6e-3.base | 3.0593 |  | `results/s2_1_lr_sweep.json#runs/sweep_B_6e-3/final_eval/macro` |
| LR sweep 0.1B | lr.6e-3.symA | 3.0296 |  | `results/s2_1_lr_sweep.json#runs/sweep_S_6e-3/final_eval/macro` |
| LR sweep 0.1B | lr.6e-3.delta | -0.0297 | SymA - Base | `results/s2_1_lr_sweep.json#runs/sweep_S_6e-3/final_eval/macro`<br>`results/s2_1_lr_sweep.json#runs/sweep_B_6e-3/final_eval/macro` |
| LR sweep 0.1B | lr.steps | 1,526 |  | `results/s2_1_lr_sweep.json#runs/sweep_B_3e-3/steps_done` |
| LR sweep 0.1B | lr.choice_base | 3e-3 |  | `results/s2_1_lr_sweep.json#lr_choice/Base/choice` |
| LR sweep 0.1B | lr.choice_symA | 3e-3 |  | `results/s2_1_lr_sweep.json#lr_choice/SymA/choice` |
| G2a | g2a.macro.base | 2.7309 |  | `results/s2_2_g2a.json#g2a/base/macro` |
| G2a | g2a.macro.symA | 2.7081 |  | `results/s2_2_g2a.json#g2a/symA/macro` |
| G2a | g2a.macro.delta | -0.0228 |  | `results/s2_2_g2a.json#g2a/delta/macro` |
| G2a | g2a.web.base | 3.6531 |  | `results/s2_2_g2a.json#g2a/base/web` |
| G2a | g2a.web.symA | 3.6337 |  | `results/s2_2_g2a.json#g2a/symA/web` |
| G2a | g2a.web.delta | -0.0195 |  | `results/s2_2_g2a.json#g2a/delta/web` |
| G2a | g2a.code.base | 1.9202 |  | `results/s2_2_g2a.json#g2a/base/code` |
| G2a | g2a.code.symA | 1.8976 |  | `results/s2_2_g2a.json#g2a/symA/code` |
| G2a | g2a.code.delta | -0.0227 |  | `results/s2_2_g2a.json#g2a/delta/code` |
| G2a | g2a.math.base | 2.6193 |  | `results/s2_2_g2a.json#g2a/base/math` |
| G2a | g2a.math.symA | 2.5930 |  | `results/s2_2_g2a.json#g2a/symA/math` |
| G2a | g2a.math.delta | -0.0263 |  | `results/s2_2_g2a.json#g2a/delta/math` |
| G2a | g2a.steps | 6,638 |  | `results/s2_2_g2a.json#runs/B_d435/steps_done` |
| G2a | g2a.verdict | continue to G2b |  | `results/s2_2_g2a.json#g2a/verdict` |
| G2a | noise.ratio | 1.9 | |macro gap| / 0.012 (the project's one-seed noise planning level, not a measurement) | `results/s2_2_g2a.json#g2a/delta/macro` |
| G2a | g2a.tokens | 0.435B | steps_done x 65,536 tokens per step (train.py default global batch) | `results/s2_2_g2a.json#runs/B_d435/steps_done` |
| G2b | g2b.macro.Base | 2.7309 |  | `results/s2_3a_g2b.json#g2b/evals/Base/macro` |
| G2b | g2b.macro.A1_fox | 2.7255 |  | `results/s2_3a_g2b.json#g2b/evals/A1_fox/macro` |
| G2b | g2b.macro.A3_ell_rope | 2.7124 |  | `results/s2_3a_g2b.json#g2b/evals/A3_ell_rope/macro` |
| G2b | g2b.macro.SymA | 2.7081 |  | `results/s2_3a_g2b.json#g2b/evals/SymA/macro` |
| G2b | g2b.web.Base | 3.6531 |  | `results/s2_3a_g2b.json#g2b/evals/Base/web` |
| G2b | g2b.web.A1_fox | 3.6463 |  | `results/s2_3a_g2b.json#g2b/evals/A1_fox/web` |
| G2b | g2b.web.A3_ell_rope | 3.6375 |  | `results/s2_3a_g2b.json#g2b/evals/A3_ell_rope/web` |
| G2b | g2b.web.SymA | 3.6337 |  | `results/s2_3a_g2b.json#g2b/evals/SymA/web` |
| G2b | g2b.code.Base | 1.9202 |  | `results/s2_3a_g2b.json#g2b/evals/Base/code` |
| G2b | g2b.code.A1_fox | 1.9195 |  | `results/s2_3a_g2b.json#g2b/evals/A1_fox/code` |
| G2b | g2b.code.A3_ell_rope | 1.9006 |  | `results/s2_3a_g2b.json#g2b/evals/A3_ell_rope/code` |
| G2b | g2b.code.SymA | 1.8976 |  | `results/s2_3a_g2b.json#g2b/evals/SymA/code` |
| G2b | g2b.math.Base | 2.6193 |  | `results/s2_3a_g2b.json#g2b/evals/Base/math` |
| G2b | g2b.math.A1_fox | 2.6108 |  | `results/s2_3a_g2b.json#g2b/evals/A1_fox/math` |
| G2b | g2b.math.A3_ell_rope | 2.5992 |  | `results/s2_3a_g2b.json#g2b/evals/A3_ell_rope/math` |
| G2b | g2b.math.SymA | 2.5930 |  | `results/s2_3a_g2b.json#g2b/evals/SymA/math` |
| G2b | g2b.c.symA_a1.macro | -0.0175 |  | `results/s2_3a_g2b.json#g2b/symA_minus_A1/macro` |
| G2b | g2b.c.symA_a1.web | -0.0126 |  | `results/s2_3a_g2b.json#g2b/symA_minus_A1/web` |
| G2b | g2b.c.symA_a1.code | -0.0220 |  | `results/s2_3a_g2b.json#g2b/symA_minus_A1/code` |
| G2b | g2b.c.symA_a1.math | -0.0178 |  | `results/s2_3a_g2b.json#g2b/symA_minus_A1/math` |
| G2b | g2b.c.a3_base.macro | -0.0184 |  | `results/s2_3a_g2b.json#g2b/A3_minus_base/macro` |
| G2b | g2b.c.a3_base.web | -0.0156 |  | `results/s2_3a_g2b.json#g2b/A3_minus_base/web` |
| G2b | g2b.c.a3_base.code | -0.0196 |  | `results/s2_3a_g2b.json#g2b/A3_minus_base/code` |
| G2b | g2b.c.a3_base.math | -0.0201 |  | `results/s2_3a_g2b.json#g2b/A3_minus_base/math` |
| G2b | g2b.c.a1_base.macro | -0.0054 | A1 - Base | `results/s2_3a_g2b.json#g2b/evals/A1_fox/macro`<br>`results/s2_3a_g2b.json#g2b/evals/Base/macro` |
| G2b | g2b.c.a1_base.web | -0.0068 | A1 - Base | `results/s2_3a_g2b.json#g2b/evals/A1_fox/web`<br>`results/s2_3a_g2b.json#g2b/evals/Base/web` |
| G2b | g2b.c.a1_base.code | -0.0007 | A1 - Base | `results/s2_3a_g2b.json#g2b/evals/A1_fox/code`<br>`results/s2_3a_g2b.json#g2b/evals/Base/code` |
| G2b | g2b.c.a1_base.math | -0.0085 | A1 - Base | `results/s2_3a_g2b.json#g2b/evals/A1_fox/math`<br>`results/s2_3a_g2b.json#g2b/evals/Base/math` |
| G2b | g2b.c.symA_a3.macro | -0.0044 | SymA - A3 | `results/s2_3a_g2b.json#g2b/evals/SymA/macro`<br>`results/s2_3a_g2b.json#g2b/evals/A3_ell_rope/macro` |
| G2b | g2b.c.symA_a3.web | -0.0038 | SymA - A3 | `results/s2_3a_g2b.json#g2b/evals/SymA/web`<br>`results/s2_3a_g2b.json#g2b/evals/A3_ell_rope/web` |
| G2b | g2b.c.symA_a3.code | -0.0030 | SymA - A3 | `results/s2_3a_g2b.json#g2b/evals/SymA/code`<br>`results/s2_3a_g2b.json#g2b/evals/A3_ell_rope/code` |
| G2b | g2b.c.symA_a3.math | -0.0062 | SymA - A3 | `results/s2_3a_g2b.json#g2b/evals/SymA/math`<br>`results/s2_3a_g2b.json#g2b/evals/A3_ell_rope/math` |
| G2b | g2b.c.symA_base.macro | -0.0228 |  | `results/s2_2_g2a.json#g2a/delta/macro` |
| G2b | g2b.c.symA_base.web | -0.0195 |  | `results/s2_2_g2a.json#g2a/delta/web` |
| G2b | g2b.c.symA_base.code | -0.0227 |  | `results/s2_2_g2a.json#g2a/delta/code` |
| G2b | g2b.c.symA_base.math | -0.0263 |  | `results/s2_2_g2a.json#g2a/delta/math` |
| G2b | g2b.verdict | continue |  | `results/s2_3a_g2b.json#g2b/verdict` |
| gap trend | trend.0p1 | -0.0344 | SymA - Base, sweep at lr 3e-3 | `results/s2_1_lr_sweep.json#runs/sweep_S_3e-3/final_eval/macro`<br>`results/s2_1_lr_sweep.json#runs/sweep_B_3e-3/final_eval/macro` |
| gap trend | trend.5500 | -0.0279 | SymA - Base at step 5,500 of the long runs (constant LR, no decay) | `results/s2_2_g2a.json#runs/S_long_a/final_eval/macro`<br>`results/s2_2_g2a.json#runs/B_long_a/final_eval/macro` |
| gap trend | trend.0p435 | -0.0228 |  | `results/s2_2_g2a.json#g2a/delta/macro` |
| gap trend | trend.step | 5,500 |  | `results/s2_2_g2a.json#runs/B_long_a/final_eval_step` |
| parameters | params.base | 21,730,944 |  | `results/benchmarks_user_measured.json#results/base/params` |
| parameters | params.symA | 21,745,056 |  | `results/benchmarks_user_measured.json#results/symA/params` |
| parameters | params.emb | 12,288,000 | vocab 32,000 x d_model 384 (palimpsa/model.py LADDER['10M']) |  |
| parameters | params.base_ne | 9,442,944 | total - embedding table | `results/benchmarks_user_measured.json#results/base/params` |
| parameters | params.symA_ne | 9,457,056 | total - embedding table | `results/benchmarks_user_measured.json#results/symA/params` |
| benchmarks | bm.arc_easy.base.acc | 40.9 |  | `results/benchmarks_user_measured.json#results/base/benchmarks/arc_easy/acc` |
| benchmarks | bm.arc_easy.base.acc_norm | 37.2 |  | `results/benchmarks_user_measured.json#results/base/benchmarks/arc_easy/acc_norm` |
| benchmarks | bm.hellaswag.base.acc | 26.7 |  | `results/benchmarks_user_measured.json#results/base/benchmarks/hellaswag/acc` |
| benchmarks | bm.hellaswag.base.acc_norm | 27.1 |  | `results/benchmarks_user_measured.json#results/base/benchmarks/hellaswag/acc_norm` |
| benchmarks | bm.piqa.base.acc | 56.2 |  | `results/benchmarks_user_measured.json#results/base/benchmarks/piqa/acc` |
| benchmarks | bm.piqa.base.acc_norm | 55.3 |  | `results/benchmarks_user_measured.json#results/base/benchmarks/piqa/acc_norm` |
| benchmarks | bm.mathqa.base.acc | 20.3 |  | `results/benchmarks_user_measured.json#results/base/benchmarks/mathqa/acc` |
| benchmarks | bm.mathqa.base.acc_norm | 20.2 |  | `results/benchmarks_user_measured.json#results/base/benchmarks/mathqa/acc_norm` |
| benchmarks | bm.arc_easy.symA.acc | 40.3 |  | `results/benchmarks_user_measured.json#results/symA/benchmarks/arc_easy/acc` |
| benchmarks | bm.arc_easy.symA.acc_norm | 36.1 |  | `results/benchmarks_user_measured.json#results/symA/benchmarks/arc_easy/acc_norm` |
| benchmarks | bm.hellaswag.symA.acc | 26.8 |  | `results/benchmarks_user_measured.json#results/symA/benchmarks/hellaswag/acc` |
| benchmarks | bm.hellaswag.symA.acc_norm | 27.5 |  | `results/benchmarks_user_measured.json#results/symA/benchmarks/hellaswag/acc_norm` |
| benchmarks | bm.piqa.symA.acc | 56.1 |  | `results/benchmarks_user_measured.json#results/symA/benchmarks/piqa/acc` |
| benchmarks | bm.piqa.symA.acc_norm | 55.2 |  | `results/benchmarks_user_measured.json#results/symA/benchmarks/piqa/acc_norm` |
| benchmarks | bm.mathqa.symA.acc | 20.8 |  | `results/benchmarks_user_measured.json#results/symA/benchmarks/mathqa/acc` |
| benchmarks | bm.mathqa.symA.acc_norm | 21.2 |  | `results/benchmarks_user_measured.json#results/symA/benchmarks/mathqa/acc_norm` |
| benchmarks | bm.arc_easy.fox.acc | 40.1 |  | `results/benchmarks_user_measured.json#results/fox/benchmarks/arc_easy/acc` |
| benchmarks | bm.arc_easy.fox.acc_norm | 36.0 |  | `results/benchmarks_user_measured.json#results/fox/benchmarks/arc_easy/acc_norm` |
| benchmarks | bm.hellaswag.fox.acc | 26.4 |  | `results/benchmarks_user_measured.json#results/fox/benchmarks/hellaswag/acc` |
| benchmarks | bm.hellaswag.fox.acc_norm | 27.0 |  | `results/benchmarks_user_measured.json#results/fox/benchmarks/hellaswag/acc_norm` |
| benchmarks | bm.piqa.fox.acc | 56.6 |  | `results/benchmarks_user_measured.json#results/fox/benchmarks/piqa/acc` |
| benchmarks | bm.piqa.fox.acc_norm | 55.3 |  | `results/benchmarks_user_measured.json#results/fox/benchmarks/piqa/acc_norm` |
| benchmarks | bm.mathqa.fox.acc | 21.0 |  | `results/benchmarks_user_measured.json#results/fox/benchmarks/mathqa/acc` |
| benchmarks | bm.mathqa.fox.acc_norm | 22.0 |  | `results/benchmarks_user_measured.json#results/fox/benchmarks/mathqa/acc_norm` |
| benchmarks | bm.arc_easy.ell_rope.acc | 40.4 |  | `results/benchmarks_user_measured.json#results/ell_rope/benchmarks/arc_easy/acc` |
| benchmarks | bm.arc_easy.ell_rope.acc_norm | 36.7 |  | `results/benchmarks_user_measured.json#results/ell_rope/benchmarks/arc_easy/acc_norm` |
| benchmarks | bm.hellaswag.ell_rope.acc | 26.7 |  | `results/benchmarks_user_measured.json#results/ell_rope/benchmarks/hellaswag/acc` |
| benchmarks | bm.hellaswag.ell_rope.acc_norm | 27.2 |  | `results/benchmarks_user_measured.json#results/ell_rope/benchmarks/hellaswag/acc_norm` |
| benchmarks | bm.piqa.ell_rope.acc | 57.5 |  | `results/benchmarks_user_measured.json#results/ell_rope/benchmarks/piqa/acc` |
| benchmarks | bm.piqa.ell_rope.acc_norm | 56.0 |  | `results/benchmarks_user_measured.json#results/ell_rope/benchmarks/piqa/acc_norm` |
| benchmarks | bm.mathqa.ell_rope.acc | 20.9 |  | `results/benchmarks_user_measured.json#results/ell_rope/benchmarks/mathqa/acc` |
| benchmarks | bm.mathqa.ell_rope.acc_norm | 20.8 |  | `results/benchmarks_user_measured.json#results/ell_rope/benchmarks/mathqa/acc_norm` |
| benchmarks | bm.arc_easy.chance | 25.0 |  | `results/benchmarks/arc_easy_0shot.json#uniform_choice_baseline` |
| benchmarks | bm.hellaswag.chance | 25.0 |  | `results/benchmarks/hellaswag_0shot.json#uniform_choice_baseline` |
| benchmarks | bm.piqa.chance | 50.0 |  | `results/benchmarks/piqa_0shot.json#uniform_choice_baseline` |
| benchmarks | bm.mathqa.chance | 20.0 |  | `results/benchmarks/mathqa_0shot.json#uniform_choice_baseline` |
| benchmarks | bm.arc_easy.n | 2,376 |  | `results/benchmarks/arc_easy_0shot.json#questions` |
| benchmarks | bm.hellaswag.n | 10,042 |  | `results/benchmarks/hellaswag_0shot.json#examples` |
| benchmarks | bm.piqa.n | 1,838 |  | `results/benchmarks/piqa_0shot.json#examples` |
| benchmarks | bm.mathqa.n | 4,475 |  | `results/benchmarks/mathqa_0shot.json#examples` |
| benchmarks | bm.maxdiff | 1.9 | largest |variant - Base| over the four benchmarks, both metrics | `results/benchmarks_user_measured.json#results/base/benchmarks/arc_easy/acc`<br>`results/benchmarks_user_measured.json#results/symA/benchmarks/arc_easy/acc`<br>`results/benchmarks_user_measured.json#results/fox/benchmarks/arc_easy/acc`<br>`results/benchmarks_user_measured.json#results/ell_rope/benchmarks/arc_easy/acc`<br>`results/benchmarks_user_measured.json#results/base/benchmarks/arc_easy/acc_norm`<br>`results/benchmarks_user_measured.json#results/symA/benchmarks/arc_easy/acc_norm`<br>`results/benchmarks_user_measured.json#results/fox/benchmarks/arc_easy/acc_norm`<br>`results/benchmarks_user_measured.json#results/ell_rope/benchmarks/arc_easy/acc_norm`<br>`results/benchmarks_user_measured.json#results/base/benchmarks/hellaswag/acc`<br>`results/benchmarks_user_measured.json#results/symA/benchmarks/hellaswag/acc`<br>`results/benchmarks_user_measured.json#results/fox/benchmarks/hellaswag/acc`<br>`results/benchmarks_user_measured.json#results/ell_rope/benchmarks/hellaswag/acc`<br>`results/benchmarks_user_measured.json#results/base/benchmarks/hellaswag/acc_norm`<br>`results/benchmarks_user_measured.json#results/symA/benchmarks/hellaswag/acc_norm`<br>`results/benchmarks_user_measured.json#results/fox/benchmarks/hellaswag/acc_norm`<br>`results/benchmarks_user_measured.json#results/ell_rope/benchmarks/hellaswag/acc_norm`<br>`results/benchmarks_user_measured.json#results/base/benchmarks/piqa/acc`<br>`results/benchmarks_user_measured.json#results/symA/benchmarks/piqa/acc`<br>`results/benchmarks_user_measured.json#results/fox/benchmarks/piqa/acc`<br>`results/benchmarks_user_measured.json#results/ell_rope/benchmarks/piqa/acc`<br>`results/benchmarks_user_measured.json#results/base/benchmarks/piqa/acc_norm`<br>`results/benchmarks_user_measured.json#results/symA/benchmarks/piqa/acc_norm`<br>`results/benchmarks_user_measured.json#results/fox/benchmarks/piqa/acc_norm`<br>`results/benchmarks_user_measured.json#results/ell_rope/benchmarks/piqa/acc_norm`<br>`results/benchmarks_user_measured.json#results/base/benchmarks/mathqa/acc`<br>`results/benchmarks_user_measured.json#results/symA/benchmarks/mathqa/acc`<br>`results/benchmarks_user_measured.json#results/fox/benchmarks/mathqa/acc`<br>`results/benchmarks_user_measured.json#results/ell_rope/benchmarks/mathqa/acc`<br>`results/benchmarks_user_measured.json#results/base/benchmarks/mathqa/acc_norm`<br>`results/benchmarks_user_measured.json#results/symA/benchmarks/mathqa/acc_norm`<br>`results/benchmarks_user_measured.json#results/fox/benchmarks/mathqa/acc_norm`<br>`results/benchmarks_user_measured.json#results/ell_rope/benchmarks/mathqa/acc_norm` |
| benchmarks | mc.minp.arc_easy | 0.121 | smallest of the six paired p-values | `results/benchmarks_paired_mcnemar.json#results/arc_easy/acc_symA/p`<br>`results/benchmarks_paired_mcnemar.json#results/arc_easy/acc_fox/p`<br>`results/benchmarks_paired_mcnemar.json#results/arc_easy/acc_ell_rope/p`<br>`results/benchmarks_paired_mcnemar.json#results/arc_easy/acc_norm_symA/p`<br>`results/benchmarks_paired_mcnemar.json#results/arc_easy/acc_norm_fox/p`<br>`results/benchmarks_paired_mcnemar.json#results/arc_easy/acc_norm_ell_rope/p` |
| benchmarks | mc.minp.hellaswag | 0.166 | smallest of the six paired p-values | `results/benchmarks_paired_mcnemar.json#results/hellaswag/acc_symA/p`<br>`results/benchmarks_paired_mcnemar.json#results/hellaswag/acc_fox/p`<br>`results/benchmarks_paired_mcnemar.json#results/hellaswag/acc_ell_rope/p`<br>`results/benchmarks_paired_mcnemar.json#results/hellaswag/acc_norm_symA/p`<br>`results/benchmarks_paired_mcnemar.json#results/hellaswag/acc_norm_fox/p`<br>`results/benchmarks_paired_mcnemar.json#results/hellaswag/acc_norm_ell_rope/p` |
| benchmarks | mc.minp.piqa | 0.091 | smallest of the six paired p-values | `results/benchmarks_paired_mcnemar.json#results/piqa/acc_symA/p`<br>`results/benchmarks_paired_mcnemar.json#results/piqa/acc_fox/p`<br>`results/benchmarks_paired_mcnemar.json#results/piqa/acc_ell_rope/p`<br>`results/benchmarks_paired_mcnemar.json#results/piqa/acc_norm_symA/p`<br>`results/benchmarks_paired_mcnemar.json#results/piqa/acc_norm_fox/p`<br>`results/benchmarks_paired_mcnemar.json#results/piqa/acc_norm_ell_rope/p` |
| benchmarks | mc.minp.mathqa | 0.001 | smallest of the six paired p-values | `results/benchmarks_paired_mcnemar.json#results/mathqa/acc_symA/p`<br>`results/benchmarks_paired_mcnemar.json#results/mathqa/acc_fox/p`<br>`results/benchmarks_paired_mcnemar.json#results/mathqa/acc_ell_rope/p`<br>`results/benchmarks_paired_mcnemar.json#results/mathqa/acc_norm_symA/p`<br>`results/benchmarks_paired_mcnemar.json#results/mathqa/acc_norm_fox/p`<br>`results/benchmarks_paired_mcnemar.json#results/mathqa/acc_norm_ell_rope/p` |
| benchmarks | mc.sig.mathqa.acc_norm_symA.pp | +1.0 | variant - Base, percentage points | `results/benchmarks_user_measured.json#results/symA/benchmarks/mathqa/acc_norm`<br>`results/benchmarks_user_measured.json#results/base/benchmarks/mathqa/acc_norm` |
| benchmarks | mc.sig.mathqa.acc_norm_symA.p | 0.0323 |  | `results/benchmarks_paired_mcnemar.json#results/mathqa/acc_norm_symA/p` |
| benchmarks | mc.sig.mathqa.acc_norm_fox.pp | +1.9 | variant - Base, percentage points | `results/benchmarks_user_measured.json#results/fox/benchmarks/mathqa/acc_norm`<br>`results/benchmarks_user_measured.json#results/base/benchmarks/mathqa/acc_norm` |
| benchmarks | mc.sig.mathqa.acc_norm_fox.p | 0.0009 |  | `results/benchmarks_paired_mcnemar.json#results/mathqa/acc_norm_fox/p` |
| benchmarks | mc.n_tests | 24 | count of paired comparisons over the four benchmarks above |  |
| benchmarks | arith.min | 32.1 | lowest acc_norm of the four models | `results/benchmarks_user_measured.json#results/base/benchmarks/arithmark3/acc_norm`<br>`results/benchmarks_user_measured.json#results/symA/benchmarks/arithmark3/acc_norm`<br>`results/benchmarks_user_measured.json#results/fox/benchmarks/arithmark3/acc_norm`<br>`results/benchmarks_user_measured.json#results/ell_rope/benchmarks/arithmark3/acc_norm` |
| benchmarks | arith.max | 36.5 | highest acc_norm of the four models |  |
| repetition | rep.1.0.0.3.B_d435 | 84 |  | `results/repetition/rep_B_d435_p1.0.json#repetition/T=0.3/all/share_repeats` |
| repetition | rep.1.0.0.3.A3_ell_rope | 87 |  | `results/repetition/rep_A3_ell_rope_p1.0.json#repetition/T=0.3/all/share_repeats` |
| repetition | rep.1.0.0.3.A1_fox | 85 |  | `results/repetition/rep_A1_fox_p1.0.json#repetition/T=0.3/all/share_repeats` |
| repetition | rep.1.0.0.3.S_d435 | 90 |  | `results/repetition/rep_S_d435_p1.0.json#repetition/T=0.3/all/share_repeats` |
| repetition | rep.1.0.0.3.a3 | +3 (0.547) | difference of the two shares (n = 100 texts per model); pooled two-proportion z-test | `results/repetition/rep_A3_ell_rope_p1.0.json#repetition/T=0.3/all/share_repeats`<br>`results/repetition/rep_B_d435_p1.0.json#repetition/T=0.3/all/share_repeats` |
| repetition | rep.1.0.0.3.s | +5 (0.285) | difference of the two shares (n = 100 texts per model); pooled two-proportion z-test | `results/repetition/rep_S_d435_p1.0.json#repetition/T=0.3/all/share_repeats`<br>`results/repetition/rep_A1_fox_p1.0.json#repetition/T=0.3/all/share_repeats` |
| repetition | rep.1.0.0.7.B_d435 | 23 |  | `results/repetition/rep_B_d435_p1.0.json#repetition/T=0.7/all/share_repeats` |
| repetition | rep.1.0.0.7.A3_ell_rope | 32 |  | `results/repetition/rep_A3_ell_rope_p1.0.json#repetition/T=0.7/all/share_repeats` |
| repetition | rep.1.0.0.7.A1_fox | 19 |  | `results/repetition/rep_A1_fox_p1.0.json#repetition/T=0.7/all/share_repeats` |
| repetition | rep.1.0.0.7.S_d435 | 25 |  | `results/repetition/rep_S_d435_p1.0.json#repetition/T=0.7/all/share_repeats` |
| repetition | rep.1.0.0.7.a3 | +9 (0.154) | difference of the two shares (n = 100 texts per model); pooled two-proportion z-test | `results/repetition/rep_A3_ell_rope_p1.0.json#repetition/T=0.7/all/share_repeats`<br>`results/repetition/rep_B_d435_p1.0.json#repetition/T=0.7/all/share_repeats` |
| repetition | rep.1.0.0.7.s | +6 (0.306) | difference of the two shares (n = 100 texts per model); pooled two-proportion z-test | `results/repetition/rep_S_d435_p1.0.json#repetition/T=0.7/all/share_repeats`<br>`results/repetition/rep_A1_fox_p1.0.json#repetition/T=0.7/all/share_repeats` |
| repetition | rep.1.2.0.3.B_d435 | 39 |  | `results/repetition/rep_B_d435_p1.2.json#repetition/T=0.3/all/share_repeats` |
| repetition | rep.1.2.0.3.A3_ell_rope | 57 |  | `results/repetition/rep_A3_ell_rope_p1.2.json#repetition/T=0.3/all/share_repeats` |
| repetition | rep.1.2.0.3.A1_fox | 43 |  | `results/repetition/rep_A1_fox_p1.2.json#repetition/T=0.3/all/share_repeats` |
| repetition | rep.1.2.0.3.S_d435 | 45 |  | `results/repetition/rep_S_d435_p1.2.json#repetition/T=0.3/all/share_repeats` |
| repetition | rep.1.2.0.3.a3 | +18 (0.011) | difference of the two shares (n = 100 texts per model); pooled two-proportion z-test | `results/repetition/rep_A3_ell_rope_p1.2.json#repetition/T=0.3/all/share_repeats`<br>`results/repetition/rep_B_d435_p1.2.json#repetition/T=0.3/all/share_repeats` |
| repetition | rep.1.2.0.3.s | +2 (0.776) | difference of the two shares (n = 100 texts per model); pooled two-proportion z-test | `results/repetition/rep_S_d435_p1.2.json#repetition/T=0.3/all/share_repeats`<br>`results/repetition/rep_A1_fox_p1.2.json#repetition/T=0.3/all/share_repeats` |
| repetition | rep.1.2.0.7.B_d435 | 1 |  | `results/repetition/rep_B_d435_p1.2.json#repetition/T=0.7/all/share_repeats` |
| repetition | rep.1.2.0.7.A3_ell_rope | 2 |  | `results/repetition/rep_A3_ell_rope_p1.2.json#repetition/T=0.7/all/share_repeats` |
| repetition | rep.1.2.0.7.A1_fox | 1 |  | `results/repetition/rep_A1_fox_p1.2.json#repetition/T=0.7/all/share_repeats` |
| repetition | rep.1.2.0.7.S_d435 | 3 |  | `results/repetition/rep_S_d435_p1.2.json#repetition/T=0.7/all/share_repeats` |
| repetition | rep.1.2.0.7.a3 | +1 (0.561) | difference of the two shares (n = 100 texts per model); pooled two-proportion z-test | `results/repetition/rep_A3_ell_rope_p1.2.json#repetition/T=0.7/all/share_repeats`<br>`results/repetition/rep_B_d435_p1.2.json#repetition/T=0.7/all/share_repeats` |
| repetition | rep.1.2.0.7.s | +2 (0.312) | difference of the two shares (n = 100 texts per model); pooled two-proportion z-test | `results/repetition/rep_S_d435_p1.2.json#repetition/T=0.7/all/share_repeats`<br>`results/repetition/rep_A1_fox_p1.2.json#repetition/T=0.7/all/share_repeats` |
| repetition | rep.wins | 8 | number of the 8 contrasts in which the model with ℓ repeats more |  |
| repetition | rep.per_domain | 50 |  | `results/repetition/rep_B_d435_p1.0.json#repetition/settings/per_domain` |
| repetition | rep.prompt_len | 32 |  | `results/repetition/rep_B_d435_p1.0.json#repetition/settings/prompt_len` |
| repetition | rep.new_tokens | 120 |  | `results/repetition/rep_B_d435_p1.0.json#repetition/settings/new_tokens` |
| repetition | rep.seed | 0 |  | `results/repetition/rep_B_d435_p1.0.json#repetition/settings/seed` |
| repetition | rep.ref_d4_web | 0.99 |  | `results/repetition/rep_B_d435_p1.0.json#repetition/reference/web/distinct4_mean` |
| repetition | rep.ref_d4_code | 0.92 |  | `results/repetition/rep_B_d435_p1.0.json#repetition/reference/code/distinct4_mean` |
| repetition | rep.ref_share | 0.0 | larger of the web and code reference shares | `results/repetition/rep_B_d435_p1.0.json#repetition/reference/web/share_repeats`<br>`results/repetition/rep_B_d435_p1.0.json#repetition/reference/code/share_repeats` |
| repetition | rep.chat_max | 3 | penalty 1.2, T 0.7: largest share over the four models | `results/repetition/rep_B_d435_p1.2.json#repetition/T=0.7/all/share_repeats`<br>`results/repetition/rep_A3_ell_rope_p1.2.json#repetition/T=0.7/all/share_repeats`<br>`results/repetition/rep_A1_fox_p1.2.json#repetition/T=0.7/all/share_repeats`<br>`results/repetition/rep_S_d435_p1.2.json#repetition/T=0.7/all/share_repeats` |
| data | data.web.tok | 1,585,818,532 |  | `results/data_stats.json#train/web/train_tokens` |
| data | data.web.docs | 1,452,946 |  | `results/data_stats.json#train/web/train_docs` |
| data | data.web.vtok | 2,185,289 |  | `results/data_stats.json#validation_dedup/web/val_tokens` |
| data | data.web.vdocs | 1,991 |  | `results/data_stats.json#validation_dedup/web/val_docs` |
| data | data.code.tok | 527,949,558 |  | `results/data_stats.json#train/code/train_tokens` |
| data | data.code.docs | 332,386 |  | `results/data_stats.json#train/code/train_docs` |
| data | data.code.vtok | 2,444,229 |  | `results/data_stats.json#validation_dedup/code/val_tokens` |
| data | data.code.vdocs | 1,494 |  | `results/data_stats.json#validation_dedup/code/val_docs` |
| data | data.math.tok | 333,731,463 |  | `results/data_stats.json#train/math/train_tokens` |
| data | data.math.docs | 207,887 |  | `results/data_stats.json#train/math/train_docs` |
| data | data.math.vtok | 2,058,640 |  | `results/data_stats.json#validation_dedup/math/val_tokens` |
| data | data.math.vdocs | 1,366 |  | `results/data_stats.json#validation_dedup/math/val_docs` |
| data | data.bpt.web | 4.414 |  | `results/data_stats.json#tokenizer/bytes_per_token/web` |
| data | data.bpt.code | 3.377 |  | `results/data_stats.json#tokenizer/bytes_per_token/code` |
| data | data.bpt.math | 3.018 |  | `results/data_stats.json#tokenizer/bytes_per_token/math` |
| data | data.vocab | 32,000 |  | `results/data_stats.json#tokenizer/vocab` |
| data | data.sample_mb | 300 |  | `results/data_stats.json#tokenizer/sample_mb` |
| data | data.ngram | 13 |  | `results/data_stats.json#validation_dedup_settings/ngram` |
| data | data.overlap | 50% |  | `results/data_stats.json#validation_dedup_settings/overlap_frac` |

## Configuration facts stated in the README text

These are settings and constants, not measurements. They are read from the code or the files named here.

| Statement in the README | Source |
|---|---|
| 10M stage: d_model 384, 6 layers, 6 heads, 2 KV heads, d_ff 1024 | `palimpsa/model.py`: `LADDER["10M"]`, `ladder_config` (heads = d/64, KV heads = heads/3, d_ff = 8d/3) |
| Vocabulary 32,000, tied embeddings | `palimpsa/model.py`: `Config.vocab`, `PalimpsaLM.forward` (head = embedding matrix) |
| Sequence 1,024, global batch 65,536 tokens, micro-batch 16, AdamW (0.9, 0.95), weight decay 0.1, clipping 1.0, z-loss 1e-4, decay over the last 15%, fp16 autocast with a gradient scaler | `palimpsa/train.py`: `build_args`, `main` |
| Warmup 200 steps, 1,953 validation windows, seed 1, step counts 1,526 / 5,642 / 6,638, LR grid 1.5e-3, 3e-3, 6e-3 | `scripts/plan_s2.py` (`python scripts/plan_s2.py --commands` prints every run) |
| Gate G2a stops only if the macro difference is >= 0 or a domain is worse by more than 0.010; G2b stops if the l bias helps in neither setting | `scripts/kaggle_s2.py`: `gate_g2a`, `gate_g2b`; checked by `tests/test_t11_s2_rules.py` |
| Training windows of 1,025 tokens; domain probabilities 0.60 / 0.25 / 0.15 | `palimpsa/loader.py`: `TrainStream.batch`, `MIX` |
| Validation de-duplication: 13-grams, 50% overlap | `results/data_stats.json` (`validation_dedup_settings`) |
| G1 models (4 layers, d_model 128, 2 heads), task definitions, 48-token copy string | `scripts/run_s1.py`: `model_config`; `palimpsa/synth.py` |
| ln 64 = 4.1589 (leakage canary reference) | `palimpsa/canary.py`: `ALPHABET = 64` |
| Repetition criterion (tail share >= 0.25 or distinct-4 < 0.5), 32-token prompts, 120 new tokens, T = 0.3 / 0.7, penalties 1.0 / 1.2 | `results/repetition/*.json` (`settings`, file names) |
| Benchmarks measured at step 6,638 | `results/benchmarks_user_measured.json` (`common_checkpoint_step`) |
| One-seed noise level 0.012 nats/token | planning assumption of the project (working-copy planning documents, not part of this repository) |
| Shard counts ("2 of 14", "2 of 64"), 5 code files, about 5.4 GB download | `configs/raw_files.json` (file list and sizes); the totals "of 14" and "of 64" are from the working-copy data notes |
| Dataset licenses as listed on 2026-11-21 | working-copy data notes (Hugging Face API metadata check of that date) |
| Python 3.14.3, torch 2.11.0, numpy 2.4.3, pytest 9.0.3 | the machine used for the checks of this snapshot |
| G1 ran with torch 2.11.0+cu128 | first line of `results/g1_synthetic.log` |
| Kaggle image with Python 3.13 | Kaggle version logs of the S2 sessions (not part of this repository) |
