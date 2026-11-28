#!/usr/bin/env python3
"""Print the numbered items that results.md cites, from the saved files in results/ (pandas and the standard library).

    python3 -I report_numbers.py > results/report_numbers.txt
"""
import json
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parent / "results"
S = json.loads((R / "summary_numbers.json").read_text())
P = S["predictions"]
tr = json.loads((R / "trace_summary.json").read_text())
ctl = json.loads((R / "control_random_drop_summary.json").read_text())
w02 = pd.read_csv(R / "with_without_02.csv")
w09 = pd.read_csv(R / "with_without_09.csv")
w03 = pd.read_csv(R / "with_without_03.csv").set_index("kind")
w04 = pd.read_csv(R / "with_without_04.csv").set_index("arm")
hd = pd.read_csv(R / "huber_delta_sweep_both_arms.csv")
har = pd.read_csv(R / "harness_check.csv", dtype={"experiment": str})
inv = pd.read_csv(R / "arm_invariance.csv", dtype={"experiment": str})
gains = pd.read_csv(R / "extra_term_gains.csv", dtype={"exp": str})
short = pd.read_csv(R / "trace_short_runs.csv")
f = lambda x, n=6: f"{x:.{n}f}"
e = lambda x: f"{x:.1e}"
items = []


def item(t):
    items.append(t)


item(f"Arms: all245 keeps all {S['n_runs']['all245']} Chinchilla runs; drop_DN_lt_1 keeps the {S['n_runs']['drop_DN_lt_1']} runs with D/N of at least 1 and removes "
     f"{tr['prediction_6']['short_runs']} runs ({tr['prediction_6']['in_1e19_column']} in the column of runs near 1e19 FLOP, 1 at N=12.57 billion near 6e20 FLOP).")
for r in har.itertuples():
    item(f"Harness check, experiment {r.experiment}, {r.what} ({r.rows} rows): largest relative difference between arm all245 and the committed outputs {e(r.max_rel_diff)}, "
         f"at {r.worst_at}.")
item(f"Prediction 5 (arm all245 reproduces committed key numbers to 1e-6): {'PASSED' if P['5']['passed'] else 'FAILED'}; largest relative difference {e(P['5']['max_rel_diff_chinchilla_rows'])} "
     f"on Chinchilla-only rows (all of experiment 04 and the Chinchilla-run rows of experiments 02, 03 and 09) and {e(P['5']['max_rel_diff_all_rows'])} over all rows (threshold {e(P['5']['threshold'])}).")
for r in inv.itertuples():
    item(f"Arm invariance, experiment {r.experiment}, {r.rows}: {r.n} rows, largest absolute difference between arms {r.max_abs_diff_between_arms:g}.")
p1 = P["1"]
item(f"Prediction 1: joint model, size holdout, held-out pooled RMSE {f(p1['rmse_02_all'])} with all runs and {f(p1['rmse_02_drop'])} without the 11 (experiment 02), "
     f"{f(p1['rmse_09_all'])} and {f(p1['rmse_09_drop'])} (experiment 09); fall {100*p1['reduction_02']:.1f}% and {100*p1['reduction_09']:.1f}%; "
     f"macro RMSE {f(p1['macro_02_all'])} to {f(p1['macro_02_drop'])}, fall {100*p1['macro_reduction_02']:.1f}%; {'PASSED' if p1['passed'] else 'FAILED'}.")
d = {(r["protocol"], r["model"]): r for r in S["decomposition"]}
for m in ("joint", "joint_extra_n", "compute_one", "compute_two", "n_only", "d_only"):
    r = d[("size", m)]
    item(f"Decomposition (post hoc), size holdout, {m}: pooled RMSE of the all-runs model {f(r['rmse_all245_model_on_all_test'])} on all {r['n_test_all']} test runs, "
         f"{f(r['rmse_all245_model_on_kept_test'])} on the {r['n_test_kept']} test runs that survive, and {f(r['rmse_drop_model_on_kept_test'])} for the model retrained without the 11 on the same "
         f"{r['n_test_kept']} runs (same test rows: {r['same_test_rows_as_drop_arm']}).")
g = gains
p2 = P["2"]
item(f"Prediction 2: across {p2['n_cases']} base-versus-extra-term comparisons in experiments 02 and 09, both arms, pooled and macro RMSE, the largest relative gain of an extra-term model over its base is "
     f"{100*p2['max_gain_any_case']:.4f}%; {p2['n_cases_with_positive_gain']} comparisons have a positive gain; {'PASSED' if p2['passed'] else 'FAILED'} (threshold 2%).")
q = g[(g.exp == "02") & (g.arm == "drop_DN_lt_1") & (g.protocol == "size") & (g.extra == "joint_extra_n")].iloc[0]
item(f"Experiment 02, arm without the 11, size holdout: joint_extra_n pooled RMSE {f(w02[(w02.protocol=='size')&(w02.model=='joint_extra_n')].test_rmse_drop.iloc[0])} against joint "
     f"{f(w02[(w02.protocol=='size')&(w02.model=='joint')].test_rmse_drop.iloc[0])}, {100*(-q.gain_pooled):.1f}% worse; macro RMSE {f(w02[(w02.protocol=='size')&(w02.model=='joint_extra_n')].test_macro_rmse_drop.iloc[0])} "
     f"against {f(w02[(w02.protocol=='size')&(w02.model=='joint')].test_macro_rmse_drop.iloc[0])}.")
for kind in ("joint", "joint_extra_n", "joint_extra_d"):
    a = w09[w09.kind == kind].iloc[0]
    item(f"Experiment 09, Chinchilla runs, {kind}: held-out RMSE {f(a.test_rmse_all)} with all runs and {f(a.test_rmse_drop)} without the 11; train RMSE {f(a.train_rmse_all)} and {f(a.train_rmse_drop)}; "
         f"AIC {f(a.aic_all,3)} and {f(a.aic_drop,3)}; BIC {f(a.bic_all,3)} and {f(a.bic_drop,3)}; collapsed to a single term: {a.collapsed_to_single_all} and {a.collapsed_to_single_drop}.")
p3 = P["3"]
item(f"Prediction 3 (experiment 03): quadratic R-squared of the joint residuals against log D {f(p3['quad_r2_logD_all'])} with all runs and {f(p3['quad_r2_logD_drop'])} without the 11; "
     f"Spearman correlation of the residuals with log N {f(w03.loc['joint','spearman_r_all'])} and {f(p3['spearman_logN_drop'])}, with log D {f(w03.loc['joint_vs_logD','spearman_r_all'])} and {f(p3['spearman_logD_drop'])}; "
     f"preregistered thresholds: quadratic R-squared below 0.10 and absolute Spearman below 0.2; {'PASSED' if p3['passed'] else 'FAILED'}.")
item(f"Experiment 03 residual RMSE of the joint fit {f(p3['residual_rmse_all'])} with all runs and {f(p3['residual_rmse_drop'])} without the 11; residuals above 0.2 in absolute value: "
     f"{p3['n_abs_gt_0.2_all']} and {p3['n_abs_gt_0.2_drop']}.")
a, b = w04.loc["all245"], w04.loc["drop_DN_lt_1"]
p4 = P["4"]
item(f"Prediction 4 (experiment 04 best joint fit): token exponent gamma {f(p4['gamma_all'])} with all runs and {f(p4['gamma_drop'])} without the 11, a fall of {f(p4['fall'])}; published value 0.36, "
     f"so the fit without the 11 sits {abs(p4['distance_to_published_drop']):.4f} {'below' if p4['distance_to_published_drop']<0 else 'above'} it; preregistered: a fall of at least 0.03 toward 0.36; {'PASSED' if p4['passed'] else 'FAILED'}.")
item(f"Experiment 04 best joint fit, all runs: E {f(a.E)}, A {f(a.A,2)}, alpha {f(a.alpha)}, G {f(a.G,1)}, gamma {f(a.gamma)}, Huber objective {f(a.objective)}; without the 11: E {f(b.E)}, A {f(b.A,2)}, "
     f"alpha {f(b.alpha)}, G {f(b.G,1)}, gamma {f(b.gamma)}, objective {f(b.objective)}.")
item(f"Experiment 04, objective at the published point (E 1.82, A 482, alpha 0.34, G 2085, gamma 0.36): {f(a.published_point_objective)} on all runs and {f(b.published_point_objective)} without the 11.")
item(f"Experiment 04 bootstrap of the joint fit (30 row-resamples), token exponent gamma: median {f(a.gamma_boot_median,3)}, 10th to 90th percentile {f(a.gamma_boot_p10,3)} to {f(a.gamma_boot_p90,3)} with all runs; "
     f"median {f(b.gamma_boot_median,3)}, {f(b.gamma_boot_p10,3)} to {f(b.gamma_boot_p90,3)} without the 11.")
sa, sb = hd[hd.arm == "all245"].gamma, hd[hd.arm == "drop_DN_lt_1"].gamma
item(f"Experiment 04 Huber-delta sweep (delta 1e-4 to 10), token exponent gamma ranges {f(sa.min(),3)} to {f(sa.max(),3)} (range {f(sa.max()-sa.min(),3)}) with all runs and {f(sb.min(),3)} to {f(sb.max(),3)} "
     f"(range {f(sb.max()-sb.min(),3)}) without the 11; multistart fits within 1% of the best objective: {int(a.multistart_near_best)} of {int(a.multistart_n)} and {int(b.multistart_near_best)} of {int(b.multistart_n)}.")
item(f"Control (post hoc), {ctl['n_draws']} random removals of 11 of the 245 runs (seed {ctl['seed']}), token exponent gamma: minimum {f(ctl['random_gamma_min'],3)}, 10th percentile {f(ctl['random_gamma_p10'],3)}, "
     f"median {f(ctl['random_gamma_median'],3)}, 90th percentile {f(ctl['random_gamma_p90'],3)}, maximum {f(ctl['random_gamma_max'],3)}; draws with gamma below 0.40: {ctl['random_draws_gamma_below_0.40']}; "
     f"draws at or below the value without the 11 short runs ({f(ctl['gamma_drop_DN_lt_1'],3)}): {ctl['random_draws_gamma_at_or_below_drop_DN_lt_1']}; failed fits {ctl['random_draws_failed']}.")
item(f"Control (post hoc), Huber objective of the best joint fit: all runs {f(ctl['objective_all245'])}, without the 11 short runs {f(ctl['objective_drop_DN_lt_1'])}; "
     f"random removals minimum {f(ctl['random_objective_min'])}, median {f(ctl['random_objective_median'])}, maximum {f(ctl['random_objective_max'])}.")
s = ctl["structured"]
item("Control (post hoc), structured removals of 11 runs, token exponent gamma and objective: " + "; ".join(
    f"{k.replace('_',' ')}: {f(v['gamma'],3)}, {f(v['objective'])}" for k, v in s.items()) + ".")
item("Random removals by number of the 11 short runs they happened to remove: " + ", ".join(f"{int(float(k))} of the 11 short runs removed in {v} draws" for k, v in ctl["random_draws_removing_k_short_runs"].items()) + ".")
p6 = tr["prediction_6"]
bn = lambda x: f"{x/1e9:.2f} billion"
item(f"Prediction 6 (provenance trace): {p6['short_runs']} runs under one token per parameter; {p6['in_1e19_column']} of them lie in the column near 1e19 FLOP, which has {p6['1e19_column_runs']} runs with FLOPs "
     f"{e(p6['1e19_column_C_range'][0])} to {e(p6['1e19_column_C_range'][1])}, and they are that column's {p6['in_1e19_column']} largest model sizes (N from {bn(p6['1e19_column_N_range_of_short'][0])} to {bn(p6['1e19_column_N_range_of_short'][1])} parameters); "
     f"the other is the N={bn(p6['others'][0]['N'])} parameter run in the column near 6e20 FLOP, with FLOPs {e(p6['others'][0]['column_C_range'][0])} to {e(p6['others'][0]['column_C_range'][1])}, rank {p6['others'][0]['N_rank_in_column']} by N; "
     f"{'PASSED' if p6['passed'] else 'FAILED'}.")
item(f"The five lowest tokens-per-parameter ratios are {', '.join(str(x) for x in tr['five_lowest_ratios'])}, all in the column near 1e19 FLOP: {tr['five_lowest_all_in_1e19_column']}; their losses are "
     f"{', '.join(str(x) for x in tr['five_lowest_loss'])} and the largest ratio of a loss to the median loss of runs within 0.1 dex of its FLOPs is {tr['max_loss_over_neighbour_median_of_five']:.3f}.")
item(f"In all {tr['n_rows']} rows, FLOPs equal 6 N D to a relative deviation of at most {e(tr['flops_equals_6ND_max_rel_dev'])}; tokens are therefore derived from FLOPs and N. "
     f"Tokens range from {e(tr['D_min'])} to {e(tr['D_max'])}; {tr['n_D_below_5e9']} rows have fewer than 5e9 tokens, {tr['n_short_D_below_5e9']} of them among the 11 short runs.")
item(f"Sorting log10 FLOPs and splitting at gaps above {tr['gap_dex']} dex gives {tr['n_columns']} columns; {tr['n_columns_with_8_or_more_runs']} columns hold 8 or more runs and together contain {tr['runs_in_those_columns']} of the {tr['n_rows']} rows.")
dr = pd.read_csv(R / "control_dose_response.csv").set_index("k_removed")
item("Dose-response (post hoc), joint fit after removing the k runs with the lowest tokens-per-parameter ratio, token exponent gamma: " + ", ".join(
    f"k={k}: {dr.loc[k,'gamma']:.4f}" for k in (0, 1, 2, 3, 4, 5, 6, 8, 11, 15)) + "; the sequence falls at every step from k=0 to k=5: "
    + str(all(dr.loc[k, "gamma"] < dr.loc[k - 1, "gamma"] for k in range(1, 6))) + ".")
r5, r11 = dr.loc[5], dr.loc[11]
item(f"Dose-response (post hoc), k=5 (the 240 runs left after the five lowest ratios, ratio at most {dr.loc[5,'largest_removed_ratio']:.4f}): E {f(r5.E,4)}, A {f(r5.A,1)}, alpha {f(r5.alpha,4)}, G {f(r5.G,1)}, "
     f"gamma {f(r5.gamma,4)}, Huber objective {f(r5.objective)}; k=11 (234 runs): E {f(r11.E,4)}, alpha {f(r11.alpha,4)}, G {f(r11.G,1)}, gamma {f(r11.gamma,4)}, objective {f(r11.objective)}; all runs: objective {f(dr.loc[0,'objective'])}.")
print("\n".join(f"{i}. {t}" for i, t in enumerate(items, 1)))
