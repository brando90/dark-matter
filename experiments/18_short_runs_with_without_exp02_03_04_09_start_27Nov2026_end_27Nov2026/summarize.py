#!/usr/bin/env python3
"""With-and-without tables for experiments 02, 03, 04 and 09, the harness check, the arm-invariance check and the
evaluation of preregistered predictions 1 to 5. Reads only the CSV files in outputs/ and the committed outputs of the
original experiments; writes results/*.csv and results/summary_numbers.json.

    python3 summarize.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
RES = HERE / "results"
RES.mkdir(exist_ok=True)
OUT = HERE / "outputs"
COMMITTED = {"02": EXP / "02_compute_and_joint_extrapolation" / "outputs", "03": EXP / "03_residual_structure" / "outputs",
             "04": EXP / "04_fit_stability" / "outputs", "09": EXP / "09_alternative_forms" / "outputs"}
ARM = {e: {a: OUT / e / a for a in ("all245", "drop_DN_lt_1")} for e in COMMITTED}
S = {}  # everything that results.md may cite


def rd(path):
    return pd.read_csv(path)


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.abs(a), 1e-12)


# ---------------------------------------------------------------- experiment 02
com02, a02, d02 = rd(COMMITTED["02"] / "metrics.csv"), rd(ARM["02"]["all245"] / "metrics.csv"), rd(ARM["02"]["drop_DN_lt_1"] / "metrics.csv")
key02 = ["dataset", "protocol", "model"]
ch = lambda df: df[df.dataset == "chinchilla_runs"]
w02 = ch(a02).merge(ch(d02), on=key02, suffixes=("_all", "_drop"))
w02["rmse_drop_pct_change"] = 100 * (w02.test_rmse_drop - w02.test_rmse_all) / w02.test_rmse_all
w02["macro_drop_pct_change"] = 100 * (w02.test_macro_rmse_drop - w02.test_macro_rmse_all) / w02.test_macro_rmse_all

# Post hoc decomposition for 02: evaluate the model trained on all 245 runs only on the test rows that survive in the
# other arm, so the change in test-set composition is separated from the change in training data.
pa = rd(ARM["02"]["all245"] / "predictions.csv"); pd_ = rd(ARM["02"]["drop_DN_lt_1"] / "predictions.csv")
dec = []
for (proto, model), g in pa[(pa.dataset == "chinchilla_runs") & (pa.split == "test")].groupby(["protocol", "model"]):
    kept = g[(g.D / g.N) >= 1.0]
    g2 = pd_[(pd_.dataset == "chinchilla_runs") & (pd_.split == "test") & (pd_.protocol == proto) & (pd_.model == model)]
    same_rows = bool(len(kept) == len(g2) and np.allclose(np.sort(kept.loss.to_numpy()), np.sort(g2.loss.to_numpy()), rtol=0, atol=0))
    rmse = lambda x: float(np.sqrt(np.mean((x.loss - x.prediction) ** 2)))
    dec.append({"protocol": proto, "model": model, "n_test_all": len(g), "n_test_kept": len(kept), "same_test_rows_as_drop_arm": same_rows,
                "rmse_all245_model_on_all_test": rmse(g), "rmse_all245_model_on_kept_test": rmse(kept), "rmse_drop_model_on_kept_test": rmse(g2)})
dec = pd.DataFrame(dec)
w02 = w02.merge(dec, on=["protocol", "model"])
w02.to_csv(RES / "with_without_02.csv", index=False)

# ---------------------------------------------------------------- experiment 09
com09, a09, d09 = rd(COMMITTED["09"] / "metrics.csv"), rd(ARM["09"]["all245"] / "metrics.csv"), rd(ARM["09"]["drop_DN_lt_1"] / "metrics.csv")
key09 = ["dataset", "family", "kind"]
w09 = a09[a09.dataset == "chinchilla_runs"].merge(d09[d09.dataset == "chinchilla_runs"], on=key09, suffixes=("_all", "_drop"))
w09.to_csv(RES / "with_without_09.csv", index=False)

# ---------------------------------------------------------------- experiment 03
com03 = rd(COMMITTED["03"] / "residual_stats.csv"); a03 = rd(ARM["03"]["all245"] / "residual_stats.csv"); d03 = rd(ARM["03"]["drop_DN_lt_1"] / "residual_stats.csv")
c03 = lambda df: df[df.curve == "chinchilla_all_runs"].set_index("kind")
w03 = c03(a03).join(c03(d03), lsuffix="_all", rsuffix="_drop")
w03.to_csv(RES / "with_without_03.csv")
sens_a, sens_d = rd(ARM["03"]["all245"] / "joint_residual_sensitivity.csv"), rd(ARM["03"]["drop_DN_lt_1"] / "joint_residual_sensitivity.csv")

# ---------------------------------------------------------------- experiment 04
def f04(path):
    pc, hd, bs = rd(path / "published_comparison.csv"), rd(path / "huber_delta_sweep.csv"), rd(path / "bootstrap_summary.csv")
    ms = rd(path / "joint_multistart.csv")
    return pc, hd, bs[bs.model == "joint_chinchilla_runs"].set_index("parameter"), ms
pc_c, hd_c, bs_c, ms_c = f04(COMMITTED["04"]); pc_a, hd_a, bs_a, ms_a = f04(ARM["04"]["all245"]); pc_d, hd_d, bs_d, ms_d = f04(ARM["04"]["drop_DN_lt_1"])
rows04 = []
for arm, (pc, hd, bs, ms) in {"all245": (pc_a, hd_a, bs_a, ms_a), "drop_DN_lt_1": (pc_d, hd_d, bs_d, ms_d)}.items():
    best = pc[pc.which == "best_of_24_starts"].iloc[0]; pub = pc[pc.which == "published_point_not_optimized"].iloc[0]
    rows04.append({"arm": arm, "objective": best.objective, "E": best.E, "A": best.A, "alpha": best.alpha, "G": best.G, "gamma": best.gamma,
                   "published_point_objective": pub.objective, "gamma_boot_median": bs.loc["gamma", "median"], "gamma_boot_p10": bs.loc["gamma", "p10"],
                   "gamma_boot_p90": bs.loc["gamma", "p90"], "alpha_boot_median": bs.loc["alpha", "median"], "E_boot_median": bs.loc["E", "median"],
                   "gamma_delta_min": hd.gamma.min(), "gamma_delta_max": hd.gamma.max(), "gamma_delta_range": hd.gamma.max() - hd.gamma.min(),
                   "multistart_n": len(ms), "multistart_near_best": int(ms.near_best.sum())})
w04 = pd.DataFrame(rows04)
w04.to_csv(RES / "with_without_04.csv", index=False)
hd_a.assign(arm="all245").pipe(lambda x: pd.concat([x, hd_d.assign(arm="drop_DN_lt_1")])).to_csv(RES / "huber_delta_sweep_both_arms.csv", index=False)

# ---------------------------------------------------------------- harness check: arm all245 against the committed outputs
def maxrel(c, a, key, cols):
    m = c.merge(a, on=key, suffixes=("_c", "_a"))
    worst, where = 0.0, ""
    for col in cols:
        r = rel(m[col + "_c"], m[col + "_a"])
        r = np.where(np.isfinite(r), r, 0.0)
        if r.max() > worst:
            worst, where = float(r.max()), f"{col} @ " + "/".join(str(m.iloc[int(r.argmax())][k]) for k in key)
    return worst, where, len(m)
h = []
w, where, n = maxrel(ch(com02), ch(a02), key02, ["test_rmse", "test_macro_rmse", "train_rmse"]); h.append(("02", "Chinchilla-run rows, RMSEs", w, where, n))
w, where, n = maxrel(com02, a02, key02, ["test_rmse", "test_macro_rmse", "train_rmse"]); h.append(("02", "all rows, RMSEs", w, where, n))
w, where, n = maxrel(com09[com09.dataset == "chinchilla_runs"], a09[a09.dataset == "chinchilla_runs"], key09, ["train_rmse", "test_rmse", "aic", "bic"]); h.append(("09", "Chinchilla-run rows, RMSEs and AIC/BIC", w, where, n))
w, where, n = maxrel(com09, a09, key09, ["train_rmse", "test_rmse", "aic", "bic"]); h.append(("09", "all rows, RMSEs and AIC/BIC", w, where, n))
cc, aa = com03[com03.curve == "chinchilla_all_runs"], a03[a03.curve == "chinchilla_all_runs"]
w, where, n = maxrel(cc, aa, ["kind"], ["quad_r2", "residual_rmse", "spearman_r"]); h.append(("03", "Chinchilla joint rows, quad R2, RMSE, Spearman", w, where, n))
w, where, n = maxrel(pc_c, pc_a, ["which"], ["objective", "E", "A", "alpha", "G", "gamma"]); h.append(("04", "best-fit rows: objective and parameters", w, where, n))
w, where, n = maxrel(hd_c, hd_a, ["delta"], ["objective", "gamma", "alpha"]); h.append(("04", "Huber-delta sweep: objective, gamma, alpha", w, where, n))
harness = pd.DataFrame(h, columns=["experiment", "what", "max_rel_diff", "worst_at", "rows"])
harness.to_csv(RES / "harness_check.csv", index=False)

# ---------------------------------------------------------------- arm invariance: rows that cannot depend on the 11 runs
inv = []
unaff02 = lambda df: df[df.dataset != "chinchilla_runs"]
m = unaff02(a02).merge(unaff02(d02), on=key02, suffixes=("_a", "_d"))
inv.append(("02", "gpt3_val and pythia_lambada rows", len(m), float(max(np.nanmax(np.abs(m[c + "_a"] - m[c + "_d"])) for c in ["test_rmse", "test_macro_rmse", "train_rmse", "objective"]))))
m = a09[a09.dataset != "chinchilla_runs"].merge(d09[d09.dataset != "chinchilla_runs"], on=key09, suffixes=("_a", "_d"))
inv.append(("09", "all rows other than chinchilla_runs", len(m), float(max(np.nanmax(np.abs(m[c + "_a"] - m[c + "_d"])) for c in ["test_rmse", "train_rmse", "aic", "bic"]))))
m = a03[a03.curve != "chinchilla_all_runs"].merge(d03[d03.curve != "chinchilla_all_runs"], on=["curve", "kind"], suffixes=("_a", "_d"))
inv.append(("03", "all curves other than chinchilla_all_runs", len(m), float(max(np.nanmax(np.abs(m[c + "_a"] - m[c + "_d"])) for c in ["quad_r2", "residual_rmse", "spearman_r"]))))
# Experiment 04's two-term bootstrap on the matched ladder is NOT in this table: it shares one random generator with the joint
# bootstrap (np.random.default_rng(0)), which draws 245 or 234 indices per replicate, so its resamples differ between arms by
# construction (first resample indices [20, 13, 23, 7, 31, 6] against [4, 28, 6, 5, 26, 28]). It is not compared between arms.
invariance = pd.DataFrame(inv, columns=["experiment", "rows", "n", "max_abs_diff_between_arms"])
invariance.to_csv(RES / "arm_invariance.csv", index=False)

# ---------------------------------------------------------------- preregistered predictions 1 to 5
g = lambda df, **kw: df[np.logical_and.reduce([df[k] == v for k, v in kw.items()])].iloc[0]
P = {}
# 1: joint model's size-holdout pooled test RMSE falls by at least half in both 02 and 09
j02a, j02d = g(a02, dataset="chinchilla_runs", protocol="size", model="joint"), g(d02, dataset="chinchilla_runs", protocol="size", model="joint")
j09a, j09d = g(a09, dataset="chinchilla_runs", kind="joint"), g(d09, dataset="chinchilla_runs", kind="joint")
red02, red09 = 1 - j02d.test_rmse / j02a.test_rmse, 1 - j09d.test_rmse / j09a.test_rmse
P["1"] = {"rmse_02_all": j02a.test_rmse, "rmse_02_drop": j02d.test_rmse, "reduction_02": red02, "rmse_09_all": j09a.test_rmse,
          "rmse_09_drop": j09d.test_rmse, "reduction_09": red09, "macro_02_all": j02a.test_macro_rmse, "macro_02_drop": j02d.test_macro_rmse,
          "macro_reduction_02": 1 - j02d.test_macro_rmse / j02a.test_macro_rmse, "passed": bool(red02 >= 0.5 and red09 >= 0.5)}
# 2: no extra-term model beats its base by more than 2 percent
cases = []
for arm, df in (("all245", a02), ("drop_DN_lt_1", d02)):
    for proto in ("size", "compute"):
        for base, extra in (("joint", "joint_extra_n"), ("compute_one", "compute_two")):
            b, e = g(df, dataset="chinchilla_runs", protocol=proto, model=base), g(df, dataset="chinchilla_runs", protocol=proto, model=extra)
            cases.append({"exp": "02", "arm": arm, "protocol": proto, "base": base, "extra": extra, "gain_pooled": 1 - e.test_rmse / b.test_rmse,
                          "gain_macro": 1 - e.test_macro_rmse / b.test_macro_rmse})
for arm, df in (("all245", a09), ("drop_DN_lt_1", d09)):
    b = g(df, dataset="chinchilla_runs", kind="joint")
    for extra in ("joint_extra_n", "joint_extra_d"):
        e = g(df, dataset="chinchilla_runs", kind=extra)
        cases.append({"exp": "09", "arm": arm, "protocol": "size", "base": "joint", "extra": extra, "gain_pooled": 1 - e.test_rmse / b.test_rmse, "gain_macro": np.nan})
cases = pd.DataFrame(cases)
cases.to_csv(RES / "extra_term_gains.csv", index=False)
gmax = float(np.nanmax(cases[["gain_pooled", "gain_macro"]].to_numpy()))
P["2"] = {"max_gain_any_case": gmax, "n_cases": int(len(cases)), "n_cases_with_positive_gain": int((cases[["gain_pooled", "gain_macro"]].max(axis=1) > 0).sum()),
          "worst_loss_vs_base_size_02_drop": float(-cases[(cases.exp == "02") & (cases.arm == "drop_DN_lt_1") & (cases.protocol == "size") & (cases.extra == "joint_extra_n")].gain_pooled.iloc[0]),
          "passed": bool(gmax <= 0.02)}
# 3: experiment 03, arm without the 11 runs
P["3"] = {"quad_r2_logD_all": float(w03.loc["joint_vs_logD", "quad_r2_all"]), "quad_r2_logD_drop": float(w03.loc["joint_vs_logD", "quad_r2_drop"]),
          "spearman_logN_drop": float(w03.loc["joint", "spearman_r_drop"]), "spearman_logD_drop": float(w03.loc["joint_vs_logD", "spearman_r_drop"]),
          "residual_rmse_all": float(w03.loc["joint", "residual_rmse_all"]), "residual_rmse_drop": float(w03.loc["joint", "residual_rmse_drop"]),
          "n_abs_gt_0.2_all": int(sens_a.iloc[0]["n_abs_gt_0.2"]), "n_abs_gt_0.2_drop": int(sens_d.iloc[0]["n_abs_gt_0.2"]),
          "passed": bool(w03.loc["joint_vs_logD", "quad_r2_drop"] < 0.10 and abs(w03.loc["joint", "spearman_r_drop"]) < 0.2 and abs(w03.loc["joint_vs_logD", "spearman_r_drop"]) < 0.2)}
# 4: experiment 04 token exponent
ga, gd = float(w04.loc[w04.arm == "all245", "gamma"].iloc[0]), float(w04.loc[w04.arm == "drop_DN_lt_1", "gamma"].iloc[0])
P["4"] = {"gamma_all": ga, "gamma_drop": gd, "fall": ga - gd, "published_gamma": 0.36, "moves_toward_published": bool(abs(gd - 0.36) < abs(ga - 0.36)),
          "distance_to_published_drop": gd - 0.36, "passed": bool(ga - gd >= 0.03 and abs(gd - 0.36) < abs(ga - 0.36))}
# 5: harness
# experiment 04 is Chinchilla-only, so its rows count as Chinchilla rows even though their label does not say so
chin_rows = harness[harness.what.str.contains("Chinchilla") | (harness.experiment == "04")]
P["5"] = {"max_rel_diff_chinchilla_rows": float(chin_rows.max_rel_diff.max()), "max_rel_diff_all_rows": float(harness.max_rel_diff.max()), "threshold": 1e-6,
          "passed": bool(harness.max_rel_diff.max() <= 1e-6)}
S["predictions"] = P
S["harness"] = harness.to_dict("records")
S["invariance"] = invariance.to_dict("records")
S["w04"] = w04.to_dict("records")
S["decomposition"] = dec.to_dict("records")
S["n_runs"] = {"all245": 245, "drop_DN_lt_1": 234}
(RES / "summary_numbers.json").write_text(json.dumps(S, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(json.dumps(P, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
print(harness.to_string()); print(invariance.to_string()); print(dec.to_string())
