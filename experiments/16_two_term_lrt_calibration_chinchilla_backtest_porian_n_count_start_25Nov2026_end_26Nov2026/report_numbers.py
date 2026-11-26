"""
Prints every number quoted in REPORT.md from the saved files in results/ (no refitting).
Each line is numbered; REPORT.md cites these numbers by item. Almost everything is computed from the CSV, JSON and JSONL
files; the smoke-test and validation values are parsed from the two saved text logs, and one value in item 42 is quoted
from the repo's own RESULTS.md and says so. Run:

    python report_numbers.py            # prints the list
    python report_numbers.py > results/report_numbers.txt
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent / "results"
SIG = ("0.003", "0.01")

_e1_all = json.loads((R / "e1_summary.json").read_text())
e1 = _e1_all["settings"]
smoke = (R / "repo_smoke_test.txt").read_text(encoding="utf-8")
valid = (R / "validate_scaling_ext.txt").read_text(encoding="utf-8")
reps = [json.loads(l) for l in (R / "e1_reps.jsonl").read_text().splitlines()]
e2 = json.loads((R / "e2_summary.json").read_text())
e3 = json.loads((R / "e3_summary.json").read_text())
chin = pd.read_csv(R / "chinchilla_245.csv")
fits2 = pd.read_csv(R / "e2_full_fits.csv", keep_default_na=False)
prof = pd.read_csv(R / "e2_profile.csv", keep_default_na=False)
a4 = pd.read_csv(R / "e4_n_proxy.csv", keep_default_na=False)
bt4 = pd.read_csv(R / "e4_backtest.csv", keep_default_na=False)
bend = pd.read_csv(R / "e4_small_n_bend.csv", keep_default_na=False)
bend2 = pd.read_csv(R / "e4_small_n_bend_by_proxy.csv", keep_default_na=False)

lines: list[str] = []
head: list[tuple[int, str]] = []


def section(title):
    lines.append(f"\n## {title}")


def item(text):
    lines.append(f"{sum(1 for l in lines if l[:1].isdigit()) + 1}. {text}")


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def both(f, fmt="{:.3g}"):
    return " | ".join(fmt.format(f(s)) for s in SIG)


def parse_joint(desc):
    """Constants of 'E=.. + A/N^a + B/D^b' as (E, A, a, B, b)."""
    E = float(re.search(r"E=([0-9.e+-]+)", desc).group(1))
    A, a = re.search(r"\+ ([0-9.e+-]+)/N\^([0-9.e+-]+)", desc).groups()
    B, b = re.search(r"\+ ([0-9.e+-]+)/D\^([0-9.e+-]+)", desc).groups()
    return E, float(A), float(a), float(B), float(b)


def grab(pattern, text, flags=0):
    m = re.search(pattern, text, flags)
    assert m, f"pattern not found: {pattern}"
    return m.group(1)


# ------------------------------------------------------------------ E1
section("E1: Is the repo's two-term likelihood-ratio test calibrated? (noise 0.3% | 1%)")
_sh = lambda N: 0.5 * (N / 1e7) ** -0.15 / (1.7 + 0.7 * (N / 1e7) ** -0.5 + 0.5 * (N / 1e7) ** -0.15)
item("Design: 150 one-term and 100 two-term ladders per noise level, 10 model sizes from 1e7 to 1.2e10. "
     "One-term truth L = 1.7 + 120/N^0.30. Two-term truth L = 1.7 + 0.7 (N/1e7)^-0.5 + 0.5 (N/1e7)^-0.15, so alpha 0.5, beta 0.15, beta/alpha 0.3; "
     f"the shallow second term is {100 * _sh(1.2e10):.0f}% of the loss at the largest size and {100 * _sh(1e7):.0f}% at the smallest. "
     f"But a one-term law E + A/N^a fitted by least squares to the noise-free two-term ladder (E {_e1_all['noiseless_two_term_truth_one_term_fit']['E']:.3f}, a {_e1_all['noiseless_two_term_truth_one_term_fit']['alpha']:.3f}) "
     f"misses it by at most {100 * _e1_all['noiseless_two_term_truth_one_term_fit']['max_abs_log_resid']:.2f}% (root mean square {100 * _e1_all['noiseless_two_term_truth_one_term_fit']['rms_log_resid']:.2f}%), "
     "which is the largest signal a test or a fit can use at zero noise.")
mx = {}
for s in SIG:
    for t in ("H0", "H1"):
        rr = [r for r in reps if r["truth"] == t and str(r["sigma"]) == s]
        mx[(s, t)] = np.array([2 * (r["h_obj1"] - r["h_obj2"]) for r in rr])
item("Repo statistic 2 x (Huber loss gain), delta 0.001, against the 5% chi-square(2) cutoff of 5.99: largest value over all 250 ladders "
     + " | ".join(f"{max(mx[(s, 'H0')].max(), mx[(s, 'H1')].max()):.2e}" for s in SIG)
     + f"; nominal false-positive rate {both(lambda s: 100 * e1[f'sigma={s}']['repo_nominal_false_positive_rate'], '{:.0f}%')} and nominal power "
     f"{both(lambda s: 100 * e1[f'sigma={s}']['repo_nominal_power'], '{:.0f}%')}; cutoff divided by largest value "
     + " | ".join(f"{5.9915 / max(mx[(s, 'H0')].max(), mx[(s, 'H1')].max()):,.0f}" for s in SIG) + ".")
item("Share of ladders where the two-term fit gains exactly nothing (statistic below 1e-12): one-term truth "
     + " | ".join(pct((mx[(s, 'H0')] <= 1e-12).mean(), 0) for s in SIG) + "; two-term truth "
     + " | ".join(pct((mx[(s, 'H1')] <= 1e-12).mean(), 0) for s in SIG) + ".")
item("Repo statistic ranking quality (AUC) " + " | ".join(f"{e1[f'sigma={s}']['repo_auc']:.3f}" for s in SIG)
     + "; power at the empirical 95th percentile of its own null distribution "
     + " | ".join(pct(e1[f'sigma={s}']['repo_corrected_power'], 0) for s in SIG) + ".")
item("Gaussian LRT K log(RSS1/RSS2), K = 10: nominal false-positive rate "
     + " | ".join(pct(e1[f'sigma={s}']['gauss_nominal_false_positive_rate']) for s in SIG)
     + " (" + " and ".join(f"{e1[f'sigma={s}']['gauss_nominal_false_positive_count'][0]} of {e1[f'sigma={s}']['gauss_nominal_false_positive_count'][1]}" for s in SIG)
     + " one-term ladders; 95% Wilson intervals " + " and ".join(f"{100 * e1[f'sigma={s}']['gauss_nominal_false_positive_ci95'][0]:.1f} to {100 * e1[f'sigma={s}']['gauss_nominal_false_positive_ci95'][1]:.1f}%" for s in SIG) + ")"
     + "; share of one-term ladders where the statistic is exactly zero " + " | ".join(pct(e1[f'sigma={s}']['gauss_stat_H0_zero_share'], 0) for s in SIG)
     + "; empirical null 95th percentile " + " | ".join(f"{e1[f'sigma={s}']['gauss_stat_H0_q95']:.2f}" for s in SIG) + " against the chi-square cutoff 5.99"
     + "; the same statistic on the seven-size training ladders (K = 7): false-positive rate " + " | ".join(pct(e1[f'sigma={s}']['train7_gauss_nominal_false_positive_rate']) for s in SIG)
     + " (" + " and ".join(f"{e1[f'sigma={s}']['train7_gauss_nominal_false_positive_count'][0]} of {e1[f'sigma={s}']['train7_gauss_nominal_false_positive_count'][1]}" for s in SIG)
     + "; intervals " + " and ".join(f"{100 * e1[f'sigma={s}']['train7_gauss_nominal_false_positive_ci95'][0]:.1f} to {100 * e1[f'sigma={s}']['train7_gauss_nominal_false_positive_ci95'][1]:.1f}%" for s in SIG) + ")"
     + "; nominal power " + " | ".join(pct(e1[f'sigma={s}']['gauss_nominal_power'], 0) for s in SIG)
     + "; size-corrected power " + " | ".join(pct(e1[f'sigma={s}']['gauss_corrected_power'], 0) for s in SIG)
     + "; AUC " + " | ".join(f"{e1[f'sigma={s}']['gauss_auc']:.3f}" for s in SIG) + ".")
item("Two-term fit lands on a boundary (a coefficient at its cap, E below 0.1% of the smallest loss, or the shallow exponent below 0.02): one-term truth "
     + " | ".join(pct(e1[f'sigma={s}']['two_term_boundary_H0']['any_boundary'], 0) for s in SIG) + "; two-term truth "
     + " | ".join(pct(e1[f'sigma={s}']['two_term_boundary_H1']['any_boundary'], 0) for s in SIG) + ".")
item("Parameter recovery on two-term ladders (truth alpha 0.5, beta 0.15): median fitted beta "
     + " | ".join(f"{e1[f'sigma={s}']['H1_beta_hat_median_IQR'][0]:.2f}" for s in SIG) + " with middle half "
     + " | ".join(f"{e1[f'sigma={s}']['H1_beta_hat_median_IQR'][1]:.3f} to {e1[f'sigma={s}']['H1_beta_hat_median_IQR'][2]:.2f}" for s in SIG)
     + "; median fitted alpha " + " | ".join(f"{e1[f'sigma={s}']['H1_alpha_hat_median_IQR'][0]:.2f}" for s in SIG) + " with middle half "
     + " | ".join(f"{e1[f'sigma={s}']['H1_alpha_hat_median_IQR'][1]:.2f} to {e1[f'sigma={s}']['H1_alpha_hat_median_IQR'][2]:.2f}" for s in SIG)
     + "; share of ladders with fitted beta within 0.05 of truth "
     + " | ".join(pct(e1[f'sigma={s}']['H1_frac_beta_within_0.05_of_truth'], 0) for s in SIG) + ".")
for tag, tname in (("backtest_H1", "two-term truth"), ("backtest_H0", "one-term truth")):
    item(f"Extrapolation backtest (fit the 7 smallest sizes, predict the 3 largest), {tname}, median error one-term vs two-term: "
         + " | ".join(f"{e1[f'sigma={s}'][tag]['always_1term']['median']:.5f} vs {e1[f'sigma={s}'][tag]['always_2term']['median']:.5f}" for s in SIG)
         + "; mean error " + " | ".join(f"{e1[f'sigma={s}'][tag]['always_1term']['mean']:.5f} vs {e1[f'sigma={s}'][tag]['always_2term']['mean']:.5f}" for s in SIG)
         + "; two-term mean error relative to one-term "
         + " | ".join(f"{100 * (e1[f'sigma={s}'][tag]['always_2term']['mean'] / e1[f'sigma={s}'][tag]['always_1term']['mean'] - 1):+.0f}%" for s in SIG)
         + "; two-term median error relative to one-term "
         + " | ".join(f"{100 * (e1[f'sigma={s}'][tag]['always_2term']['median'] / e1[f'sigma={s}'][tag]['always_1term']['median'] - 1):+.0f}%" for s in SIG)
         + " (95% bootstrap intervals over ladders "
         + " and ".join(f"{100 * e1[f'sigma={s}'][tag]['median_error_ratio_2term_vs_1term']['ci95'][0]:+.1f}% to {100 * e1[f'sigma={s}'][tag]['median_error_ratio_2term_vs_1term']['ci95'][1]:+.1f}%" for s in SIG) + ")"
         + "; ladder by ladder, two-term wins / ties / losses against one-term (|difference| within 1e-6 is a tie) "
         + " | ".join("{} / {} / {}".format(*e1[f'sigma={s}'][tag]['two_term_wins_ties_losses']) for s in SIG)
         + " (two-sided sign test on wins against losses, p = " + " and ".join(f"{e1[f'sigma={s}'][tag]['sign_test_p_wins_vs_losses']:.2g}" for s in SIG) + ").")
item("Gate (use the two-term model only if the training-set LRT clears its empirical null 95th percentile) fires on one-term ladders "
     + " | ".join(pct(e1[f'sigma={s}']['backtest_H0']['gate_fires']) for s in SIG) + " and on two-term ladders "
     + " | ".join(pct(e1[f'sigma={s}']['backtest_H1']['gate_fires'], 0) for s in SIG)
     + "; two-term truth, gated median error " + " | ".join(f"{e1[f'sigma={s}']['backtest_H1']['gated']['median']:.5f}" for s in SIG)
     + " against always-one-term " + " | ".join(f"{e1[f'sigma={s}']['backtest_H1']['always_1term']['median']:.5f}" for s in SIG)
     + " and always-two-term " + " | ".join(f"{e1[f'sigma={s}']['backtest_H1']['always_2term']['median']:.5f}" for s in SIG)
     + "; relative to always-one-term, the gated median error is " + " | ".join(f"{100 * (e1[f'sigma={s}']['backtest_H1']['gated']['median'] / e1[f'sigma={s}']['backtest_H1']['always_1term']['median'] - 1):+.1f}%" for s in SIG)
     + " and the always-two-term median error is " + " | ".join(f"{100 * (e1[f'sigma={s}']['backtest_H1']['always_2term']['median'] / e1[f'sigma={s}']['backtest_H1']['always_1term']['median'] - 1):+.1f}%" for s in SIG)
     + ". The gate threshold is the 95th percentile of the Gaussian statistic on simulated one-term ladders at the true noise level (known only in a simulation).")
_two = smoke.split("(2) FITTING TWO-TERM")[1].split("(3)")[0]
_s_stat = grab(r"Test statistic:\s*(-?[0-9.]+)", smoke)
_s_p = float(grab(r"p-value:\s*([0-9.e+-]+)", smoke))
_s_sel = grab(r"Selected (\d+) points for N-ladder", smoke)
_s_n = grab(r"Generated (\d+) synthetic data points", smoke)
_s_a = grab(r"α\s*=\s*([0-9.]+)", _two)
_s_b = grab(r"β\s*=\s*([0-9.]+)", _two)
item("Repo smoke test (python py_src/scaling_laws_fit.py, parsed from results/repo_smoke_test.txt): "
     f"LRT statistic {_s_stat}, p-value {_s_p:.3f}; the two-term step reports 'Selected {_s_sel} points for N-ladder' "
     f"and fits all {_s_n} joint (N, D) points, returning alpha {_s_a} and beta {_s_b} "
     "although the generating law has no second N term. Both fits use the same points, so a statistic at or below zero means the one-term baseline scored at least as well as the two-term fit.")

# ------------------------------------------------------------------ E2
section("E2: Chinchilla (MassiveText) points: joint law, extra terms, identifiability")
item(f"Data: {len(chin)} points, {chin.N.nunique()} distinct N from {chin.N.min():.3g} to {chin.N.max():.4g}, D from {chin.D.min():.3g} to {chin.D.max():.4g}, "
     f"loss from {chin.L.min():.2f} to {chin.L.max():.2f}.")
fj = fits2[(fits2.exp_max.astype(float) == 1.5) & (fits2.model == "joint")].iloc[0]
fnd = fits2[(fits2.exp_max.astype(float) == 1.5) & (fits2.model == "+N dark")].iloc[0]
fdd = fits2[(fits2.exp_max.astype(float) == 1.5) & (fits2.model == "+D dark")].iloc[0]
_E, _A, _a, _B, _b = parse_joint(fj.desc)
_t4 = valid.split("TEST 4")[1].split("TEST 5")[0]
_seed_fits = [d.strip() for d in re.findall(r"seed \d+: huber=[0-9.]+\s+(E=.*)", _t4)]
_seeds = f"identical across {len(_seed_fits)} seeds" if len(set(_seed_fits)) == 1 else f"{len(set(_seed_fits))} different fits across {len(_seed_fits)} seeds"
_t3 = valid.split("TEST 3")[1].split("TEST 4")[0]
_rd, _rh, _rr = re.search(r"repo single start\s+(E=\S+ \+ \S+/N\^\S+ \+ \S+/D\^\S+)\s+huber=([0-9.]+) rss=([0-9.]+)", _t3).groups()
_rE, _rA, _ra, _rB, _rb = parse_joint(_rd)
item(f"Multi-start joint fit ({_seeds}; results/validate_scaling_ext.txt): {fj.desc}; Huber {fj.huber:.6f}; RSS {fj.rss:.4f}; compute-optimal exponent beta/(alpha+beta) {_b / (_a + _b):.4f}. "
     f"Repo single-start fit_joint_powerlaw (same log): {_rd}; exponent {_rb / (_ra + _rb):.4f}, Huber {_rh}, RSS {_rr}.")
item(f"Adding a second N term leaves the objective unchanged ({fnd.huber:.6f}); the fit switches the term off ({fnd.desc.split('+')[-1].strip()}). "
     f"Adding a second D term: Huber {fdd.huber:.6f} ({100 * (1 - fdd.huber / fj.huber):.1f}% lower), RSS {fdd.rss:.4f} ({100 * (1 - fdd.rss / fj.rss):.0f}% lower), fit {fdd.desc}.")
ax = e2["cap=1.5"]["median_test_rmse_by_axis"]
ax4 = e2["cap=4.0"]["median_test_rmse_by_axis"]
for a_, nm in (("N", "to larger models"), ("D", "to more tokens"), ("C", "to more compute")):
    item(f"Held-out median error extrapolating {nm} (four cuts at the 40/50/60/70% quantile, exponents capped at 1.5): joint {ax[a_]['joint']:.4f}, +N term {ax[a_]['+N dark']:.4f}, "
         f"+D term {ax[a_]['+D dark']:.4f}, both {ax[a_]['+both dark']:.4f}, repo single-start {ax[a_]['repo']:.4f}; with the cap at 4.0 the +D term gives {ax4[a_]['+D dark']:.4f}.")
mr = e2["cap=1.5"]["median_test_rmse"]
mr4 = e2["cap=4.0"]["median_test_rmse"]
item(f"Median over all 12 splits (cap 1.5): joint {mr['joint']:.4f}, +N {mr['+N dark']:.4f}, +D {mr['+D dark']:.4f}, both {mr['+both dark']:.4f}, repo {mr['repo']:.4f} "
     f"(cap 4.0: joint {mr4['joint']:.4f}, +N {mr4['+N dark']:.4f}, +D {mr4['+D dark']:.4f}). "
     "Against the joint fit, [wins, ties, losses] over the 12 splits (a difference in held-out RMSE within 1e-6 is a tie; a strict comparison would call a fit that switches the extra term off a win): "
     f"+D {e2['cap=1.5']['wins_ties_losses_vs_joint']['+D dark']} at cap 1.5 and {e2['cap=4.0']['wins_ties_losses_vs_joint']['+D dark']} at cap 4.0; "
     f"+N {e2['cap=1.5']['wins_ties_losses_vs_joint']['+N dark']} and {e2['cap=4.0']['wins_ties_losses_vs_joint']['+N dark']}; "
     f"both {e2['cap=1.5']['wins_ties_losses_vs_joint']['+both dark']} and {e2['cap=4.0']['wins_ties_losses_vs_joint']['+both dark']}. "
     f"The repo single-start fit against the multi-start joint fit (cap 1.5): better by more than 1% in {e2['cap=1.5']['repo_vs_joint']['repo_better_by_over_1pct']} of 12 splits, "
     f"within 1% in {e2['cap=1.5']['repo_vs_joint']['within_1pct']}, worse by more than 1% in {e2['cap=1.5']['repo_vs_joint']['repo_worse_by_over_1pct']}.")
item(f"Boundary-flagged fits among 12 splits (cap 1.5): joint {e2['cap=1.5']['boundary_flag_counts']['joint']}, +N {e2['cap=1.5']['boundary_flag_counts']['+N dark']}, "
     f"+D {e2['cap=1.5']['boundary_flag_counts']['+D dark']}, both {e2['cap=1.5']['boundary_flag_counts']['+both dark']}.")
pb = e2["profile b (data exponent) within 1%"]
pb5 = e2["profile b (data exponent) within 5%"]
pa = e2["profile a (model exponent) within 1%"]
pa5 = e2["profile a (model exponent) within 5%"]
item(f"Profile over the data exponent beta (grid 0.01 apart near the optimum and 16% apart in the wide first grid), objective within 1% of best: beta {pb['fixed_range'][0]:.3f} to {pb['fixed_range'][1]:.3f}, "
     f"compute-optimal exponent {pb['a_opt_range'][0]:.3f} to {pb['a_opt_range'][1]:.3f} ({pb['n_grid_points']} grid points inside); "
     f"within 5%: beta {pb5['fixed_range'][0]:.3f} to {pb5['fixed_range'][1]:.3f}, exponent {pb5['a_opt_range'][0]:.3f} to {pb5['a_opt_range'][1]:.3f} ({pb5['n_grid_points']} grid points). "
     "The band edges are the outermost grid points inside the band, so the true bands are slightly wider.")
item(f"Profile over the model exponent alpha, within 1%: alpha {pa['fixed_range'][0]:.3f} to {pa['fixed_range'][1]:.3f}, exponent {pa['a_opt_range'][0]:.3f} to {pa['a_opt_range'][1]:.3f} ({pa['n_grid_points']} grid points); "
     f"within 5%: alpha {pa5['fixed_range'][0]:.3f} to {pa5['fixed_range'][1]:.3f}, exponent {pa5['a_opt_range'][0]:.3f} to {pa5['a_opt_range'][1]:.3f} ({pa5['n_grid_points']} grid points).")

# ------------------------------------------------------------------ E3
section("E3: Real fixed-data ladders (loss = natural log of LAMBADA perplexity, final checkpoints)")
for r in e3["full_ladder"]:
    item(f"{r['ladder']}: {r['K']} sizes from {min(r['N']):.3g} to {max(r['N']):.3g} parameters; residual scale {r['sigma_hat']:.3f}; Gaussian LRT {r['gauss_lrt_stat']:.3f}; "
         f"bootstrap p-value {r['boot_pvalue']:.3f} ({r['boot_B']} draws, null 95th percentile {r['boot_q95']:.2f}); two-term fit on the boundary: {r['corner_2term']}; "
         f"one-term {r['fit_1term_ls']}; two-term {r['fit_2term_ls']}.")
t = e3["two_term_vs_one_term"]
fc = pd.read_csv(R / "e3_forecasts.csv", keep_default_na=False)
_pv = fc.pivot_table(index=["ladder", "origin_m", "horizon", "target"], columns="model", values="abs_log_err").reset_index()
_loss = _pv[_pv["2-term"] - _pv["1-term"] > t["tie_tol"]]
assert len(_loss) == t["losses"] and int((_pv["1-term"] - _pv["2-term"] > t["tie_tol"]).sum()) == t["wins"], "E3 recount disagrees with e3_summary.json"
_loss_txt = "; ".join(f"{r.ladder} with {int(r.origin_m)} training sizes (target {r.target}), two-term {r['2-term']:.5f} vs one-term {r['1-term']:.5f}" for _, r in _loss.iterrows())
_med = {k: e3["median_abs_log_err_all"][k] for k in ("1-term", "2-term", "2-term beta=alpha/2")}
_med_txt = (f"{_med['1-term']:.5f} for all three" if len({round(v, 5) for v in _med.values()}) == 1
            else ", ".join(f"{k} {v:.5f}" for k, v in _med.items()))
item(f"Rolling-origin forecasts: {e3['n_forecasts']} (train on the m smallest sizes, m at least 5, predict the next one or two). Two-term vs one-term: {t['ties']} ties "
     f"(absolute log errors within {t['tie_tol']:g}), {t['wins']} wins, {t['losses']} loss; the loss is {_loss_txt}. "
     f"Mean absolute log error one-term {e3['mean_abs_log_err_all']['1-term']:.5f}, two-term {e3['mean_abs_log_err_all']['2-term']:.5f}, beta fixed at alpha/2 {e3['mean_abs_log_err_all']['2-term beta=alpha/2']:.5f}; "
     f"median {_med_txt}. Both models in a forecast use the same optimizer budget and the two-term fit is warm-started from the one-term fit, "
     "so a tie means the second term found nothing to add.")

# ------------------------------------------------------------------ E4
section("E4: Porian et al. grid: parameter-count definition and the small-model bend")
sub = a4[(a4.min_steps == 1000) & (a4.D_def == "D_repo")]
sz = pd.read_csv(R / "e4_sizes.csv")
_szs = {ds: sz[sz.dataset == ds].reset_index(drop=True) for ds in ("rw", "owt2")}
assert (_szs["rw"].N_ckpt.values == _szs["owt2"].N_ckpt.values).all() and (_szs["rw"].n_runs.values == _szs["owt2"].n_runs.values).all(), "the two datasets have different size grids"
_g0 = _szs["rw"]
_small = _g0[_g0.N_ckpt < 3e7]
assert int(_g0.n_runs.sum()) == int(sub[sub.N_proxy == "ckpt"].n.iloc[0]), "size table and fit table disagree on the run count"
item(f"Subset: Chinchilla learning-rate schedule, at least 1,000 steps: {int(_g0.n_runs.sum())} runs per dataset (RefinedWeb and OpenWebText2 have the same sizes and the same run count at each size), "
     f"{len(_g0)} model sizes from {_g0.N_ckpt.min() / 1e6:.0f}M to {_g0.N_ckpt.max() / 1e6:.0f}M parameters, of which {len(_small)} sizes "
     f"({', '.join(f'{v / 1e6:.0f}M' for v in _small.N_ckpt)}; {int(_small.n_runs.sum())} runs) are below 30M.")
for ds, nm in (("rw", "RefinedWeb"), ("owt2", "OpenWebText2")):
    g = sub[sub.dataset == ds].set_index("N_proxy")
    item(f"{nm}, compute-optimal exponent / alpha / beta / E by N definition: "
         + "; ".join(f"{p} {g.loc[p, 'a_opt']:.3f} / {g.loc[p, 'alpha']:.3f} / {g.loc[p, 'beta']:.3f} / {g.loc[p, 'E']:.3f}" for p in ("wxd", "ckpt", "12ld2", "12ld2+emb"))
         + f". Width-times-depth minus exact label: exponent {g.loc['wxd', 'a_opt'] - g.loc['ckpt', 'a_opt']:+.3f}, E {g.loc['wxd', 'E'] - g.loc['ckpt', 'E']:+.3f}. "
         f"Implied tokens per parameter at 1e21 FLOPs: wxd {g.loc['wxd', 'DoverN_1e21']:.0f}, exact label {g.loc['ckpt', 'DoverN_1e21']:.1f}.")
def _aopt(ds, ms, proxy):
    return a4[(a4.dataset == ds) & (a4.min_steps == ms) & (a4.D_def == "D_repo") & (a4.N_proxy == proxy)].a_opt.iloc[0]


item("Sensitivity to the minimum-steps filter (exponent with width-times-depth vs exact label, then the shift): "
     + "; ".join(f"{nm}: " + ", ".join(
         f"min steps {int(ms)} (n {int(a4[(a4.dataset == ds) & (a4.min_steps == ms) & (a4.D_def == 'D_repo')].n.iloc[0])}) "
         f"{_aopt(ds, ms, 'wxd'):.3f} vs {_aopt(ds, ms, 'ckpt'):.3f} ({_aopt(ds, ms, 'wxd') - _aopt(ds, ms, 'ckpt'):+.3f})" for ms in (1, 500, 1000, 2000))
                 for ds, nm in (("rw", "RefinedWeb"), ("owt2", "OpenWebText2"))) + ".")
for ds, nm in (("rw", "RefinedWeb"), ("owt2", "OpenWebText2")):
    g = bend[bend.dataset == ds]
    item(f"{nm}, exact label, fit-objective reduction from a second N term as the smallest kept model grows (all, 30M, 60M, 120M and up): "
         + ", ".join(f"{max(v, 0):.1f}%" for v in g.improvement_pct) + "; runs "
         + ", ".join(str(int(v)) for v in g.n_runs) + "; sizes " + ", ".join(str(int(v)) for v in g.n_sizes)
         + "; boundary flags " + ", ".join(f"'{f}'" if f else "none" for f in g.flags_Ndark) + ".")
for ds, nm in (("rw", "RefinedWeb"), ("owt2", "OpenWebText2")):
    g = bend2[(bend2.dataset == ds) & (bend2.n_min_ckpt == 0.0)].set_index("N_proxy")
    g30 = bend2[(bend2.dataset == ds) & (bend2.n_min_ckpt == 3e7)].set_index("N_proxy")
    item(f"{nm}, second N term on all 15 sizes by N definition (width-times-depth, exact label, no embeddings, plus embeddings): "
         + ", ".join(f"{max(g.loc[p, 'improvement_pct'], 0):.2f}%" for p in ("wxd", "ckpt", "12ld2", "12ld2+emb"))
         + "; boundary flags " + ", ".join(f"'{g.loc[p, 'flags_Ndark']}'" if g.loc[p, 'flags_Ndark'] else "none" for p in ("wxd", "ckpt", "12ld2", "12ld2+emb"))
         + "; keeping only 30M and up: largest gain " + f"{g30.improvement_pct.max():.1e}%".replace("e-0", "e-").replace("e+0", "e+") + ".")
e4s = json.loads((R / "e4_summary.json").read_text())
c15, c40 = e4s["backtest cap=1.5"], e4s["backtest cap=4.0"]
_w15, _w40 = c15["wins_ties_losses_vs_joint"], c40["wins_ties_losses_vs_joint"]
item(f"Porian backtests (8 splits: two size cuts, token cut, compute cut, per dataset), median held-out error at cap 1.5: joint {c15['median_rmse']['joint']:.4f}, +N {c15['median_rmse']['+N dark']:.4f}, "
     f"+D {c15['median_rmse']['+D dark']:.4f}, both {c15['median_rmse']['+both dark']:.4f}; "
     f"[wins, ties, losses] against the joint fit over the {c15['n_splits']} splits (ties are differences within {c15['tie_tol']:g}) at cap 1.5: +N {_w15['+N dark']}, +D {_w15['+D dark']}, both {_w15['+both dark']}, "
     f"at cap 4.0: +N {_w40['+N dark']}, +D {_w40['+D dark']}, both {_w40['+both dark']}; "
     f"boundary-flagged fits out of 8: +N {c15['boundary_flag_counts']['+N dark']}, +D {c15['boundary_flag_counts']['+D dark']}, both {c15['boundary_flag_counts']['+both dark']}, joint {c15['boundary_flag_counts']['joint']}. "
     f"At cap 4.0: joint {c40['median_rmse']['joint']:.4f}, +N {c40['median_rmse']['+N dark']:.4f}, +D {c40['median_rmse']['+D dark']:.4f}, both {c40['median_rmse']['+both dark']:.4f}.")

# ------------------------------------------------------------------ E2b (post hoc, appended so items 1-34 keep their numbers)
section("E2b: Does the Chinchilla result survive dropping the shortest runs? (post hoc: cut-offs chosen after seeing the residuals)")
F1, F2 = (json.loads((R / "e2b" / k / "e2b_summary.json").read_text()) for k in ("f1_drop_DN_lt_0.35", "f2_drop_DN_lt_1"))
SUBS = (("all 245 runs", e2, None), ("without the 4 shortest (D/N below 0.35)", F1, 0.35), ("without the 11 shortest (D/N below 1)", F2, 1.0))


def huber_sum(df, E, A, a, B, b, delta=1e-3):
    r = np.log(df.L.values) - np.log(E + A / df.N.values ** a + B / df.D.values ** b)
    ar = np.abs(r)
    return float(np.sum(np.where(ar <= delta, 0.5 * r * r, delta * (ar - 0.5 * delta))))


Ej, Aj, aj, Bj, bj = parse_joint(fj.desc)
rj = np.log(chin.L.values) - np.log(Ej + Aj / chin.N.values ** aj + Bj / chin.D.values ** bj)
short = (chin.D / chin.N).values < 0.35
_hj = np.where(np.abs(rj) <= 1e-3, 0.5 * rj * rj, 1e-3 * (np.abs(rj) - 0.5e-3))   # per-point Huber objective, delta 0.001
_ED, _AD, _aD = (float(x) for x in re.search(r"E=([0-9.e+-]+) \+ ([0-9.e+-]+)/N\^([0-9.e+-]+)", fdd.desc).groups())
_d1, _d2 = re.findall(r"\+ ([0-9.e+-]+)/D\^([0-9.e+-]+)", fdd.desc)
_rD = np.log(chin.L.values) - np.log(_ED + _AD / chin.N.values ** _aD + float(_d1[0]) / chin.D.values ** float(_d1[1]) + float(_d2[0]) / chin.D.values ** float(_d2[1]))
_hD = np.where(np.abs(_rD) <= 1e-3, 0.5 * _rD * _rD, 1e-3 * (np.abs(_rD) - 0.5e-3))   # same objective under the second-D-term fit (cap 1.5)
item(f"The {int(short.sum())} runs with D/N below 0.35 (N, D, D/N): "
     + "; ".join(f"{n:.3g}, {d:.3g}, {d / n:.2f}" for n, d in zip(chin.N.values[short], chin.D.values[short]))
     + f". Under the joint fit of item 13 they carry {100 * (rj[short] ** 2).sum() / (rj ** 2).sum():.0f}% of the residual sum of squares "
     f"and {100 * _hj[short].sum() / _hj.sum():.0f}% of the Huber objective "
     f"(log residuals {rj[short].min():+.3f} to {rj[short].max():+.3f}, observed loss above the fit); the other {int((~short).sum())} runs have a root-mean-square residual of {np.sqrt(np.mean(rj[~short] ** 2)):.4f} "
     f"against {np.sqrt(np.mean(rj ** 2)):.4f} for all {len(rj)}. Under the second-D-term fit (item 14) the four runs supply "
     f"{100 * (_hj[short].sum() - _hD[short].sum()) / (_hj.sum() - _hD.sum()):.0f}% of that fit's objective reduction and still carry {100 * _hD[short].sum() / _hD.sum():.0f}% of its objective. The next-lowest D/N in the data is {np.sort((chin.D / chin.N).values)[4]:.2f}.")
item("Multi-start joint fit by subset (exponents uncapped; compute-optimal exponent b/(a+b)): "
     + "; ".join(f"{nm}: {fit} | Huber {hub:.3e}, RSS {rss:.4f}, exponent {ao:.4f}" for nm, fit, hub, rss, ao in (
         ("all runs", fj.desc, float(fj.huber), float(fj.rss), bj / (aj + bj)),
         ("without 4", F1["free_joint"]["desc"], F1["free_joint"]["huber"], F1["free_joint"]["rss"], F1["free_joint"]["a_opt"]),
         ("without 11", F2["free_joint"]["desc"], F2["free_joint"]["huber"], F2["free_joint"]["rss"], F2["free_joint"]["a_opt"]))) + ".")
gain = []
for nm, S, thr in SUBS[1:]:
    fjs, fds = S["full_data_fits"]["cap=1.5|joint"], S["full_data_fits"]["cap=1.5|+D dark"]
    gain.append(f"{nm}: Huber {100 * (1 - fds['huber'] / fjs['huber']):.1f}% lower, RSS {100 * (1 - fds['rss'] / fjs['rss']):.0f}% lower, flags {fds['flags'] or 'none'}, fit {fds['desc']}")
item(f"Second D term, full-data fit (exponents capped at 1.5), improvement over the joint fit. All runs: Huber {100 * (1 - fdd.huber / fj.huber):.1f}% lower, RSS {100 * (1 - fdd.rss / fj.rss):.0f}% lower. "
     + "; ".join(gain) + ".")
for lab, capk in (("exponents capped at 1.5", "cap=1.5"), ("exponents capped at 4.0", "cap=4.0")):
    parts = []
    for nm, S, thr in SUBS:
        ba, ov = S[capk]["median_test_rmse_by_axis"], S[capk]["median_test_rmse"]
        parts.append(f"{nm}: size axis {ba['N']['joint']:.4f} -> {ba['N']['+D dark']:.4f}, token axis {ba['D']['joint']:.4f} -> {ba['D']['+D dark']:.4f}, "
                     f"compute axis {ba['C']['joint']:.4f} -> {ba['C']['+D dark']:.4f}, median over twelve cuts {ov['joint']:.4f} -> {ov['+D dark']:.4f} "
                     f"(token axis {100 * (1 - ba['D']['+D dark'] / ba['D']['joint']):.0f}% lower, all cuts {100 * (1 - ov['+D dark'] / ov['joint']):.0f}% lower)")
    item(f"Held-out median error, joint -> joint plus second D term ({lab}): " + "; ".join(parts) + ".")
item("Against the joint fit, [wins, ties, losses] over the twelve cuts (a held-out RMSE difference within 1e-6 is a tie), by subset, cap 1.5 | cap 4.0. Second D term: "
     + "; ".join(f"{nm}: " + " | ".join(str(S[k]['wins_ties_losses_vs_joint']['+D dark']) for k in ('cap=1.5', 'cap=4.0')) for nm, S, thr in SUBS)
     + ". Second N term: "
     + "; ".join(f"{nm}: " + " | ".join(str(S[k]['wins_ties_losses_vs_joint']['+N dark']) for k in ('cap=1.5', 'cap=4.0')) for nm, S, thr in SUBS)
     + ". Boundary-flagged fits among the twelve (cap 1.5), joint / +N / +D / both: "
     + "; ".join(f"{nm}: " + " / ".join(str(S['cap=1.5']['boundary_flag_counts'][m]) for m in ('joint', '+N dark', '+D dark', '+both dark')) for nm, S, thr in SUBS) + ".")
item("Repo single-start fit against the multi-start joint fit over the twelve cuts (cap 1.5), by subset, as repo better by more than 1% / within 1% / repo worse by more than 1% "
     "(relative difference in held-out RMSE): "
     + "; ".join(f"{nm}: {S['cap=1.5']['repo_vs_joint']['repo_better_by_over_1pct']} / {S['cap=1.5']['repo_vs_joint']['within_1pct']} / {S['cap=1.5']['repo_vs_joint']['repo_worse_by_over_1pct']}" for nm, S, thr in SUBS)
     + "; median held-out error over twelve cuts, repo fit against multi-start joint fit: "
     + "; ".join(f"{nm}: {S['cap=1.5']['median_test_rmse']['repo']:.4f} against {S['cap=1.5']['median_test_rmse']['joint']:.4f}" for nm, S, thr in SUBS)
     + "; token-axis median held-out error, repo single-start fit against multi-start joint fit against the multi-start fit with a second D term: "
     + "; ".join(f"{nm}: {S['cap=1.5']['median_test_rmse_by_axis']['D']['repo']:.4f} against {S['cap=1.5']['median_test_rmse_by_axis']['D']['joint']:.4f} against {S['cap=1.5']['median_test_rmse_by_axis']['D']['+D dark']:.4f}" for nm, S, thr in SUBS) + ".")
PUB = (1.82, 482.0, 0.34, 2085.0, 0.36)  # the two-decimal constants written in AGENTS.md
item("Two-decimal replication constants from AGENTS.md (E 1.82, A 482, alpha 0.34, B 2085, beta 0.36), Huber objective at delta 0.001: "
     + "; ".join(f"{nm}: {huber_sum(chin if thr is None else chin[(chin.D / chin.N) >= thr], *PUB):.3e}" for nm, S, thr in SUBS)
     + "; the repo's RESULTS.md quotes 1.19e-2 for all 245 runs. For comparison the multi-start minimum is "
     + "; ".join(f"{nm}: {(float(fj.huber) if thr is None else S['free_joint']['huber']):.3e}" for nm, S, thr in SUBS) + ".")

# ------------------------------------------------------------------ cross-check of another agent's experiment (appended so items 1-42 keep their numbers)
section("X: Cross-check of experiment 09 on brando90/dark-matter main (my recomputation from its saved predictions)")
_x = json.loads((R / "crosscheck_exp09.json").read_text())
_c = _x["cells"]
_nm = {"chinchilla_runs": "Chinchilla runs", "epoch_chinchilla_figure4": "Epoch extraction"}
_md = {"joint": "joint", "joint_extra_n": "+N term", "joint_extra_d": "+D term"}
_readme = {"chinchilla_runs": 0.2501, "epoch_chinchilla_figure4": 0.2421}      # transcribed from experiment 09's README
_match = [k for k, v in _c.items() if round(v["rmse_all"], 4) == _readme[k.split("|")[0]]]
_miss = [(k, v["rmse_all"]) for k, v in _c.items() if k not in _match]
item(f"Experiment 09 joint size-holdout test error ({_x['source']}). Test RMSE in loss units, then the number of runs with D/N below {_x['threshold_D_over_N']} in the test split and in the train split, "
     "their share of the test squared error, and the test RMSE without them: "
     + "; ".join(f"{_nm[k.split('|')[0]]}, {_md[k.split('|')[1]]}: {v['rmse_all']:.4f}, {v['n_short_in_test']} in test of {v['n_test']} and {v['n_short_in_train']} in train of {v['n_train']}, "
                 f"{100 * v['share_sq_err_short']:.1f}% of squared error, {v['rmse_without_short']:.4f} without them" for k, v in _c.items())
     + ". The README values 0.2501 (Chinchilla runs) and 0.2421 (Epoch extraction) are transcribed from that experiment's README, not computed here; "
     f"{len(_match)} of the {len(_c)} test RMSE values above match them to four decimals"
     + (f", and the other{'s are' if len(_miss) > 1 else ' is'} " + ", ".join(f"{_nm[k.split('|')[0]]} {_md[k.split('|')[1]]} {v:.4f}" for k, v in _miss) if _miss else "") + ".")

section("Y: The four Chinchilla runs with D/N below 0.35 (from the 245-point cache)")
_s = chin[(chin.D / chin.N) < 0.35].sort_values("N")
item("Runs, as model size N in parameters, tokens D, D/N, loss, and the share of all 245 runs with a model size at or below it: "
     + "; ".join(f"N {r.N:.3g}, D {r.D:.3g}, D/N {r.D / r.N:.3f}, loss {r.L:.3f}, size share {100 * (chin.N <= r.N).mean():.0f}%" for r in _s.itertuples())
     + f". Across the four, N runs from {_s.N.min():.3g} to {_s.N.max():.3g} and D from {_s.D.min():.3g} to {_s.D.max():.3g}; the largest N among all 245 runs is {chin.N.max():.3g}.")

print("\n".join(lines).lstrip("\n"))
