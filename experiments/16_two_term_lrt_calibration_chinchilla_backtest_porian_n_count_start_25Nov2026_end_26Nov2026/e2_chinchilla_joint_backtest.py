"""
E2 -- Does an extra additive "dark matter" term help the joint Chinchilla law extrapolate, and how well are
the joint parameters identified at all?

Data: the 245 Chinchilla (MassiveText) training points in data/data_hitchhikers_guide (N, D, loss).

Part A (backtests): train on the small end, test on the large end, for three ways of cutting the data:
    by model size N, by token count D, by compute C = 6 N D, each at the 40/50/60/70 % quantile.
  Models (all additive, all Huber delta=1e-3 on log-residuals, multi-start):
    repo     : fit_joint_powerlaw from py_src/scaling_laws_fit.py (single L-BFGS-B start, as shipped)
    joint    : E + A/N^a + B/D^b
    +N dark  : joint + c/N^(rho a)         (rho in (0,1))
    +D dark  : joint + c/D^(rho b)
    +both    : joint + both dark terms
Part B (identifiability): profile the Huber objective over the data exponent b and over the model exponent a
  on all 245 points, and report the implied compute-optimal allocation exponent  a_opt = b / (a + b).
Part C: full-data fits of all four additive models with boundary diagnostics.
Every dark-term fit is run twice: unconstrained (exponents up to 4, so a very steep term can act as a localised
correction) and 'smooth' (exponents <= 1.5). Fits that end on a boundary are flagged (see scaling_ext.diagnose).
Wins against the joint fit are counted tie-aware (|difference in held-out RMSE| <= 1e-6 is a tie), because a fit that
switches an extra term off differs from the joint fit by 1e-10 and a strict '<' would call that a win.

Usage: python e2_chinchilla_joint_backtest.py [--quick]
       python e2_chinchilla_joint_backtest.py --recount-only     (tie-aware counts from the saved backtest, no fitting)
       python e2_chinchilla_joint_backtest.py --profile-only     (add the fine exponent grid to the saved profile)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from py_src import scaling_ext as se  # noqa: E402
from py_src.scaling_laws_fit import fit_joint_powerlaw  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
DATA_GZ = REPO_ROOT / "data" / "data_hitchhikers_guide" / "hitchhickers_guide_unified.csv.gz"
CACHE = OUT / "chinchilla_245.csv"          # tiny derived file (not under data/, so git keeps it)

QUANTILES = (0.4, 0.5, 0.6, 0.7)
EXP_CAPS = (4.0, 1.5)      # unconstrained / smooth power laws only
SPECS = [
    ("joint", se.JOINT, None),
    ("+N dark", se.JOINT_N_DARK, "joint"),
    ("+D dark", se.JOINT_D_DARK, "joint"),
    ("+both dark", se.JOINT_BOTH_DARK, "+N dark"),
]
TIE_TOL = 1e-6          # held-out RMSE differences below this are ties (an extra term switched off, or optimizer noise); as in E3
REPO_REL_TOL = 0.01     # the repo single-start fit and the multi-start joint fit count as close when within 1% relative RMSE
# Exponent grids for the profile (part B). The first grid is wide but 16% apart; the second is 0.01 apart around the optimum so
# that the 1% and 5% bands are resolved (they were read off two or three grid points of the coarse grid before).
PROFILE_GRIDS = [("b (data exponent)", 1, np.geomspace(0.15, 1.2, 15)), ("a (model exponent)", 0, np.geomspace(0.12, 0.9, 15))]
FINE_GRIDS = [("b (data exponent)", 1, np.round(np.arange(0.34, 0.601, 0.01), 3)),
              ("a (model exponent)", 0, np.round(np.arange(0.26, 0.451, 0.01), 3))]


def split_table(bt: pd.DataFrame, em: float) -> pd.DataFrame:
    """Held-out RMSE by split (index: axis, quantile) and model (columns, repo included) for one exponent cap."""
    repo = bt[bt.model == "repo"].set_index(["axis", "quantile"]).test_rmse
    piv = bt[bt.exp_max == em].pivot_table(index=["axis", "quantile"], columns="model", values="test_rmse")
    piv["repo"] = repo
    return piv


def win_fields(piv: pd.DataFrame) -> dict:
    """Tie-aware counts over the splits in `piv`: each extended model against the joint fit as [wins, ties, losses], and the
    repo single-start fit against the multi-start joint fit by relative difference (so that near-ties are visible).
    A strict '<' would count differences of 1e-10, from fits that switch the extra term off, as wins."""
    out = dict(tie_tol=TIE_TOL, n_splits=int(len(piv)), wins_ties_losses_vs_joint={})
    for m in ("+N dark", "+D dark", "+both dark"):
        d = (piv[m] - piv["joint"]).to_numpy()
        out["wins_ties_losses_vs_joint"][m] = [int(np.sum(d < -TIE_TOL)), int(np.sum(np.abs(d) <= TIE_TOL)), int(np.sum(d > TIE_TOL))]
    rel = ((piv["repo"] - piv["joint"]) / piv["joint"]).to_numpy()
    out["repo_vs_joint"] = dict(repo_better_by_over_1pct=int(np.sum(rel < -REPO_REL_TOL)), within_1pct=int(np.sum(np.abs(rel) <= REPO_REL_TOL)),
                                repo_worse_by_over_1pct=int(np.sum(rel > REPO_REL_TOL)))
    return out


def band_summary(prof: pd.DataFrame) -> dict:
    """Parameter and compute-optimal-exponent ranges whose profiled objective is within 1% and 5% of the best fit."""
    out = {}
    for which in prof.profiled.unique():
        p = prof[prof.profiled == which]
        for tol in (1.0, 5.0):
            band = p[p.excess_pct <= tol]
            out[f"profile {which} within {tol:g}%"] = dict(
                fixed_range=[float(band.fixed_value.min()), float(band.fixed_value.max())] if len(band) else None,
                a_opt_range=[float(band.a_opt.min()), float(band.a_opt.max())] if len(band) else None,
                n_grid_points=int(len(band)))
    return out


def recount(summary_path: Path, bt_path: Path) -> None:
    """Replace the strict win counts in a saved summary by tie-aware ones, from the saved backtest CSV (no refitting)."""
    bt = pd.read_csv(bt_path)
    summ = json.loads(summary_path.read_text())
    for em in EXP_CAPS:
        s = summ[f"cap={em}"]
        for old in ("win_rates_vs_joint", "wins_vs_joint_count", "joint_beats_repo_rate"):
            s.pop(old, None)
        s.update(win_fields(split_table(bt, em)))
    summary_path.write_text(json.dumps(summ, indent=2))


def load_chinchilla() -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    if CACHE.exists():
        return pd.read_csv(CACHE)
    cols = ["family", "dataset", "num_params", "tokens", "loss"]
    df = pd.read_csv(DATA_GZ, usecols=cols, low_memory=False)
    ch = df[(df.family == "chinchilla") & (df.dataset == "MassiveText") & (df.num_params > 0)]
    ch = ch[["num_params", "tokens", "loss"]].rename(columns={"num_params": "N", "tokens": "D", "loss": "L"})
    ch = ch.reset_index(drop=True)
    ch.to_csv(CACHE, index=False)
    return ch


def repo_as_result(jp):
    return se.FitResult(se.JOINT, jp.E, np.array([jp.A, jp.B]), np.array([jp.alpha, jp.beta]), 0.0, 0.0, 0, True)


def split_masks(df: pd.DataFrame):
    C = 6.0 * df.N.values * df.D.values
    keys = {"N": df.N.values, "D": df.D.values, "C": C}
    for axis, v in keys.items():
        for q in QUANTILES:
            cut = float(np.quantile(v, q))
            yield axis, q, cut, v <= cut


def metrics(res, d: pd.DataFrame):
    r = se.log_resid(res, d.N.values, d.L.values, d.D.values)
    return dict(rmse=float(np.sqrt(np.mean(r ** 2))), bias=float(np.mean(r)), mae=float(np.mean(np.abs(r))))


def part_a(df: pd.DataFrame, n_starts: int, n_refine: int):
    rows = []
    for axis, q, cut, mask in split_masks(df):
        tr, te = df[mask], df[~mask]
        N, D, L = tr.N.values, tr.D.values, tr.L.values
        t0 = time.time()
        jp = fit_joint_powerlaw(N, D, L)
        res = repo_as_result(jp)
        mt, me = metrics(res, tr), metrics(res, te)
        rows.append(dict(axis=axis, quantile=q, cut=cut, n_train=len(tr), n_test=len(te), model="repo", exp_max="n/a",
                         n_params=5, train_huber=se.objective_of(res, N, L, D), train_rmse=mt["rmse"],
                         test_rmse=me["rmse"], test_bias=me["bias"], test_mae=me["mae"], flags="", desc=se.describe(res)))
        for em in EXP_CAPS:
            prev = {}
            for name, spec, warm_name in SPECS:
                warm = prev.get(warm_name) if warm_name else None
                res = se.fit(spec, N, L, D, n_starts=n_starts, n_refine=n_refine, seed=7, warm=warm, max_nfev=1500, exp_max=em)
                prev[name] = res
                mt, me = metrics(res, tr), metrics(res, te)
                rows.append(dict(axis=axis, quantile=q, cut=cut, n_train=len(tr), n_test=len(te), model=name, exp_max=em,
                                 n_params=res.spec.n_params, train_huber=se.objective_of(res, N, L, D),
                                 train_rmse=mt["rmse"], test_rmse=me["rmse"], test_bias=me["bias"], test_mae=me["mae"],
                                 flags=se.diagnose(res, L, em), desc=se.describe(res)))
        print(f"[E2a] split {axis} q={q}: train {len(tr)} / test {len(te)}  ({time.time() - t0:.1f}s)", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "e2_backtest.csv", index=False)
    return out


def part_c(df: pd.DataFrame, n_starts: int, n_refine: int):
    N, D, L = df.N.values, df.D.values, df.L.values
    rows = []
    for em in EXP_CAPS:
        prev = {}
        for name, spec, warm_name in SPECS:
            warm = prev.get(warm_name) if warm_name else None
            res = se.fit(spec, N, L, D, n_starts=max(n_starts, 300), n_refine=max(n_refine, 5), seed=9, warm=warm,
                         max_nfev=3000, exp_max=em)
            prev[name] = res
            rows.append(dict(exp_max=em, model=name, n_params=res.spec.n_params, huber=res.objective, rss=res.rss,
                             flags=se.diagnose(res, L, em), desc=se.describe(res)))
            print(f"[E2c] cap {em} {name}: {se.describe(res)} | huber {res.objective:.4e} | flags {rows[-1]['flags'] or 'none'}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "e2_full_fits.csv", index=False)
    return out


def profile_rows(df: pd.DataFrame, n_starts: int, best: float, grids) -> list:
    N, D, L = df.N.values, df.D.values, df.L.values
    rows = []
    for which, idx, grid in grids:
        for g in grid:
            r = se.fit(se.JOINT, N, L, D, n_starts=n_starts, n_refine=4, seed=5, fix_main={idx: float(g)}, max_nfev=1500)
            a, b = float(r.e[0]), float(r.e[1])
            rows.append(dict(profiled=which, fixed_value=float(g), huber=r.objective, excess_pct=100 * (r.objective / best - 1),
                             E=r.E, A=float(r.c[0]), a=a, B=float(r.c[1]), b=b, a_opt=b / (a + b)))
            print(f"[E2] profile {which} = {g:.3f}: excess {rows[-1]['excess_pct']:.3f}%", flush=True)
    return rows


def part_b(df: pd.DataFrame, n_starts: int):
    N, D, L = df.N.values, df.D.values, df.L.values
    free = se.fit(se.JOINT, N, L, D, n_starts=300, n_refine=6, seed=3)
    prof = pd.DataFrame(profile_rows(df, n_starts, free.objective, PROFILE_GRIDS + FINE_GRIDS))
    prof = prof.sort_values(["profiled", "fixed_value"]).reset_index(drop=True)
    prof.to_csv(OUT / "e2_profile.csv", index=False)
    return free, prof


def refine_profile(df: pd.DataFrame, n_starts: int) -> None:
    """Add the fine grid to the saved profile and refresh the band entries of e2_summary.json (the other fits are untouched)."""
    N, D, L = df.N.values, df.D.values, df.L.values
    free = se.fit(se.JOINT, N, L, D, n_starts=300, n_refine=6, seed=3)      # the same call as in part_b, so 'best' is identical
    prof = pd.read_csv(OUT / "e2_profile.csv", keep_default_na=False)
    prof = prof[~prof.fixed_value.round(3).isin({g for _, _, grid in FINE_GRIDS for g in grid})]  # idempotent on re-runs
    prof = pd.concat([prof, pd.DataFrame(profile_rows(df, n_starts, free.objective, FINE_GRIDS))])
    prof = prof.sort_values(["profiled", "fixed_value"]).reset_index(drop=True)
    prof.to_csv(OUT / "e2_profile.csv", index=False)
    summ = json.loads((OUT / "e2_summary.json").read_text())
    for k in [k for k in summ if k.startswith("profile ")]:
        del summ[k]
    summ.update(band_summary(prof))
    (OUT / "e2_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(band_summary(prof), indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--recount-only", action="store_true",
                    help="rewrite the win counts in results/e2_summary.json as tie-aware counts from the saved backtest CSV; no fitting")
    ap.add_argument("--profile-only", action="store_true",
                    help="add the fine exponent grid to results/e2_profile.csv and refresh the profile bands in e2_summary.json")
    a = ap.parse_args()
    ns, nr = (60, 2) if a.quick else (200, 4)
    if a.recount_only:
        recount(OUT / "e2_summary.json", OUT / "e2_backtest.csv")
        print("[E2] win counts recounted from the saved backtest", flush=True)
        return
    df = load_chinchilla()
    if a.profile_only:
        refine_profile(df, ns)
        return
    print(f"[E2] {len(df)} Chinchilla points; N in [{df.N.min():.3g},{df.N.max():.3g}], D in [{df.D.min():.3g},{df.D.max():.3g}]", flush=True)

    free, prof = part_b(df, ns)
    print("[E2] free joint fit:", se.describe(free), "| huber", free.objective, flush=True)

    bt = part_a(df, ns, nr)
    full = part_c(df, ns, nr)

    # ---- summary ------------------------------------------------------------------------------
    summ = dict(n_points=int(len(df)), free_joint=dict(desc=se.describe(free), huber=free.objective, rss=free.rss),
                full_data_fits={f"cap={r.exp_max}|{r.model}": dict(huber=r.huber, flags=r.flags, desc=r.desc) for r in full.itertuples()})
    for em in EXP_CAPS:
        sub = bt[bt.exp_max == em]
        piv = split_table(bt, em)
        key = f"cap={em}"
        summ[key] = dict(
            test_rmse_by_split={f"{k[0]}@{k[1]}": {m: float(v) for m, v in row.items()} for k, row in piv.iterrows()},
            median_test_rmse={m: float(piv[m].median()) for m in piv.columns},
            boundary_flag_counts={m: int((sub[sub.model == m]["flags"] != "").sum()) for m in ("joint", "+N dark", "+D dark", "+both dark")},
            median_train_huber={m: float(sub[sub.model == m].train_huber.median()) for m in ("joint", "+N dark", "+D dark", "+both dark")},
        )
        summ[key].update(win_fields(piv))
        by_axis = {}
        for axis in ("N", "D", "C"):
            pa = piv.loc[axis]
            by_axis[axis] = {m: float(pa[m].median()) for m in pa.columns}
        summ[key]["median_test_rmse_by_axis"] = by_axis
    summ.update(band_summary(prof))
    (OUT / "e2_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
