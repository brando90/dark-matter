"""
E3 -- Real fixed-data model-size ladders: is there evidence for a second, shallower N term, and does it forecast better?

Ladders (all models of a family saw the same number of tokens, final checkpoint only), loss = log(LAMBADA perplexity)
from the ColPret / Hitchhiker's Guide per-family files ("lambada_openai ppl" is the only language-modelling style
metric present at final checkpoints in these files):
    pythia           EleutherAI Pythia (Pile, no dedup), ~300B tokens
    pythia-deduped   Pythia on the deduplicated Pile, ~300B tokens
    opt              OPT, 300B tokens
BLOOM has only five sizes and is skipped (a 5-parameter fit would interpolate it exactly).

For each ladder:
  A. full-ladder fits of 1-term, 2-term (beta<alpha) and 2-term with beta = alpha/2, plus a parametric-bootstrap
     p-value for the Gaussian LRT K log(RSS1/RSS2) under H0 = the fitted 1-term model with log-normal noise.
  B. rolling-origin forecasts: train on the m smallest models (m >= 5), predict the next one (h=1) and next two (h=2).

Usage: python e3_real_ladders.py [--boot 200] [--workers 2] [--analyze-only]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from py_src import scaling_ext as se  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
DATA = REPO_ROOT / "data" / "data_hitchhikers_guide"
METRIC = "lambada_openai ppl"
FAMILIES = {
    "pythia": "colpret_pythia.csv.gz",
    "pythia-deduped": "colpret_pythia-deduped.csv.gz",
    "opt": "colpret_OPT.csv.gz",
}
FIT = dict(n_starts=150, n_refine=3, max_nfev=1000)
M_MIN = 5


def load_ladder(fname: str) -> pd.DataFrame:
    d = pd.read_csv(DATA / fname, usecols=["model_name", "num_params", "tokens_seen", METRIC], low_memory=False)
    d = d.dropna(subset=[METRIC])
    d = d[d.tokens_seen >= 0.99 * d.tokens_seen.max()]
    d = d.sort_values("tokens_seen").groupby("model_name").tail(1)
    names = set(d.model_name)
    # Pythia's 6.7b is the old name of 6.9b: keep one of the two
    drop = {n for n in names if "6.7b" in n and n.replace("6.7b", "6.9b") in names}
    d = d[~d.model_name.isin(drop)]
    # "pythia-19m(-deduped)" is not part of the standard Pythia suite and is non-monotone in the data (its LAMBADA
    # perplexity, 89, is lower than the 70m model's, 138), so a smooth size ladder starts at 70m.
    d = d[d.num_params >= 5e7].sort_values("num_params").reset_index(drop=True)
    d["L"] = np.log(d[METRIC].astype(float))
    return d.rename(columns={"num_params": "N"})


def _boot_one(args):
    N, L1_clean, sigma, seed = args
    rng = np.random.default_rng(seed)
    L = L1_clean * np.exp(rng.normal(0.0, sigma, len(N)))
    one = se.fit(se.ONE_N, N, L, objective="ls", seed=seed, **FIT)
    two = se.fit(se.TWO_N, N, L, objective="ls", seed=seed + 1, warm=one, **FIT)
    return float(len(N) * np.log(one.rss / max(two.rss, 1e-300)))


def part_a(name, d, boot, workers):
    N, L = d.N.values, d.L.values
    K = len(N)
    one_l = se.fit(se.ONE_N, N, L, objective="ls", n_starts=300, n_refine=5, seed=1)
    two_l = se.fit(se.TWO_N, N, L, objective="ls", n_starts=300, n_refine=5, seed=2, warm=one_l)
    one_h = se.fit(se.ONE_N, N, L, objective="huber", n_starts=300, n_refine=5, seed=3)
    two_h = se.fit(se.TWO_N, N, L, objective="huber", n_starts=300, n_refine=5, seed=4, warm=one_h)
    half_h = se.fit(se.TWO_N_HALF, N, L, objective="huber", n_starts=300, n_refine=5, seed=5, warm=None)
    stat = float(K * np.log(one_l.rss / max(two_l.rss, 1e-300)))
    sigma = float(np.sqrt(one_l.rss / max(K - 3, 1)))
    clean = se.predict(one_l, N)
    jobs = [(N, clean, sigma, 9_000 + i) for i in range(boot)]
    with Pool(workers) as p:
        stars = np.array(p.map(_boot_one, jobs, chunksize=4))
    pval = float((1 + np.sum(stars >= stat)) / (boot + 1))
    return dict(
        ladder=name, K=K, models=d.model_name.tolist(), N=N.tolist(), lambada_ppl=d[METRIC].tolist(),
        sigma_hat=sigma, gauss_lrt_stat=stat, boot_B=boot, boot_pvalue=pval,
        boot_q95=float(np.quantile(stars, 0.95)), boot_median=float(np.median(stars)),
        fit_1term_ls=se.describe(one_l), fit_2term_ls=se.describe(two_l),
        fit_1term_huber=se.describe(one_h), fit_2term_huber=se.describe(two_h), fit_half_huber=se.describe(half_h),
        rss_1term=one_l.rss, rss_2term=two_l.rss, beta_over_alpha_2term=float(two_l.e[1] / two_l.e[0]),
        corner_2term=bool(two_l.e[1] < 0.02 or two_l.E < 0.2 * one_l.E),
    )


def part_b(name, d):
    N, L = d.N.values, d.L.values
    K = len(N)
    rows = []
    for m in range(M_MIN, K):
        tr = slice(0, m)
        fits = {}
        fits["1-term"] = se.fit(se.ONE_N, N[tr], L[tr], objective="huber", seed=31, **FIT)
        fits["2-term"] = se.fit(se.TWO_N, N[tr], L[tr], objective="huber", seed=32, warm=fits["1-term"], **FIT)
        fits["2-term beta=alpha/2"] = se.fit(se.TWO_N_HALF, N[tr], L[tr], objective="huber", seed=33, **FIT)
        for h in (1, 2):
            j = m + h - 1
            if j >= K:
                continue
            for mname, res in fits.items():
                pred = float(se.predict(res, N[j:j + 1])[0])
                rows.append(dict(ladder=name, origin_m=m, horizon=h, target=d.model_name[j], N_target=float(N[j]),
                                 model=mname, pred=pred, obs=float(L[j]), abs_log_err=abs(np.log(pred) - np.log(L[j])),
                                 signed_log_err=float(np.log(pred) - np.log(L[j]))))
    return rows


TIE_TOL = 1e-6    # two forecasts whose absolute log errors differ by less than this count as identical


def forecast_summary(bt):
    """Aggregate the rolling-origin forecasts. Ties are counted explicitly: a strict '<' on floats would score
    numerical noise as a win for one model."""
    out = {}
    piv = bt.groupby(["ladder", "horizon", "model"]).abs_log_err.mean().unstack("model")
    out["mean_abs_log_forecast_error"] = {f"{k[0]}|h={k[1]}": {m: float(v) for m, v in row.items()} for k, row in piv.iterrows()}
    wide = bt.pivot_table(index=["ladder", "origin_m", "horizon"], columns="model", values="abs_log_err")
    out["n_forecasts"] = int(len(wide))
    for key, col in (("two_term_vs_one_term", "2-term"), ("half_vs_one_term", "2-term beta=alpha/2")):
        d = wide[col] - wide["1-term"]
        out[key] = dict(ties=int((d.abs() < TIE_TOL).sum()), wins=int((d < -TIE_TOL).sum()),
                        losses=int((d > TIE_TOL).sum()), max_abs_diff=float(d.abs().max()), tie_tol=TIE_TOL)
    out["median_abs_log_err_all"] = {m: float(wide[m].median()) for m in wide.columns}
    out["mean_abs_log_err_all"] = {m: float(wide[m].mean()) for m in wide.columns}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--analyze-only", action="store_true",
                    help="recompute only the forecast aggregates from results/e3_forecasts.csv (keeps the saved full-ladder fits)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.analyze_only:
        old = json.loads((OUT / "e3_summary.json").read_text())
        summ = dict(full_ladder=old["full_ladder"])
        summ.update(forecast_summary(pd.read_csv(OUT / "e3_forecasts.csv")))
        (OUT / "e3_summary.json").write_text(json.dumps(summ, indent=2))
        print(json.dumps({k: v for k, v in summ.items() if k != "full_ladder"}, indent=2))
        return
    full, back = [], []
    for name, fname in FAMILIES.items():
        d = load_ladder(fname)
        print(f"[E3] ladder {name}: K={len(d)}; sizes {d.N.min():.3g}..{d.N.max():.3g}; "
              f"ppl {d[METRIC].round(2).tolist()}", flush=True)
        r = part_a(name, d, a.boot, a.workers)
        print(f"[E3a] {name}: LRT stat {r['gauss_lrt_stat']:.3f}, bootstrap p {r['boot_pvalue']:.3f} (q95 {r['boot_q95']:.2f}); "
              f"1-term {r['fit_1term_ls']} | 2-term {r['fit_2term_ls']}", flush=True)
        full.append(r)
        back += part_b(name, d)
    bt = pd.DataFrame(back)
    bt.to_csv(OUT / "e3_forecasts.csv", index=False)
    summ = dict(full_ladder=full)
    summ.update(forecast_summary(bt))
    (OUT / "e3_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({k: v for k, v in summ.items() if k != "full_ladder"}, indent=2))

if __name__ == "__main__":
    main()
