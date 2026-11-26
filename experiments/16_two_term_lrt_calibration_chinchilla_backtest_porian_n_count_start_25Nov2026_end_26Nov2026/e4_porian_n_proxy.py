"""
E4 -- How much does the parameter-count proxy matter on the Porian et al. (arXiv:2406.19146) grid, and does an
additive dark-matter term help extrapolation on this cleaner (N, D) grid?

AGENTS.md warns that py_src/chinchilla_verification.py uses N = width * depth (a rough proxy). Here we fit the joint law
    L = E + A/N^alpha + B/D^beta
with four different N definitions on the same runs:
    wxd       width * depth                              (current repo proxy)
    ckpt      the exact parameter count in the checkpoint label (params=611M ...), looked up per (width, depth)
    12ld2     12 * depth * width^2                       (non-embedding transformer estimate)
    12ld2+emb 12 * depth * width^2 + vocab * width       (adds one tied embedding matrix)
with D = max_step * seq_len * bs (repo convention) and, as a robustness check, D from the last evaluation index.

Subsets: dataset in {rw, owt2}, decay == 'chinchilla' (100 runs each, 16 model sizes, one LR, bs 256, short warmup).
Runs with very few optimisation steps sit in the early-training transient (loss 6-8, far above a power-law regime), so
the primary analysis keeps runs with max_step >= 1000 (about 54 per dataset); thresholds 1, 500, 1000, 2000 are reported
as a sensitivity check.
Part B backtests joint vs +N dark vs +D dark vs +both dark with the exact N label on the max_step >= 1000 subset.
Part C asks whether the N dark term is a long-range second power law or a localised small-N bend: refit joint vs
+N dark after dropping the smallest model sizes (N >= 0, 30M, 60M, 120M).
Part C2 repeats part C on the same run subsets (cut by the exact parameter label) but fits with each of the four N
proxies, to see whether the small-N bend is just an embedding-dominated parameter count at the small end.

Run with isolated, empty environment:   env -i PATH="$PATH" HOME=/tmp python3 -I e4_porian_n_proxy.py
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
from py_src.safe_pickle import load_xz_pickle  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PICKLE = REPO_ROOT / "data" / "data_resolving_discrepancies" / "experiment_results.pickle.xz"
# SHA-256 of the xz file as committed in this repo; load_xz_pickle refuses a file with another digest.
PICKLE_SHA256 = "eae02848c22e72f90d1a08fb4d3f9782941563fd70636a33ce30692ec709bc34"
C_REF = 1e21
MIN_STEPS = (1, 500, 1000, 2000)
EXP_CAPS = (4.0, 1.5)      # unconstrained / smooth power laws only
PRIMARY_MIN_STEPS = 1000
TIE_TOL = 1e-6             # |difference in held-out RMSE| at or below this is a tie (same tolerance as E2 and E3)


def load_runs() -> pd.DataFrame:
    df = load_xz_pickle(PICKLE, sha256=PICKLE_SHA256).copy()
    df["L"] = [float(np.asarray(s)[-1]) for s in df["val/loss"]]
    df["last_idx"] = [float(s.index[-1]) for s in df["val/loss"]]
    df["D_repo"] = df.max_step.astype(float) * df.seq_len.astype(float) * df.bs.astype(float)
    df["D_last"] = df.last_idx * df.seq_len.astype(float) * df.bs.astype(float)
    pm = df.ckpt_path.str.extract(r"params=([0-9.]+)M")[0].astype(float) * 1e6
    lab = pd.DataFrame({"width": df.width, "depth": df.depth, "p": pm}).dropna().drop_duplicates(["width", "depth"])
    lab_map = {(w, d): p for w, d, p in lab.itertuples(index=False)}
    missing = sorted({(w, d) for w, d in zip(df.width, df.depth) if (w, d) not in lab_map})
    if missing:
        raise ValueError(f"no 'params=...M' label in any checkpoint path for (width, depth) = {missing[:5]}"
                         f"{' ...' if len(missing) > 5 else ''}; the exact parameter count cannot be looked up for those runs")
    df["N_ckpt"] = [lab_map[(w, d)] for w, d in zip(df.width, df.depth)]
    df["N_wxd"] = df.width * df.depth
    df["N_12ld2"] = 12.0 * df.depth * df.width ** 2
    df["N_12ld2emb"] = df["N_12ld2"] + df.vocab_size * df.width
    return df


PROXIES = {"wxd": "N_wxd", "ckpt": "N_ckpt", "12ld2": "N_12ld2", "12ld2+emb": "N_12ld2emb"}


def compute_opt(res: se.FitResult, C: float = C_REF):
    A, B = float(res.c[0]), float(res.c[1])
    a, b = float(res.e[0]), float(res.e[1])
    a_opt = b / (a + b)
    G = (a * A / (b * B)) ** (1.0 / (a + b))
    N_opt = G * (C / 6.0) ** a_opt
    D_opt = (1.0 / G) * (C / 6.0) ** (1.0 - a_opt)
    return a_opt, N_opt, D_opt


def write_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, model size) of the primary subset: the exact parameter label and how many runs it has.
    The report quotes the size range and the small-model counts from this table instead of from memory."""
    rows = []
    for ds in ("rw", "owt2"):
        sub = df[(df.dataset == ds) & (df.decay == "chinchilla") & (df.max_step >= PRIMARY_MIN_STEPS)]
        for (w, d, n_ckpt), g in sub.groupby(["width", "depth", "N_ckpt"]):
            rows.append(dict(dataset=ds, width=int(w), depth=int(d), N_ckpt=float(n_ckpt), n_runs=len(g)))
    out = pd.DataFrame(rows).sort_values(["dataset", "N_ckpt"]).reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "e4_sizes.csv", index=False)
    return out


def part_a(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in ("rw", "owt2"):
        for ms in MIN_STEPS:
            sub = df[(df.dataset == ds) & (df.decay == "chinchilla") & (df.max_step >= ms)]
            for dname in (("D_repo", "D_last") if ms == PRIMARY_MIN_STEPS else ("D_repo",)):
                for pname, col in PROXIES.items():
                    N, D, L = sub[col].values, sub[dname].values, sub.L.values
                    r = se.fit(se.JOINT, N, L, D, n_starts=300, n_refine=6, seed=11)
                    a_opt, N_opt, D_opt = compute_opt(r)
                    rows.append(dict(dataset=ds, min_steps=ms, D_def=dname, N_proxy=pname, n=len(sub), E=r.E,
                                     A=float(r.c[0]), alpha=float(r.e[0]), B=float(r.c[1]), beta=float(r.e[1]),
                                     huber=r.objective, rss=r.rss, a_opt=a_opt, N_opt_1e21=N_opt, D_opt_1e21=D_opt,
                                     DoverN_1e21=D_opt / N_opt, flags=se.diagnose(r, L)))
                    print(f"[E4a] {ds} min_steps={ms} {dname} {pname} (n={len(sub)}): {se.describe(r)} | huber {r.objective:.4g} a_opt {a_opt:.3f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "e4_n_proxy.csv", index=False)
    return out


def part_b(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    specs = [("joint", se.JOINT, None), ("+N dark", se.JOINT_N_DARK, "joint"),
             ("+D dark", se.JOINT_D_DARK, "joint"), ("+both dark", se.JOINT_BOTH_DARK, "+N dark")]
    for ds in ("rw", "owt2"):
        sub = df[(df.dataset == ds) & (df.decay == "chinchilla") & (df.max_step >= PRIMARY_MIN_STEPS)].reset_index(drop=True)
        sizes = np.sort(sub.N_ckpt.unique())
        k = len(sizes)
        cut_a, cut_b = sizes[int(round(0.67 * k)) - 1], sizes[int(round(0.75 * k)) - 1]
        C = 6 * sub.N_ckpt.values * sub.D_repo.values
        masks = {
            f"N: train sizes <= {cut_a / 1e6:.0f}M": sub.N_ckpt.values <= cut_a,
            f"N: train sizes <= {cut_b / 1e6:.0f}M": sub.N_ckpt.values <= cut_b,
            "D: lowest 75% tokens": sub.D_repo.values <= np.quantile(sub.D_repo, 0.75),
            "C: lowest 75% compute": C <= np.quantile(C, 0.75),
        }
        for split, mask in masks.items():
            tr, te = sub[mask], sub[~mask]
            for em in EXP_CAPS:
                prev = {}
                for name, spec, warm_name in specs:
                    res = se.fit(spec, tr.N_ckpt.values, tr.L.values, tr.D_repo.values, n_starts=200, n_refine=4, seed=21,
                                 warm=prev.get(warm_name) if warm_name else None, max_nfev=1500, exp_max=em)
                    prev[name] = res
                    r_te = se.log_resid(res, te.N_ckpt.values, te.L.values, te.D_repo.values)
                    rows.append(dict(dataset=ds, split=split, exp_max=em, n_train=len(tr), n_test=len(te), model=name,
                                     n_params=spec.n_params, train_huber=res.objective,
                                     test_rmse=float(np.sqrt(np.mean(r_te ** 2))), test_bias=float(np.mean(r_te)),
                                     flags=se.diagnose(res, tr.L.values, em), desc=se.describe(res)))
            print(f"[E4b] {ds} | {split}: done ({len(tr)}/{len(te)})", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "e4_backtest.csv", index=False)
    return out


def part_c(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds in ("rw", "owt2"):
        sub = df[(df.dataset == ds) & (df.decay == "chinchilla") & (df.max_step >= PRIMARY_MIN_STEPS)]
        for n_min in (0.0, 3e7, 6e7, 1.2e8):
            s2 = sub[sub.N_ckpt >= n_min]
            N, D, L = s2.N_ckpt.values, s2.D_repo.values, s2.L.values
            j = se.fit(se.JOINT, N, L, D, n_starts=250, n_refine=5, seed=41, exp_max=1.5)
            nd = se.fit(se.JOINT_N_DARK, N, L, D, n_starts=250, n_refine=5, seed=42, warm=j, max_nfev=2500, exp_max=1.5)
            rows.append(dict(dataset=ds, n_min=n_min, n_runs=len(s2), n_sizes=int(s2.N_ckpt.nunique()),
                             huber_joint=j.objective, huber_Ndark=nd.objective,
                             improvement_pct=100 * (1 - nd.objective / j.objective), flags_Ndark=se.diagnose(nd, L, 1.5),
                             desc_joint=se.describe(j), desc_Ndark=se.describe(nd)))
            print(f"[E4c] {ds} N>={n_min:.0e} ({len(s2)} runs): joint {j.objective:.4e} -> +N dark {nd.objective:.4e} "
                  f"({rows[-1]['improvement_pct']:.1f}% lower) | {se.describe(nd)} | flags {rows[-1]['flags_Ndark'] or 'none'}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "e4_small_n_bend.csv", index=False)
    return out


def _c2_job(args):
    ds, pname, n_min, N, D, L = args
    j = se.fit(se.JOINT, N, L, D, n_starts=250, n_refine=5, seed=41, exp_max=1.5)
    nd = se.fit(se.JOINT_N_DARK, N, L, D, n_starts=250, n_refine=5, seed=42, warm=j, max_nfev=2500, exp_max=1.5)
    return dict(dataset=ds, N_proxy=pname, n_min_ckpt=n_min, n_runs=len(N), huber_joint=j.objective, huber_Ndark=nd.objective,
                improvement_pct=100 * (1 - nd.objective / j.objective), flags_Ndark=se.diagnose(nd, L, 1.5),
                a_opt_joint=compute_opt(j)[0], desc_joint=se.describe(j), desc_Ndark=se.describe(nd))


def part_c2(df: pd.DataFrame, workers: int = 2) -> pd.DataFrame:
    jobs = []
    for ds in ("rw", "owt2"):
        sub = df[(df.dataset == ds) & (df.decay == "chinchilla") & (df.max_step >= PRIMARY_MIN_STEPS)]
        for pname, col in PROXIES.items():
            for n_min in (0.0, 3e7, 6e7, 1.2e8):
                s2 = sub[sub.N_ckpt >= n_min]          # same run subset for every proxy
                jobs.append((ds, pname, n_min, s2[col].values, s2.D_repo.values, s2.L.values))
    with Pool(workers) as pool:
        rows = pool.map(_c2_job, jobs, chunksize=1)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "e4_small_n_bend_by_proxy.csv", index=False)
    for r in rows:
        print(f"[E4c2] {r['dataset']} {r['N_proxy']:<10s} runs with exact N >= {r['n_min_ckpt']:.0e} ({r['n_runs']}): "
              f"+N dark {r['improvement_pct']:.2f}% lower | flags {r['flags_Ndark'] or 'none'}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default="a,b,c,c2", help="comma-separated parts to (re)run; the rest are read from results/")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    parts = set(args.parts.split(","))
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_runs()
    print(f"[E4] {len(df)} runs loaded; (width,depth)->params map has {df[['width','depth']].drop_duplicates().shape[0]} entries", flush=True)
    write_sizes(df)
    a = part_a(df) if "a" in parts else pd.read_csv(OUT / "e4_n_proxy.csv", keep_default_na=False)
    b = part_b(df) if "b" in parts else pd.read_csv(OUT / "e4_backtest.csv", keep_default_na=False)
    c = part_c(df) if "c" in parts else pd.read_csv(OUT / "e4_small_n_bend.csv", keep_default_na=False)
    c2 = part_c2(df, args.workers) if "c2" in parts else pd.read_csv(OUT / "e4_small_n_bend_by_proxy.csv", keep_default_na=False)
    summ = {}
    for (ds, ms, dd), g in a.groupby(["dataset", "min_steps", "D_def"]):
        ref = g[g.N_proxy == "ckpt"].iloc[0]
        summ[f"{ds}|min_steps={ms}|{dd}"] = {
            r.N_proxy: dict(n=int(r.n), alpha=r.alpha, beta=r.beta, a_opt=r.a_opt, huber=r.huber, E=r.E,
                            DoverN_1e21=r.DoverN_1e21, d_alpha_vs_ckpt=r.alpha - ref.alpha, d_aopt_vs_ckpt=r.a_opt - ref.a_opt)
            for r in g.itertuples()
        }
    for em in EXP_CAPS:
        sub = b[b.exp_max == em]
        piv = sub.pivot_table(index=["dataset", "split"], columns="model", values="test_rmse")
        key = f"backtest cap={em}"
        diff = {m: (piv[m] - piv["joint"]).values for m in ("+N dark", "+D dark", "+both dark")}
        summ[key] = dict(
            test_rmse={f"{k[0]} | {k[1]}": {m: float(v) for m, v in row.items()} for k, row in piv.iterrows()},
            # [wins, ties, losses] against the joint fit over the 8 splits; a difference within TIE_TOL is a tie (a strict
            # comparison would count a fit that merely switches its extra term off as a win by optimizer noise)
            wins_ties_losses_vs_joint={m: [int((d < -TIE_TOL).sum()), int((np.abs(d) <= TIE_TOL).sum()), int((d > TIE_TOL).sum())]
                                       for m, d in diff.items()},
            tie_tol=TIE_TOL, n_splits=int(len(piv)),
            median_rmse={m: float(piv[m].median()) for m in piv.columns},
            boundary_flag_counts={m: int((sub[sub.model == m]["flags"] != "").sum()) for m in piv.columns},
            median_train_huber={m: float(sub[sub.model == m].train_huber.median()) for m in piv.columns},
        )
    summ["small_n_bend"] = {f"{r.dataset}|N>={r.n_min:.0e}": dict(n_runs=int(r.n_runs), n_sizes=int(r.n_sizes), improvement_pct=float(r.improvement_pct), flags=r.flags_Ndark, desc=r.desc_Ndark) for r in c.itertuples()}
    summ["small_n_bend_by_proxy"] = {f"{r.dataset}|{r.N_proxy}|exactN>={r.n_min_ckpt:.0e}": dict(n_runs=int(r.n_runs), improvement_pct=float(r.improvement_pct), flags=r.flags_Ndark, a_opt_joint=float(r.a_opt_joint)) for r in c2.itertuples()}
    (OUT / "e4_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
