"""E3 (PREREG.md, 11-25-2026): DM-5 rolling-origin small-to-large forecast with the E1 parameter-count arms.

For every ladder and k = 4..n-1: fit the smallest k sizes (by N in the c_std count), predict every larger size with
  one        1-term at c_std
  drop1/2    1-term at c_std without the smallest 1 / 2 fit sizes (k - d >= 4)
  lrt        DM-1b bootstrap-LRT-selected 2-term at c_std (k > 6; 1-term otherwise, as DM-5)
  e1_strict  1-term in N_c, c profiled on the training fold only (k >= 5)
  e1_common  1-term in N_c at E1's shared c (chosen on all sizes of all ladders: not a strict forecast)
  e1_lrt     LRT-selected 2-term in N_c at the e1_strict c (k > 6; optional arm, not in the verdict)

    python e3.py --workers 12      # needs ../12_e1_profiled_param_count/results/{ladders,e1_pooled}.csv
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.abspath(os.path.join(HERE, "..", "12_e1_profiled_param_count"))
sys.path.insert(0, E1)
import e1lib as L  # noqa: E402

SEED = 20261126
MIN_FIT, MIN_FIT_C = 4, 5
ARMS = ["one", "drop1", "drop2", "lrt", "e1_strict", "e1_common", "e1_lrt"]


def strict_c(ne, nemb, y):
    """c profiled on the given (training) sizes only: 0.05 grid over [0, 3] + bounded refinement."""
    r = [L.fit1(L.n_of(c, ne, nemb), y)["rss"] for c in L.GRID_005]
    return L.refine(lambda c: L.fit1(L.n_of(c, ne, nemb), y)["rss"], L.GRID_005, r)[0]


def run_fold(job):
    """One fold: rows (arm, target). job = (ladder, N_nonemb, N_emb, y, c_std, c_sh, k, jid)."""
    lad, ne, nemb, y, c_std, c_sh, k, jid = job
    ne, nemb, y = (np.asarray(v, float) for v in (ne, nemb, y))
    N_std = ne + c_std * nemb
    tr, tg = np.arange(k), np.arange(k, len(y))
    preds, info = {}, {}
    n_std = L.n_of(c_std, ne, nemb)
    r = L.lrt(n_std[tr], y[tr], [SEED, jid, 0])
    preds["one"] = L.pred1(dict(E=r["f1"]["E"], a=r["f1"]["a"], p=r["f1"]["alpha"]), n_std[tg])
    preds["lrt"] = L.pred2(r["f2"], n_std[tg]) if r["detect"] else preds["one"]
    info["lrt"] = dict(used2=r["detect"], p_lrt=r["p"], tested=r["tested"], c=c_std)
    info["one"] = dict(c=c_std)
    for d in (1, 2):
        if k - d >= MIN_FIT:
            preds[f"drop{d}"] = L.pred1(L.fit1(n_std[d:k], y[d:k]), n_std[tg])
        info[f"drop{d}"] = dict(c=c_std)
    nc = L.n_of(c_sh, ne, nemb)
    preds["e1_common"] = L.pred1(L.fit1(nc[tr], y[tr]), nc[tg])
    info["e1_common"] = dict(c=c_sh)
    if k >= MIN_FIT_C:
        cs = strict_c(ne[tr], nemb[tr], y[tr])
        n = L.n_of(cs, ne, nemb)
        preds["e1_strict"] = L.pred1(L.fit1(n[tr], y[tr]), n[tg])
        info["e1_strict"] = dict(c=cs)
        r2 = L.lrt(n[tr], y[tr], [SEED, jid, 1])
        preds["e1_lrt"] = L.pred2(r2["f2"], n[tg]) if r2["detect"] else preds["e1_strict"]
        info["e1_lrt"] = dict(c=cs, used2=r2["detect"], p_lrt=r2["p"], tested=r2["tested"])
    rows = []
    for arm in ARMS:
        p = preds.get(arm)
        for i, j in enumerate(tg):
            pr = float(p[i]) if p is not None else np.nan
            rows.append(dict(ladder=lad, k=k, n_sizes=len(y), arm=arm, defined=int(p is not None),
                             N_fit_max=float(N_std[k - 1]), N_pred=float(N_std[j]), target_rank=int(j),
                             ratio=float(N_std[j] / N_std[k - 1]), y=float(y[j]), pred=pr, err=pr - float(y[j]),
                             abs_err=abs(pr - float(y[j])), **{f"info_{a}": b for a, b in info.get(arm, {}).items()}))
    return rows


def jobs_for(lads, c_sh):
    jobs, jid = [], 0
    for lad, g in lads.items():
        for k in range(MIN_FIT, len(g)):
            jobs.append((lad, g.N_nonemb.values, g.N_emb.values, g.y.values, float(g.c_std.iloc[0]), c_sh, k, jid))
            jid += 1
    return jobs


def load():
    lad = pd.read_csv(os.path.join(E1, "results", "ladders.csv"))
    po = pd.read_csv(os.path.join(E1, "results", "e1_pooled.csv"))
    c_sh = float(po.loc[po.test == "primary_1term_all4", "c_sh"].iloc[0])
    return {k: g.sort_values("N_std").reset_index(drop=True) for k, g in lad.groupby("ladder", sort=False)}, c_sh


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "folds.csv"))
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    lads, c_sh = load()
    jobs = jobs_for(lads, c_sh)
    t0 = time.time()
    with Pool(min(a.workers, 16)) as pool:
        out = pool.map(run_fold, jobs, chunksize=1)
    df = pd.DataFrame([r for o in out for r in o])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(jobs)} folds, {len(df)} rows, c_sh = {c_sh:.4f}, {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
