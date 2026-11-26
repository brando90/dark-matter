"""E4 (PREREG.md, 11-25-2026): DM-1b bootstrap LRT for a second N term on every Porian convention x 3 counts,
plus the reverse-ladder residual at the smallest sizes.

    python e4.py --workers 8     # writes results/cells/*.json (cache, resumable), results/e4_cells.csv
"""
import argparse
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "14_e2_token_budget_hparams_small_end"))
import e2lib as L  # noqa: E402

B_BOOT = 1000
LEVEL = 0.05
SEED = 20261125
OUT = os.path.join(HERE, "results")


def lrt_full(n, y, rng, B=B_BOOT):
    """DM-1b's rule with all B refits run (no early stop): returns f1, f2 (DM-1b names), T, p, k, nbad."""
    f1, f2 = L.e1lib.fit2(n, y)
    npts = len(n)
    T = float(L.dm1b.lrt_stat(npts, f1["rss"], f2["rss"]))
    if not T > 0:
        return f1, f2, T, 1.0, B, 0
    sig = np.sqrt(f1["rss"] / (npts - 3))
    mu = f1["E"] + f1["a"] * n ** -f1["alpha"]
    Ys = mu[None, :] + sig * rng.normal(0, 1, (B, npts))
    g1, g2 = L.dm1b.fit_chunked(n, Ys)
    Ts = L.dm1b.lrt_stat(npts, g1["rss"], g2["rss"])
    bad = ~np.isfinite(Ts)
    k = int(((Ts >= T - 1e-9 * max(T, 1.0)) | bad).sum())
    return f1, f2, T, (1 + k) / (B + 1), k, int(bad.sum())


def bic(rss, npts, k):
    return npts * np.log(max(rss, 1e-300) / npts) + k * np.log(npts)


def cell_id(ds, rc, sl, cname):
    return f"{ds}_{rc}_{sl}_{cname}"


def run_cell(job):
    ds, rc, sl, cname, B = job
    path = os.path.join(OUT, "cells", cell_id(ds, rc, sl, cname) + ".json")
    if os.path.exists(path):
        return json.load(open(path))
    t0 = time.time()
    d = L.ladder(ds, rc, sl)
    N = L.count(L.COUNTS[cname], d.N_nonemb.values, d.width.values)
    o = np.argsort(N)
    d, N = d.iloc[o].reset_index(drop=True), N[o]
    y = d.y.values
    n = N / L.N_REF
    npts = len(y)
    row = dict(dataset=ds, recipe=rc, slice=str(sl), count=cname, npts=npts, N_min=float(N.min()), N_max=float(N.max()),
               widths=",".join(str(int(w)) for w in d.width), testable=int(npts > 6))
    if npts > 6:
        rng = np.random.default_rng([SEED, zlib.crc32(cell_id(ds, rc, sl, cname).encode())])
        f1, f2, T, p, k, nbad = lrt_full(n, y, rng, B)
        nm = L.e1lib.two_term_named(f2)
        share = (nm["A_fast"] * n.min() ** -nm["alpha_fast"]) / max(nm["B_slow"] * n.min() ** -nm["beta_slow"], 1e-300)
        row.update(T=T, p_lrt=p, k_exceed=k, B=B, boot_nonfinite=nbad, detect=int(p <= LEVEL),
                   E1=f1["E"], A1=f1["a"], alpha1=f1["alpha"], rss1=f1["rss"], sd1=float(np.sqrt(f1["rss"] / (npts - 3))),
                   E2=nm["E"], A_fast=nm["A_fast"], alpha_fast=nm["alpha_fast"], B_slow=nm["B_slow"],
                   beta_slow=nm["beta_slow"], rss2=nm["rss"], sd2=float(np.sqrt(nm["rss"] / (npts - 5))),
                   dbic=float(bic(f1["rss"], npts, 3) - bic(nm["rss"], npts, 5)),
                   fast_share_at_Nmin=float(share) if nm["A_fast"] > 0 else 0.0)
    else:
        row.update(detect=0)
    # reverse ladder: reference = tied N >= 1e8 (same models under every count)
    ref = d.N_tied.values >= L.REF_MIN
    r, f = L.reverse(N, y, ref)
    if r is None:
        row.update(rev_ok=0, rev_signature="n/a")
    else:
        small = ~ref
        rs = r[small]
        for w, lab in L.SMALL.items():
            m = d.width.values == w
            row[f"rev_{lab}"] = float(r[m][0]) if m.any() else np.nan
        frac = float((rs > 0).mean()) if len(rs) else np.nan
        mean = float(rs.mean()) if len(rs) else np.nan
        r5 = row["rev_5M"]
        if np.isfinite(r5) and r5 > 0 and frac >= 0.75 and mean > 0.02:
            sig = "POSITIVE"
        elif np.isfinite(r5) and r5 < 0 and frac <= 0.25 and mean < -0.02:
            sig = "NEGATIVE"
        else:
            sig = "FLAT"
        row.update(rev_ok=1, rev_nref=int(ref.sum()), rev_nsmall=int(small.sum()), rev_mean_small=mean,
                   rev_frac_pos=frac, rev_signature=sig, rev_E=f["E"], rev_A=f["a"], rev_alpha=f["p"])
    row["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(row, open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--B", type=int, default=B_BOOT)
    args = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    jobs = [(ds, rc, sl, c, args.B) for ds, rc, sl in L.conventions() for c in L.COUNTS]
    t0 = time.time()
    rows = []
    with Pool(min(args.workers, 8)) as pool:
        for i, r in enumerate(pool.imap_unordered(run_cell, jobs, chunksize=1)):
            rows.append(r)
            if i % 10 == 0:
                print(f"{i + 1}/{len(jobs)} cells, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows).sort_values(["count", "dataset", "recipe", "slice"])
    df.to_csv(os.path.join(OUT, "e4_cells.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
