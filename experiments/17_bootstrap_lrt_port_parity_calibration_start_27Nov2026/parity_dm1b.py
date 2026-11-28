#!/usr/bin/env python3
"""Criteria 1 and 2 of the folder 17 README: is py_src/two_term_lrt.py DM-1b, bit for bit?

1. Source identity: the block between the BEGIN and END markers in py_src/two_term_lrt.py must equal
   experiments/dm1_power_analysis/dm1b.py's fitter block (_proj .. fit_chunked) after three renames.
2. Row parity: regenerate sampled DM-1b replicates from their seeds, run bootstrap_lr_test with DM-1b's early stop, and
   compare with the rows committed in results_dm1b/raw_dm1b_shard000_of001.csv.

    python parity_dm1b.py [--n-per-s 40] [--n-large 20]
Writes results/parity.csv and results/parity_summary.json; exits 1 if any criterion fails.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "dm1_power_analysis"))
import dm1  # noqa: E402  (DM-1's generator and grids)
from py_src import two_term_lrt as T  # noqa: E402

SEED = 20261127
RAW = REPO / "experiments" / "dm1_power_analysis" / "results_dm1b" / "raw_dm1b_shard000_of001.csv"
BEGIN = "# --- BEGIN verbatim port"
END = "# --- END verbatim port"
RENAMES = [("dm1.ALPHA_GRID", "ALPHA_GRID"), ("dm1.A2_GRID", "A2_GRID"), ("dm1.D2_GRID", "D2_GRID")]


def source_identity():
    src = (REPO / "experiments" / "dm1_power_analysis" / "dm1b.py").read_text().splitlines()
    a = next(i for i, l in enumerate(src) if l.startswith("def _proj("))
    b = next(i for i, l in enumerate(src) if l.startswith("def run_cell("))
    expect = "\n".join(src[a:b]).rstrip()
    for old, new in RENAMES:
        expect = expect.replace(old, new)
    port = (REPO / "py_src" / "two_term_lrt.py").read_text().splitlines()
    i = next(k for k, l in enumerate(port) if l.startswith(BEGIN)) + 1
    j = next(k for k, l in enumerate(port) if l.startswith(END))
    got = "\n".join(port[i:j]).rstrip()
    return got == expect, len(expect.splitlines())


def sample_rows(raw, n_per_s, n_large):
    rng = np.random.default_rng(SEED)
    t = raw[raw.testable == 1]
    picks = []
    for s in sorted(t.s.unique()):
        pool = t[(t.s == s) & (t.npts <= 17)]
        picks.append(pool.iloc[np.sort(rng.choice(len(pool), n_per_s, replace=False))])
    big = t[t.npts > 17]
    picks.append(big.iloc[np.sort(rng.choice(len(big), n_large, replace=False))])
    return pd.concat(picks)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-per-s", type=int, default=40)
    ap.add_argument("--n-large", type=int, default=20)
    args = ap.parse_args()
    out = HERE / "results"
    out.mkdir(exist_ok=True)

    same, nlines = source_identity()
    print(f"[1] source identity: {'IDENTICAL' if same else 'DIFFERENT'} ({nlines} lines compared)", flush=True)

    raw = pd.read_csv(RAW)
    rows = sample_rows(raw, args.n_per_s, args.n_large)
    cells = {c["cell"]: c for c in dm1.cells()}
    recs = []
    t0 = time.time()
    for _, ref in rows.iterrows():
        cell, rep = int(ref.cell), int(ref.rep)
        c = cells[cell]
        for key in ("nmin", "nmax", "ppd", "sigma", "s", "beta"):
            assert float(ref[key]) == float(c[key]), f"cell {cell}: committed {key} differs from dm1.cells()"
        N = dm1.ladder(c["nmin"], c["nmax"], c["ppd"])
        n = N / dm1.N_REF
        y = dm1.law(n, dm1.E0, dm1.A_SCALED, dm1.ALPHA0, c["s"] * dm1.A_SCALED, c["beta"]) + \
            np.random.default_rng([cell, rep, 20261019]).normal(0, c["sigma"], len(n))
        r = T.bootstrap_lr_test(N, y, B=200, level=0.05, early_stop=True,
                                rng=np.random.default_rng([cell, rep, 20261019, 1]))
        recs.append(dict(
            cell=cell, rep=rep, npts=len(n), s=c["s"], T_lib=r.T, T_ref=float(ref["T"]), k_lib=r.k_exceed,
            k_ref=int(ref.k_exceed), bused_lib=r.B_used, bused_ref=int(ref.b_used), p_lib=r.p_value,
            p_ref=float(ref.p_lrt), det_lib=int(r.detect), det_ref=int(ref.detect_lrt),
            rss1_lib=r.one_term.rss, rss1_ref=float(ref.rss1), rss2_lib=r.two_term.rss, rss2_ref=float(ref.rss2),
            nonfinite_boot=r.n_nonfinite_boot))
    df = pd.DataFrame(recs)
    rel = lambda a, b: np.abs(a - b) / np.maximum(1.0, np.abs(b))  # noqa: E731
    df["T_ok"] = rel(df.T_lib, df.T_ref) <= 1e-9
    df["p_ok"] = np.abs(df.p_lib - df.p_ref) <= 1e-12
    df["k_ok"] = df.k_lib == df.k_ref
    df["b_ok"] = df.bused_lib == df.bused_ref
    df["det_ok"] = df.det_lib == df.det_ref
    df["all_ok"] = df[["T_ok", "p_ok", "k_ok", "b_ok", "det_ok"]].all(axis=1)
    df.to_csv(out / "parity.csv", index=False)
    summary = dict(
        source_identical=bool(same), source_lines_compared=nlines, rows=len(df), rows_all_ok=int(df.all_ok.sum()),
        k_mismatch=int((~df.k_ok).sum()), b_mismatch=int((~df.b_ok).sum()), det_mismatch=int((~df.det_ok).sum()),
        T_mismatch=int((~df.T_ok).sum()), p_mismatch=int((~df.p_ok).sum()),
        max_T_rel_diff=float(rel(df.T_lib, df.T_ref).max()), max_rss1_rel_diff=float(rel(df.rss1_lib, df.rss1_ref).max()),
        max_rss2_rel_diff=float(rel(df.rss2_lib, df.rss2_ref).max()),
        rows_with_nonfinite_bootstrap=int((df.nonfinite_boot > 0).sum()),
        detections_lib=int(df.det_lib.sum()), detections_ref=int(df.det_ref.sum()),
        rows_by_s={str(k): int(v) for k, v in df.groupby("s").size().items()},
        rows_large_ladders=int((df.npts > 17).sum()), seconds=round(time.time() - t0, 1),
        numpy=np.__version__, python=sys.version.split()[0], sample_seed=SEED,
        criteria_1_pass=bool(same), criteria_2_pass=bool(df.all_ok.all()))
    (out / "parity_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    sys.exit(0 if (same and df.all_ok.all()) else 1)


if __name__ == "__main__":
    main()
