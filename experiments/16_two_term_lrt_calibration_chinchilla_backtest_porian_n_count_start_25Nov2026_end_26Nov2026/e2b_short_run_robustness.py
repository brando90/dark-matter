#!/usr/bin/env python3
"""E2b (post hoc): does the E2 result survive dropping the very shortest Chinchilla runs?

Why this exists. The residuals of the joint fit on the 245 points are dominated by four runs with D/N < 0.35
(fewer than a third of a token per parameter): they carry about nine tenths of the residual sum of squares, and the repo's
own experiments/RESULTS.md (item 3) reports the same four runs. E2 found that an extra additive D term cut the median
held-out error on token extrapolation by about four fifths. This script repeats E2 parts A (twelve train-small /
test-large splits) and C (full-data fits) on subsets without those runs.

The thresholds were chosen AFTER seeing the residuals, so this is a labelled post hoc check, not a pre-planned test.
F1 drops D/N < 0.35 (the four runs); F2 drops D/N < 1 (every run with fewer than one token per parameter).

Outputs go to results/e2b/<filter>/ and never overwrite the E2 files. Run one filter per process (single-threaded):
    python3 -I e2b_short_run_robustness.py --filter f1_drop_DN_lt_0.35
    python3 -I e2b_short_run_robustness.py --filter f2_drop_DN_lt_1
Wins against the joint fit are tie-aware (see e2_chinchilla_joint_backtest.win_fields). To redo only the counts from a saved
backtest, without fitting:  python3 -I e2b_short_run_robustness.py --filter <name> --recount-only
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

import e2_chinchilla_joint_backtest as e2  # noqa: E402  (puts the repo root on sys.path and caps threads at one)

se = e2.se
FILTERS = {"f1_drop_DN_lt_0.35": 0.35, "f2_drop_DN_lt_1": 1.0}
MODELS = ("joint", "+N dark", "+D dark", "+both dark")


def summarize(df: pd.DataFrame, dropped: pd.DataFrame, bt: pd.DataFrame, full: pd.DataFrame, free) -> dict:
    a, b = float(free.e[0]), float(free.e[1])
    summ = dict(
        n_points=int(len(df)), n_dropped=int(len(dropped)),
        dropped=[dict(N=float(r.N), D=float(r.D), L=float(r.L), D_over_N=float(r.D / r.N)) for r in dropped.itertuples()],
        free_joint=dict(desc=se.describe(free), huber=float(free.objective), rss=float(free.rss), a_opt=b / (a + b),
                        N_exponent=a, D_exponent=b, E=float(free.E)),
        full_data_fits={f"cap={r.exp_max}|{r.model}": dict(huber=float(r.huber), rss=float(r.rss), flags=r.flags, desc=r.desc)
                        for r in full.itertuples()},
    )
    for em in e2.EXP_CAPS:
        sub = bt[bt.exp_max == em]
        piv = e2.split_table(bt, em)
        by_axis = {ax: {m: float(piv.loc[ax][m].median()) for m in piv.columns} for ax in ("N", "D", "C")}
        summ[f"cap={em}"] = dict(
            median_test_rmse={m: float(piv[m].median()) for m in piv.columns},
            median_test_rmse_by_axis=by_axis,
            boundary_flag_counts={m: int((sub[sub.model == m]["flags"] != "").sum()) for m in MODELS},
        )
        summ[f"cap={em}"].update(e2.win_fields(piv))      # tie-aware [wins, ties, losses] and the repo-versus-joint counts
    return summ


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", required=True, choices=list(FILTERS))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--recount-only", action="store_true",
                    help="rewrite the win counts in this filter's e2b_summary.json as tie-aware counts from its saved backtest CSV; no fitting")
    a = ap.parse_args()
    thr = FILTERS[a.filter]
    out = HERE / "results" / "e2b" / a.filter
    out.mkdir(parents=True, exist_ok=True)
    if a.recount_only:
        e2.recount(out / "e2b_summary.json", out / "e2_backtest.csv")
        print(f"[E2b {a.filter}] win counts recounted from the saved backtest", flush=True)
        return
    e2.OUT = out  # part_a and part_c write their CSVs through this global, so E2's own files stay untouched

    full_df = pd.read_csv(HERE / "results" / "chinchilla_245.csv")
    keep = (full_df.D / full_df.N) >= thr
    df, dropped = full_df[keep].reset_index(drop=True), full_df[~keep]
    print(f"[E2b {a.filter}] keep {len(df)} of {len(full_df)}; dropped {len(dropped)} runs with D/N < {thr}: "
          + ", ".join(f"N={r.N:.3g} D={r.D:.3g} (D/N {r.D / r.N:.2f})" for r in dropped.itertuples()), flush=True)

    ns, nr = (60, 2) if a.quick else (200, 4)
    t0 = time.time()
    free = se.fit(se.JOINT, df.N.values, df.L.values, df.D.values, n_starts=300, n_refine=6, seed=3)
    print(f"[E2b] free joint fit on the subset: {se.describe(free)} | huber {free.objective:.4e} | "
          f"a_opt {float(free.e[1]) / float(free.e[0] + free.e[1]):.4f} ({time.time() - t0:.0f}s)", flush=True)
    bt = e2.part_a(df, ns, nr)
    full = e2.part_c(df, ns, nr)
    summ = summarize(df, dropped, bt, full, free)
    (out / "e2b_summary.json").write_text(json.dumps(summ, indent=2))
    for em in e2.EXP_CAPS:
        s = summ[f"cap={em}"]
        print(f"[E2b {a.filter}] cap {em}: median held-out error by axis {json.dumps(s['median_test_rmse_by_axis'])}")
        print(f"[E2b {a.filter}] cap {em}: [wins, ties, losses] vs joint over {s['n_splits']} splits {s['wins_ties_losses_vs_joint']}; "
              f"repo vs joint {s['repo_vs_joint']}; boundary flags {s['boundary_flag_counts']}")
    print(f"[E2b {a.filter}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
