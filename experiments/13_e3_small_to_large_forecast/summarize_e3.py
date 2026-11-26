"""E3 summaries and verdict (PREREG.md): >= 10x pairs, largest third, ratio-bin table, figure.

    python summarize_e3.py     # reads results/folds.csv; writes results/{at10x,largest_third,largest_third_pooled,
                               # ratio_bins,verdict}.csv and results/e3_error_vs_ratio.png
"""
import math
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
ARMS = ["one", "drop1", "drop2", "lrt", "e1_strict", "e1_common", "e1_lrt"]
B = 2000
SEED = 20261127
BINS = [(1, 3), (3, 10), (10, 30), (30, np.inf)]


def paired(df, arm, base):
    a = df[df.arm == arm][["ladder", "k", "target_rank", "ratio", "abs_err", "err", "defined"]]
    b = df[df.arm == base][["ladder", "k", "target_rank", "abs_err", "err", "defined"]]
    m = a.merge(b, on=["ladder", "k", "target_rank"], suffixes=("", "_base"))
    return m[(m.defined == 1) & (m.defined_base == 1)]


def cluster_ci(m, rng):
    """95% interval of the median gain, resampling folds (k) with replacement."""
    ks = m.k.unique()
    groups = {k: (m.abs_err_base - m.abs_err)[m.k == k].values for k in ks}
    meds = []
    for _ in range(B):
        pick = rng.choice(ks, len(ks), replace=True)
        meds.append(np.median(np.concatenate([groups[k] for k in pick])))
    return np.percentile(meds, [2.5, 97.5])


def at10(df):
    rng = np.random.default_rng(SEED)
    rows = []
    for lad in sorted(df.ladder.unique()):
        d = df[(df.ladder == lad) & (df.ratio >= 10)]
        for arm in ARMS[1:]:
            m = paired(d, arm, "one")
            if not len(m):
                rows.append(dict(ladder=lad, arm=arm, pairs=0, folds=0))
                continue
            gain = m.abs_err_base - m.abs_err
            lo, hi = cluster_ci(m, rng)
            rows.append(dict(ladder=lad, arm=arm, pairs=len(m), folds=m.k.nunique(), med_abs_arm=m.abs_err.median(),
                             med_abs_one=m.abs_err_base.median(), mean_abs_arm=m.abs_err.mean(),
                             mean_abs_one=m.abs_err_base.mean(), med_gain=gain.median(), gain_ci_lo=lo,
                             gain_ci_hi=hi, wins=int((gain > 0).sum()), losses=int((gain < 0).sum()),
                             ties=int((gain == 0).sum()), med_signed_err_arm=m.err.median()))
    return pd.DataFrame(rows)


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def largest_third(df):
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for lad in sorted(df.ladder.unique()):
        d = df[df.ladder == lad]
        n = int(d.n_sizes.iloc[0])
        kstar = n - math.ceil(n / 3)
        f = d[d.k == kstar]
        base = {b: f[f.arm == b].sort_values("target_rank") for b in ("one", "drop2")}
        for arm in ARMS:
            a = f[f.arm == arm].sort_values("target_rank")
            row = dict(ladder=lad, arm=arm, k_fit=kstar, targets=len(a), defined=int(a.defined.all()))
            if a.defined.all():
                row["rmse"] = rmse(a.err)
                for b, fb in base.items():
                    if fb.defined.all() and arm != b:
                        ea, eb = a.err.values, fb.err.values
                        idx = rng.integers(0, len(ea), (B, len(ea)))
                        diff = np.sqrt((ea[idx] ** 2).mean(1)) - np.sqrt((eb[idx] ** 2).mean(1))
                        row[f"rmse_minus_{b}"] = rmse(ea) - rmse(eb)
                        row[f"rmse_minus_{b}_ci_lo"], row[f"rmse_minus_{b}_ci_hi"] = np.percentile(diff, [2.5, 97.5])
            rows.append(row)
    return pd.DataFrame(rows)


def largest_third_pooled(df):
    rows = []
    for lad in sorted(df.ladder.unique()):
        d = df[df.ladder == lad]
        n = int(d.n_sizes.iloc[0])
        d = d[d.target_rank >= n - math.ceil(n / 3)]
        for arm in ARMS:
            m = paired(d, arm, "one") if arm != "one" else d[(d.arm == "one")].assign(err_base=d.err)
            if len(m):
                rows.append(dict(ladder=lad, arm=arm, pairs=len(m), rmse_arm=rmse(m.err), rmse_one=rmse(m.err_base)))
    return pd.DataFrame(rows)


def ratio_bins(df):
    rows = []
    for (lad, arm), d in df[df.defined == 1].groupby(["ladder", "arm"]):
        row = dict(ladder=lad, arm=arm)
        for lo, hi in BINS:
            x = d[(d.ratio >= lo) & (d.ratio < hi)].abs_err
            row[f"med_{lo}-{hi}"] = x.median() if len(x) else np.nan
            row[f"n_{lo}-{hi}"] = len(x)
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(A, T, arm):
    a = A[(A.arm == arm) & (A.pairs > 0)]
    pos = int((a.med_gain > 0).sum())
    medg = float(a.med_gain.median()) if len(a) else np.nan
    t = T[(T.arm == arm) & T.rmse_minus_drop2.notna()] if "rmse_minus_drop2" in T else T.iloc[:0]
    vs_drop2 = float(t.rmse_minus_drop2.median()) if len(t) else np.nan
    c1, c2, c3 = pos >= 3, len(a) > 0 and medg >= 0.005, len(t) > 0 and vs_drop2 <= 0.005
    if len(a) < 3:
        v = "NOT ESTABLISHED (insufficient ladders at >= 10x)"
    elif c1 and c2 and c3:
        v = "HELPS"
    else:
        why = [s for s, ok in (("(i) fewer than 3 ladders gain", c1), ("(ii) median gain < 0.005", c2),
                                ("(iii) worse than drop2 on the largest third", c3)) if not ok]
        v = "NOT ESTABLISHED: " + "; ".join(why)
    return dict(arm=arm, ladders_at_10x=len(a), ladders_with_gain=pos, ladders_with_loss=int((a.med_gain < 0).sum()),
                median_ladder_gain=medg, ladders_vs_drop2=len(t), median_rmse_minus_drop2=vs_drop2,
                cond_i=c1, cond_ii=c2, cond_iii=c3, verdict=v)


def figure(df):
    lads = sorted(df.ladder.unique())
    fig, axs = plt.subplots(1, len(lads), figsize=(4.2 * len(lads), 3.8), sharey=False)
    centers = [math.sqrt(lo * (hi if np.isfinite(hi) else 100)) for lo, hi in BINS]
    for ax, lad in zip(axs, lads):
        d = df[(df.ladder == lad) & (df.defined == 1)]
        for arm, mk in zip(ARMS, "osD^v*x"):
            ys = []
            for lo, hi in BINS:
                x = d[(d.arm == arm) & (d.ratio >= lo) & (d.ratio < hi)].abs_err
                ys.append(x.median() if len(x) else np.nan)
            ax.plot(centers, ys, marker=mk, label=arm)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(lad)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("extrapolation ratio N_pred / N_max_fit (bin)")
        ax.grid(True)
    axs[0].set_ylabel("median |error| (nats)")
    axs[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e3_error_vs_ratio.png"), dpi=130)


def main():
    df = pd.read_csv(os.path.join(OUT, "folds.csv"))
    A, T = at10(df), largest_third(df)
    A.to_csv(os.path.join(OUT, "at10x.csv"), index=False)
    T.to_csv(os.path.join(OUT, "largest_third.csv"), index=False)
    largest_third_pooled(df).to_csv(os.path.join(OUT, "largest_third_pooled.csv"), index=False)
    R = ratio_bins(df)
    R.to_csv(os.path.join(OUT, "ratio_bins.csv"), index=False)
    V = pd.DataFrame([verdict(A, T, "e1_strict"), verdict(A, T, "e1_common")])
    V.to_csv(os.path.join(OUT, "verdict.csv"), index=False)
    figure(df)
    pd.set_option("display.width", 250)
    print(A.round(4).to_string())
    print(T.round(4).to_string())
    print(R.round(4).to_string())
    print(V.to_string())


if __name__ == "__main__":
    main()
