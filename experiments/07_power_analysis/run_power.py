"""How large a ladder has to be before a slow B N^{-beta} term shows up in held-out RMSE.

Truth is fixed from experiment 05 (separated exponents). The detection rule is
fixed here, before any cell is scored:

  detectable if win rate >= 0.80 and median (single RMSE - two-term RMSE) >= 0.01 nats

over replicates where both fits succeed and every held-out prediction is positive.
A win is two-term test RMSE strictly below the single-term test RMSE.
Alpha is the fast exponent, beta < alpha is the slow one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from py_src.conjecture_fits import fit_power, holdout_count, predict  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
N_SIZES = [8, 16, 24, 36, 48, 64]
SPANS = [2, 3, 4]  # log10(N_max / N_min), N_min = 1e8
SIGMAS = [0.005, 0.01, 0.02, 0.05]
N_REPS = 16
WIN_RATE_MIN = 0.80
GAP_MIN = 0.01  # nats
TRUTH = {"E": 1.5, "A": 12649.07, "alpha": 0.50, "B": 25.24, "beta": 0.20}


def _mean(n: np.ndarray) -> np.ndarray:
    return TRUTH["E"] + TRUTH["A"] * n ** (-TRUTH["alpha"]) + TRUTH["B"] * n ** (-TRUTH["beta"])


def _one(n_sizes: int, span: int, sigma: float, rep: int) -> dict:
    n = np.geomspace(1e8, 1e8 * 10.0 ** span, n_sizes)
    k = holdout_count(n_sizes)
    rng = np.random.default_rng(n_sizes * 1_000_003 + span * 10_007 + int(sigma * 1000) * 97 + rep)
    y = _mean(n) * np.exp(rng.normal(0.0, sigma, size=n_sizes))
    y_tr, y_te = y[:-k], y[-k:]
    n_tr, n_te = n[:-k], n[-k:]
    assert n_tr.max() < n_te.min()
    row = {
        "n_sizes": n_sizes,
        "log10_span": span,
        "sigma": sigma,
        "rep": rep,
        "n_train": int(len(n_tr)),
        "n_test": int(len(n_te)),
        "dof_two_term": int(len(n_tr) - 5),
    }
    for kind in ("single_term", "two_term"):
        fit = fit_power(kind, n_tr, y_tr, n_starts=6, seed=rep)
        yhat = predict(kind, fit.params, N=n_te) if not fit.failed else np.full(len(n_te), np.nan)
        ok = bool(fit.success) and np.all(np.isfinite(yhat)) and np.all(yhat > 0)
        err = yhat - y_te
        row[f"{kind}_ok"] = ok
        row[f"{kind}_rmse"] = float(np.sqrt(np.mean(err ** 2))) if ok else np.nan
        row[f"{kind}_collapsed"] = bool(fit.diagnostics.get("collapsed_to_single", False))
    if row["single_term_ok"] and row["two_term_ok"]:
        row["gap"] = row["single_term_rmse"] - row["two_term_rmse"]
        row["two_wins"] = bool(row["two_term_rmse"] < row["single_term_rmse"])
    else:
        row["gap"] = np.nan
        row["two_wins"] = False
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    cells = [(n, s, sig) for n in N_SIZES for s in SPANS for sig in SIGMAS]
    for i, (n, span, sigma) in enumerate(cells, start=1):
        for rep in range(N_REPS):
            rows.append(_one(n, span, sigma, rep))
        print(f"cell {i}/{len(cells)} n={n} span={span} sigma={sigma}", flush=True)
    reps = pd.DataFrame(rows)
    reps.to_csv(OUT / "replicates.csv", index=False)

    summary = []
    for (n, span, sigma), g in reps.groupby(["n_sizes", "log10_span", "sigma"]):
        usable = g[g["single_term_ok"] & g["two_term_ok"]]
        win = float(usable["two_wins"].mean()) if len(usable) else np.nan
        med_gap = float(usable["gap"].median()) if len(usable) else np.nan
        detected = bool(len(usable) == N_REPS and win >= WIN_RATE_MIN and med_gap >= GAP_MIN)
        summary.append(
            {
                "n_sizes": int(n),
                "log10_span": int(span),
                "sigma": float(sigma),
                "n_reps_ok": int(len(usable)),
                "n_train": int(g["n_train"].iloc[0]),
                "dof_two_term": int(g["dof_two_term"].iloc[0]),
                "win_rate": win,
                "median_gap_nats": med_gap,
                "mean_gap_nats": float(usable["gap"].mean()) if len(usable) else np.nan,
                "median_single_rmse": float(usable["single_term_rmse"].median()) if len(usable) else np.nan,
                "median_two_rmse": float(usable["two_term_rmse"].median()) if len(usable) else np.nan,
                "two_collapsed_rate": float(g["two_term_collapsed"].mean()),
                "detectable": detected,
            }
        )
    summary_df = pd.DataFrame(summary).sort_values(["sigma", "log10_span", "n_sizes"])
    summary_df.to_csv(OUT / "power_grid.csv", index=False)

    # Smallest n that is detectable, per span and noise. Empty if none.
    frontiers = []
    for (span, sigma), g in summary_df.groupby(["log10_span", "sigma"]):
        hit = g[g["detectable"]].sort_values("n_sizes")
        frontiers.append(
            {
                "log10_span": int(span),
                "sigma": float(sigma),
                "min_n_detectable": int(hit["n_sizes"].iloc[0]) if len(hit) else np.nan,
                "n_grid_max": int(g["n_sizes"].max()),
            }
        )
    pd.DataFrame(frontiers).to_csv(OUT / "detection_frontier.csv", index=False)

    fig, axes = plt.subplots(1, len(SPANS), figsize=(11.2, 3.6), sharey=True)
    for ax, span in zip(axes, SPANS):
        sub = summary_df[summary_df["log10_span"] == span]
        mat = sub.pivot(index="sigma", columns="n_sizes", values="win_rate").sort_index()
        im = ax.imshow(mat.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels(mat.columns)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels(mat.index)
        ax.set_xlabel("ladder sizes")
        ax.set_title(f"N spans 10^{span}")
        for i, sig in enumerate(mat.index):
            for j, n in enumerate(mat.columns):
                val = mat.loc[sig, n]
                mark = summary_df[(summary_df.log10_span == span) & (summary_df.sigma == sig) & (summary_df.n_sizes == n)]
                star = "*" if bool(mark["detectable"].iloc[0]) else ""
                ax.text(j, i, f"{val:.2f}{star}", ha="center", va="center", color="white", fontsize=7)
    axes[0].set_ylabel("log-noise σ")
    fig.colorbar(im, ax=axes, fraction=0.02, label="P(two-term RMSE < single)")
    fig.suptitle(f"* = detectable (win rate ≥ {WIN_RATE_MIN:.2f} and median gap ≥ {GAP_MIN} nats)")
    fig.tight_layout()
    fig.savefig(OUT / "win_rate_grid.png", dpi=130)
    plt.close(fig)
    print(summary_df.to_string(index=False))
    print(pd.DataFrame(frontiers).to_string(index=False))


if __name__ == "__main__":
    main()
