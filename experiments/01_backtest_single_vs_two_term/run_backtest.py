"""Held-out extrapolation: single-term vs two-term vs log correction vs broken power law.

Fit only on smaller N. Score MAE and RMSE on the largest held-out model sizes.
Primary metric is RMSE of the loss (nats) on the held-out sizes, at the largest
token budget that every model in the ladder shares. The mean of that RMSE across
token budgets is reported separately and is not an independent-sample standard error.
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

from py_src.conjecture_fits import (  # noqa: E402
    N_PARAMS,
    assert_self_check,
    fit_power,
    predict,
    regression_metrics,
)
from py_src.ladders import (  # noqa: E402
    chinchilla_matched,
    gpt3_binned,
    pythia_lambada,
    with_size_holdout,
)

OUT = Path(__file__).resolve().parent / "outputs"
KINDS = ["single_term", "two_term", "log_correction", "broken_power"]
COLORS = {
    "single_term": "#0072B2",
    "two_term": "#D55E00",
    "log_correction": "#009E73",
    "broken_power": "#CC79A7",
}
LABELS = {
    "single_term": r"single $E+AN^{-\alpha}$",
    "two_term": r"two-term $E+AN^{-\alpha}+BN^{-\beta}$",
    "log_correction": r"log correction",
    "broken_power": r"broken power",
}


def _fit_slice(dataset: str, sub: pd.DataFrame, pred_rows: list, fit_rows: list) -> None:
    train = sub[sub["split"] == "train"]
    test = sub[sub["split"] == "test"]
    if train["N"].nunique() < 4 or len(test) == 0:
        return
    if train["N"].max() >= test["N"].min():
        raise AssertionError(f"holdout overlap in {dataset}")
    d_id = sub["d_id"].iloc[0]
    for kind in KINDS:
        fit = fit_power(kind, train["N"].to_numpy(), train["loss"].to_numpy(), n_starts=12, seed=0)
        yhat_map = {}
        for split_name, part in (("train", train), ("test", test)):
            if fit.failed:
                yhat = np.full(len(part), np.nan)
            else:
                yhat = predict(kind, fit.params, N=part["N"].to_numpy())
            yhat_map[split_name] = yhat
            for i, (_, row) in enumerate(part.iterrows()):
                pred_rows.append(
                    {
                        "dataset": dataset,
                        "d_id": d_id,
                        "kind": kind,
                        "split": split_name,
                        "model_name": row["model_name"],
                        "N": row["N"],
                        "D": row["D"],
                        "C": row["C"],
                        "loss": row["loss"],
                        "prediction": float(yhat[i]),
                    }
                )
        tr_m = regression_metrics(train["loss"], yhat_map["train"])
        te_m = regression_metrics(test["loss"], yhat_map["test"])
        n_params = N_PARAMS[kind]
        fit_rows.append(
            {
                "dataset": dataset,
                "d_id": d_id,
                "kind": kind,
                "n_train": tr_m["n"],
                "n_test": te_m["n"],
                "n_params": n_params,
                "dof": tr_m["n"] - n_params,
                "saturated": tr_m["n"] - n_params <= 0,
                "train_rmse": tr_m["rmse"],
                "train_mae": tr_m["mae"],
                "test_rmse": te_m["rmse"],
                "test_mae": te_m["mae"],
                "test_log_rmse": te_m["log_rmse"],
                "test_log_mae": te_m["log_mae"],
                "n_nonpositive_pred_test": te_m["n_nonpositive_pred"],
                "objective": fit.objective,
                "mean_huber": fit.mean_huber,
                "optimizer_success": fit.success,
                "n_finite_starts": fit.n_finite,
                "n_success_flag": fit.n_success_flag,
                "rel_gap_second": fit.rel_gap_second,
                "collapsed_to_single": fit.diagnostics.get("collapsed_to_single", False),
                "median_second_fraction": fit.diagnostics.get("median_second_fraction", np.nan),
                "beta_over_alpha": fit.diagnostics.get("beta_over_alpha", np.nan),
                **{f"param_{k}": v for k, v in fit.params.items()},
            }
        )


def _largest_slice_table(fits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, sub in fits.groupby("dataset"):
        # d_id may be str or int; the largest token budget is the max in numeric order when int.
        ids = sub["d_id"].unique().tolist()
        if all(isinstance(x, (int, np.integer)) or str(x).lstrip("-").isdigit() for x in ids):
            primary = max(int(x) for x in ids)
            piece = sub[sub["d_id"].astype(int) == primary]
        else:
            primary = ids[0]
            piece = sub
        for _, row in piece.iterrows():
            rows.append({**row.to_dict(), "which": "largest_token_budget"})
        # Mean across slices. Slices share models, so this is a stability average.
        for kind, g in sub.groupby("kind"):
            rows.append(
                {
                    "dataset": dataset,
                    "d_id": "mean_over_slices",
                    "kind": kind,
                    "which": "mean_over_slices",
                    "n_slices": int(g["d_id"].nunique()),
                    "n_train": int(g["n_train"].median()),
                    "n_test": int(g["n_test"].median()),
                    "n_params": int(g["n_params"].iloc[0]),
                    "dof": int(g["dof"].median()),
                    "saturated": bool(g["saturated"].iloc[0]),
                    "test_rmse": float(g["test_rmse"].mean()),
                    "test_mae": float(g["test_mae"].mean()),
                    "test_rmse_median_across_slices": float(g["test_rmse"].median()),
                    "test_rmse_std_across_slices": float(g["test_rmse"].std(ddof=0)),
                    "train_rmse": float(g["train_rmse"].mean()),
                    "optimizer_success_rate": float(g["optimizer_success"].mean()),
                    "collapsed_rate": float(g["collapsed_to_single"].mean()),
                    "n_nonpositive_pred_test": int(g["n_nonpositive_pred_test"].sum()),
                }
            )
    return pd.DataFrame(rows)


def _plot_ladder(df: pd.DataFrame, fits: pd.DataFrame, dataset: str, path: Path, ylabel: str) -> None:
    sub = df[df["dataset"] == dataset]
    ids = sub["d_id"].unique().tolist()
    if all(str(x).lstrip("-").isdigit() for x in ids):
        primary = max(int(x) for x in ids)
        sub = sub[sub["d_id"].astype(int) == primary]
        fit_sub = fits[fits["dataset"] == dataset]
        fit_sub = fit_sub[fit_sub["d_id"].astype(int) == primary]
        title = f"{dataset} at the largest shared token budget"
    else:
        fit_sub = fits[fits["dataset"] == dataset]
        title = dataset
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for split, marker, color in (("train", "o", "#4d4d4d"), ("test", "s", "#000000")):
        part = sub[sub["split"] == split]
        ax.scatter(part["N"], part["loss"], marker=marker, c=color, s=36, zorder=3, label=split)
    xs = np.geomspace(sub["N"].min(), sub["N"].max(), 250)
    train = sub[sub["split"] == "train"]
    note = []
    for kind in KINDS:
        row = fit_sub[fit_sub["kind"] == kind]
        if row.empty:
            continue
        params = {}
        for col in row.columns:
            if col.startswith("param_"):
                val = row.iloc[0][col]
                if pd.notna(val):
                    params[col.replace("param_", "")] = float(val)
        if not params:
            continue
        ys = predict(kind, params, N=xs)
        outside = np.mean((ys < sub["loss"].min() - 1) | (ys > sub["loss"].max() + 1) | ~np.isfinite(ys))
        if outside > 0.4:
            note.append(kind)
        ax.plot(xs, ys, color=COLORS[kind], lw=2, label=LABELS[kind])
    pad = 0.2 * (sub["loss"].max() - sub["loss"].min())
    ax.set_ylim(sub["loss"].min() - pad, sub["loss"].max() + pad)
    ax.set_xscale("log")
    ax.set_xlabel("parameters N")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    if note:
        ax.text(0.01, 0.01, "curve leaves frame: " + ", ".join(note), transform=ax.transAxes, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_bars(summary: pd.DataFrame, path: Path) -> None:
    """Left: largest token budget. Right: mean over budgets (same number for Chinchilla)."""
    keep = ["chinchilla_matched_r20", "gpt3_val", "pythia_lambada"]
    labels = ["Chinchilla\nD/N~20", "GPT-3 val", "Pythia LAMBADA"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=False)
    for ax, which, title in (
        (axes[0], "largest_token_budget", "Largest shared token budget"),
        (axes[1], "mean_over_slices", "Mean over token budgets"),
    ):
        piece = summary[(summary["which"] == which) & (summary["dataset"].isin(keep))]
        x = np.arange(len(keep))
        width = 0.18
        for i, kind in enumerate(KINDS):
            vals = []
            for d in keep:
                row = piece[(piece["dataset"] == d) & (piece["kind"] == kind)]
                vals.append(float(row["test_rmse"].iloc[0]) if len(row) else np.nan)
            ax.bar(x + (i - 1.5) * width, vals, width=width, color=COLORS[kind], label=LABELS[kind])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("held-out RMSE (nats)")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
    axes[1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Extrapolation error on the largest held-out model sizes")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert_self_check()
    frames = [
        with_size_holdout(chinchilla_matched(20.0)),
        with_size_holdout(chinchilla_matched(15.0)),
        with_size_holdout(chinchilla_matched(30.0)),
        with_size_holdout(gpt3_binned()),
        with_size_holdout(pythia_lambada()),
    ]
    # chinchilla_matched() already sets dataset name from the ratio.
    data = pd.concat(frames, ignore_index=True)
    # Re-apply dataset-wise holdout. with_size_holdout was applied per frame,
    # and concat keeps those labels. Confirm each dataset separately.
    for dataset, sub in data.groupby("dataset"):
        train_n = sub.loc[sub["split"] == "train", "N"]
        test_n = sub.loc[sub["split"] == "test", "N"]
        if train_n.max() >= test_n.min():
            raise AssertionError(dataset)
    pred_rows: list = []
    fit_rows: list = []
    for dataset, sub in data.groupby("dataset"):
        for _, slice_df in sub.groupby("d_id"):
            _fit_slice(dataset, slice_df, pred_rows, fit_rows)
        print(f"fitted {dataset}", flush=True)
    preds = pd.DataFrame(pred_rows)
    fits = pd.DataFrame(fit_rows)
    summary = _largest_slice_table(fits)
    preds.to_csv(OUT / "predictions.csv", index=False)
    fits.to_csv(OUT / "metrics_by_slice.csv", index=False)
    summary.to_csv(OUT / "metrics_summary.csv", index=False)
    data.to_csv(OUT / "ladders_used.csv", index=False)
    _plot_bars(summary, OUT / "heldout_rmse.png")
    _plot_ladder(data, fits, "chinchilla_matched_r20", OUT / "chinchilla_ladder.png", "loss (nats)")
    _plot_ladder(data, fits, "gpt3_val", OUT / "gpt3_ladder.png", "validation loss (nats)")
    _plot_ladder(data, fits, "pythia_lambada", OUT / "pythia_ladder.png", r"LAMBADA NLL $\ln$(ppl)")
    show = summary[summary["which"].isin(["largest_token_budget", "mean_over_slices"])]
    cols = ["dataset", "which", "kind", "n_train", "n_test", "dof", "saturated", "test_rmse", "test_mae", "train_rmse", "optimizer_success"]
    # mean rows use optimizer_success_rate instead
    print(summary[["dataset", "which", "kind", "test_rmse", "test_mae", "train_rmse"]].to_string(index=False))
    # Silence unused in case of refactor.
    del cols, show


if __name__ == "__main__":
    main()
