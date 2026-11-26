"""Extrapolation in compute and in the joint (N, D) plane.

Same ladders as experiment 01, but each training run is a point (N, D, L)
rather than one point per size. Two held-out protocols:

- size: largest model sizes (every token budget of those sizes)
- compute: runs with C above the 80th percentile of C (C = 6 N D)

Models are additive. The joint law is the Chinchilla form
L = E + A N^{-alpha} + G D^{-gamma}. The extra-N model adds a second
parameter power with a shallower exponent. Nothing here is multiplicative
in N and D.
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
    fit_joint,
    fit_power,
    holdout_count,
    predict,
    regression_metrics,
)
from py_src.ladders import gpt3_binned, load_chinchilla_runs, pythia_lambada  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"

# label, how to fit, parameter count
MODELS = ["n_only", "d_only", "compute_one", "compute_two", "joint", "joint_extra_n"]
COLORS = {
    "n_only": "#0072B2",
    "d_only": "#56B4E9",
    "compute_one": "#009E73",
    "compute_two": "#E69F00",
    "joint": "#D55E00",
    "joint_extra_n": "#CC79A7",
}
PRETTY = {
    "n_only": r"N only",
    "d_only": r"D only",
    "compute_one": r"C one-term",
    "compute_two": r"C two-term",
    "joint": r"joint $E+AN^{-\alpha}+GD^{-\gamma}$",
    "joint_extra_n": r"joint + second $N$ term",
}


def _datasets() -> dict[str, pd.DataFrame]:
    chi = load_chinchilla_runs()
    chi["dataset"] = "chinchilla_runs"
    gpt = gpt3_binned()
    gpt["dataset"] = "gpt3_val"
    py = pythia_lambada()
    py["dataset"] = "pythia_lambada"
    return {"chinchilla_runs": chi, "gpt3_val": gpt, "pythia_lambada": py}


def _size_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sizes = np.sort(out["N"].unique())
    k = holdout_count(len(sizes))
    test = set(sizes[-k:])
    out["split"] = np.where(out["N"].isin(test), "test", "train")
    out["protocol"] = "size"
    return out


def _compute_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    threshold = float(out["C"].quantile(0.80))
    # Ties at the threshold stay in train, so every test FLOP count is strictly larger.
    out["split"] = np.where(out["C"] > threshold, "test", "train")
    out["protocol"] = "compute"
    out["compute_threshold"] = threshold
    if out.loc[out["split"] == "train", "C"].max() >= out.loc[out["split"] == "test", "C"].min():
        raise AssertionError("compute split did not separate C")
    return out


def _fit_predict(label: str, train: pd.DataFrame, part: pd.DataFrame):
    if label == "n_only":
        fit = fit_power("single_term", train["N"], train["loss"], n_starts=12, seed=0)
        kind, x = "single_term", part["N"]
    elif label == "d_only":
        fit = fit_power("single_term", train["D"], train["loss"], n_starts=12, seed=0)
        kind, x = "single_term", part["D"]
    elif label == "compute_one":
        fit = fit_power("single_term", train["C"], train["loss"], n_starts=12, seed=0)
        kind, x = "single_term", part["C"]
    elif label == "compute_two":
        fit = fit_power("two_term", train["C"], train["loss"], n_starts=12, seed=0)
        kind, x = "two_term", part["C"]
    elif label == "joint":
        fit = fit_joint(train["N"], train["D"], train["loss"], extra=False, n_starts=12, seed=0)
        kind, x = "joint", part["N"]
    elif label == "joint_extra_n":
        fit = fit_joint(train["N"], train["D"], train["loss"], extra=True, n_starts=12, seed=0)
        kind, x = "joint_extra_n", part["N"]
    else:
        raise KeyError(label)
    if fit.failed:
        return fit, np.full(len(part), np.nan)
    yhat = predict(kind, fit.params, N=x, D=part["D"], C=part["C"])
    return fit, np.asarray(yhat, dtype=np.float64)


def _macro_rmse(part: pd.DataFrame) -> float:
    scores = []
    for _, g in part.groupby("N"):
        err = g["prediction"].to_numpy() - g["loss"].to_numpy()
        if np.any(~np.isfinite(err)):
            return np.nan
        scores.append(float(np.sqrt(np.mean(err ** 2))))
    return float(np.mean(scores)) if scores else np.nan


def _coverage(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    def frac_inside(values, lo, hi):
        return float(np.mean((values >= lo) & (values <= hi)))

    return {
        "frac_test_N_inside_train": frac_inside(test["N"], train["N"].min(), train["N"].max()),
        "frac_test_D_inside_train": frac_inside(test["D"], train["D"].min(), train["D"].max()),
        "frac_test_C_inside_train": frac_inside(test["C"], train["C"].min(), train["C"].max()),
        "n_train_sizes": int(train["N"].nunique()),
        "n_test_sizes": int(test["N"].nunique()),
        "train_N_max": float(train["N"].max()),
        "test_N_min": float(test["N"].min()),
        "train_C_max": float(train["C"].max()),
        "test_C_min": float(test["C"].min()),
    }


def _plot_bars(summary: pd.DataFrame, protocol: str, path: Path, title: str) -> None:
    piece = summary[summary["protocol"] == protocol]
    datasets = ["chinchilla_runs", "gpt3_val", "pythia_lambada"]
    labels = ["Chinchilla runs", "GPT-3 val", "Pythia LAMBADA"]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(datasets))
    width = 0.12
    for i, label in enumerate(MODELS):
        vals = []
        for d in datasets:
            row = piece[(piece["dataset"] == d) & (piece["model"] == label)]
            vals.append(float(row["test_macro_rmse"].iloc[0]) if len(row) else np.nan)
        ax.bar(x + (i - 2.5) * width, vals, width=width, color=COLORS[label], label=PRETTY[label])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("held-out macro RMSE (nats)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_pred(preds: pd.DataFrame, path: Path) -> None:
    sub = preds[(preds["dataset"] == "chinchilla_runs") & (preds["protocol"] == "size") & (preds["split"] == "test")]
    labels = ["n_only", "joint", "joint_extra_n"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharex=True, sharey=True)
    for ax, label in zip(axes, labels):
        part = sub[sub["model"] == label]
        ax.scatter(part["loss"], part["prediction"], s=16, alpha=0.75, c=COLORS[label])
        lo = min(part["loss"].min(), part["prediction"].min())
        hi = max(part["loss"].max(), part["prediction"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_title(PRETTY[label])
        ax.set_xlabel("held-out loss")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("predicted loss")
    fig.suptitle("Chinchilla size holdout: predicted vs actual")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert_self_check()
    pred_rows = []
    metric_rows = []
    for dataset, raw in _datasets().items():
        for protocol, splitter in (("size", _size_split), ("compute", _compute_split)):
            df = splitter(raw)
            train = df[df["split"] == "train"]
            test = df[df["split"] == "test"]
            if protocol == "size" and train["N"].max() >= test["N"].min():
                raise AssertionError(f"{dataset} size split overlaps")
            cover = _coverage(train, test)
            print(f"{dataset} {protocol}: train {len(train)} test {len(test)}", flush=True)
            for label in MODELS:
                kind = {
                    "n_only": "single_term",
                    "d_only": "single_term",
                    "compute_one": "single_term",
                    "compute_two": "two_term",
                    "joint": "joint",
                    "joint_extra_n": "joint_extra_n",
                }[label]
                fit, y_test_hat = _fit_predict(label, train, test)
                if fit.failed:
                    y_train_hat = np.full(len(train), np.nan)
                    y_test_hat = np.full(len(test), np.nan)
                else:
                    x_train = {"d_only": train["D"], "compute_one": train["C"], "compute_two": train["C"]}.get(label, train["N"])
                    y_train_hat = predict(kind, fit.params, N=x_train, D=train["D"], C=train["C"])
                for split_name, part, yhat in (("train", train, y_train_hat), ("test", test, y_test_hat)):
                    for i, (_, row) in enumerate(part.iterrows()):
                        pred_rows.append(
                            {
                                "dataset": dataset,
                                "protocol": protocol,
                                "model": label,
                                "split": split_name,
                                "model_name": row["model_name"],
                                "N": row["N"],
                                "D": row["D"],
                                "C": row["C"],
                                "loss": row["loss"],
                                "prediction": float(yhat[i]),
                            }
                        )
                te = pd.DataFrame({"N": test["N"].to_numpy(), "loss": test["loss"].to_numpy(), "prediction": y_test_hat})
                tr = pd.DataFrame({"N": train["N"].to_numpy(), "loss": train["loss"].to_numpy(), "prediction": y_train_hat})
                te_m = regression_metrics(test["loss"], y_test_hat)
                tr_m = regression_metrics(train["loss"], y_train_hat)
                n_params = {"n_only": 3, "d_only": 3, "compute_one": 3, "compute_two": 5, "joint": 5, "joint_extra_n": 7}[label]
                metric_rows.append(
                    {
                        "dataset": dataset,
                        "protocol": protocol,
                        "model": label,
                        "n_train": len(train),
                        "n_test": len(test),
                        "n_params": n_params,
                        "test_rmse": te_m["rmse"],
                        "test_mae": te_m["mae"],
                        "test_macro_rmse": _macro_rmse(te),
                        "train_rmse": tr_m["rmse"],
                        "train_macro_rmse": _macro_rmse(tr),
                        "n_nonpositive_pred_test": te_m["n_nonpositive_pred"],
                        "optimizer_success": fit.success,
                        "objective": fit.objective,
                        "mean_huber": fit.mean_huber,
                        "n_finite_starts": fit.n_finite,
                        "collapsed_to_single": fit.diagnostics.get("collapsed_to_single", False),
                        "median_second_fraction": fit.diagnostics.get("median_second_fraction", np.nan),
                        **cover,
                        **{f"param_{k}": v for k, v in fit.params.items()},
                    }
                )
                print(f"  {label}: test macro RMSE {metric_rows[-1]['test_macro_rmse']:.4f} success {fit.success}", flush=True)
    preds = pd.DataFrame(pred_rows)
    metrics = pd.DataFrame(metric_rows)
    preds.to_csv(OUT / "predictions.csv", index=False)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    _plot_bars(metrics, "size", OUT / "size_holdout_rmse.png", "Hold out the largest model sizes")
    _plot_bars(metrics, "compute", OUT / "compute_holdout_rmse.png", "Hold out the top 20% of compute")
    _plot_pred(preds, OUT / "chinchilla_size_pred_vs_actual.png")
    show = metrics[["dataset", "protocol", "model", "test_macro_rmse", "test_rmse", "train_rmse", "optimizer_success"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
