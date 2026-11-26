"""Experiment 01 protocol on ladders that were not in the ColPret back-test.

Hold out the largest sizes (2 if fewer than 15 sizes, else 20%). Fit only on
smaller N. Alpha is the fast exponent and beta < alpha is the slow one.
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

from py_src.conjecture_fits import N_PARAMS, fit_power, holdout_count, predict, regression_metrics  # noqa: E402

DATA = REPO / "data" / "public_ladders"
OUT = Path(__file__).resolve().parent / "outputs"
FILES = [
    "epoch_chinchilla_matched_r20.csv.gz",
    "pythia_pile_val_final.csv.gz",
    "porian_rw_matched_r20.csv.gz",
    "porian_owt2_matched_r20.csv.gz",
    "scaling_c4_matched_r20.csv.gz",
]
KINDS = ["single_term", "two_term", "log_correction", "broken_power"]
COLORS = {"single_term": "#0072B2", "two_term": "#D55E00", "log_correction": "#009E73", "broken_power": "#CC79A7"}


def _with_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sizes = np.sort(out["N"].unique())
    k = holdout_count(len(sizes))
    test = set(sizes[-k:])
    out["split"] = np.where(out["N"].isin(test), "test", "train")
    if out.loc[out["split"] == "train", "N"].max() >= out.loc[out["split"] == "test", "N"].min():
        raise AssertionError("size split overlap")
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pred_rows = []
    fit_rows = []
    frames = []
    for name in FILES:
        df = _with_split(pd.read_csv(DATA / name))
        frames.append(df)
        dataset = df["dataset"].iloc[0]
        train = df[df["split"] == "train"]
        test = df[df["split"] == "test"]
        print(f"{dataset}: train {train['N'].nunique()} test {test['N'].nunique()}", flush=True)
        for kind in KINDS:
            fit = fit_power(kind, train["N"], train["loss"], n_starts=12, seed=0)
            yhat = {}
            for split, part in (("train", train), ("test", test)):
                pred = np.full(len(part), np.nan) if fit.failed else predict(kind, fit.params, N=part["N"].to_numpy())
                yhat[split] = pred
                for i, (_, row) in enumerate(part.iterrows()):
                    pred_rows.append(
                        {
                            "dataset": dataset,
                            "kind": kind,
                            "split": split,
                            "model_name": row["model_name"],
                            "N": row["N"],
                            "D": row["D"],
                            "loss": row["loss"],
                            "prediction": float(pred[i]),
                        }
                    )
            te = regression_metrics(test["loss"], yhat["test"])
            tr = regression_metrics(train["loss"], yhat["train"])
            k = N_PARAMS[kind]
            fit_rows.append(
                {
                    "dataset": dataset,
                    "kind": kind,
                    "n_train": tr["n"],
                    "n_test": te["n"],
                    "n_params": k,
                    "dof": tr["n"] - k,
                    "saturated": tr["n"] - k <= 0,
                    "train_rmse": tr["rmse"],
                    "test_rmse": te["rmse"],
                    "test_mae": te["mae"],
                    "optimizer_success": fit.success,
                    "collapsed_to_single": fit.diagnostics.get("collapsed_to_single", False),
                    "median_second_fraction": fit.diagnostics.get("median_second_fraction", np.nan),
                    "n_nonpositive_pred_test": te["n_nonpositive_pred"],
                    **{f"param_{key}": val for key, val in fit.params.items()},
                }
            )
    preds = pd.DataFrame(pred_rows)
    fits = pd.DataFrame(fit_rows)
    preds.to_csv(OUT / "predictions.csv", index=False)
    fits.to_csv(OUT / "metrics.csv", index=False)
    pd.concat(frames, ignore_index=True).to_csv(OUT / "ladders_used.csv", index=False)

    # Recompute test RMSE from predictions.
    te = preds[preds["split"] == "test"]
    recomputed = te.groupby(["dataset", "kind"]).apply(
        lambda g: float(np.sqrt(np.mean((g["prediction"] - g["loss"]) ** 2))), include_groups=False
    )
    merged = fits.merge(recomputed.rename("re_rmse").reset_index(), on=["dataset", "kind"])
    if (merged["test_rmse"] - merged["re_rmse"]).abs().max() > 1e-9:
        raise AssertionError("RMSE mismatch")

    datasets = list(dict.fromkeys(fits["dataset"]))
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    x = np.arange(len(datasets))
    width = 0.18
    for i, kind in enumerate(KINDS):
        vals = [float(fits[(fits.dataset == d) & (fits.kind == kind)]["test_rmse"].iloc[0]) for d in datasets]
        ax.bar(x + (i - 1.5) * width, vals, width=width, color=COLORS[kind], label=kind.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(["Epoch Fig. 4\nD/N~20", "Pythia Pile val\nfixed 300B", "Porian rw\nD/N~20", "Porian owt2\nD/N~20", "C4 scaling\nD/N~20"], fontsize=8)
    ax.set_ylabel("held-out RMSE (nats)")
    ax.set_title("Back-test on newly fetched ladders")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "heldout_rmse.png", dpi=130)
    plt.close(fig)
    print(fits[["dataset", "kind", "n_train", "n_test", "dof", "saturated", "test_rmse", "test_mae", "optimizer_success", "collapsed_to_single"]].to_string(index=False))


if __name__ == "__main__":
    main()
