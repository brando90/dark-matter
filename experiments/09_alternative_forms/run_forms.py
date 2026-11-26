"""Held-out RMSE and Huber AIC/BIC for alternative dark-matter forms.

N-only forms, on one row per size: single term, two-term (beta < alpha, alpha
fast), log correction, broken power law.
Joint forms, on every (N, D) run, holding out the largest sizes: Chinchilla
joint, a second N power, and a second D power (delta < gamma, gamma fast).

AIC = 2k + 2 * Huber objective and BIC = k ln(n) + 2 * Huber objective, both
on the training rows only. The Huber density's additive constant is the same
for every model at delta=1e-3, so differences are comparable. Lower is better.
These are not Gaussian AIC/BIC.
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
    fit_joint,
    fit_joint_extra_d,
    fit_power,
    holdout_count,
    predict,
    regression_metrics,
)
from py_src.ladders import chinchilla_matched, load_chinchilla_runs  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
PUBLIC = REPO / "data" / "public_ladders"
N_KINDS = ["single_term", "two_term", "log_correction", "broken_power"]
J_KINDS = ["joint", "joint_extra_n", "joint_extra_d"]


def _split_by_size(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    sizes = np.sort(out["N"].unique())
    k = holdout_count(len(sizes))
    test = set(sizes[-k:])
    out["split"] = np.where(out["N"].isin(test), "test", "train")
    if out.loc[out.split == "train", "N"].max() >= out.loc[out.split == "test", "N"].min():
        raise AssertionError(out["dataset"].iloc[0])
    return out


def _info(fit, n_train: int) -> dict:
    k = N_PARAMS[fit.kind]
    obj = float(fit.objective)
    return {
        "n_params": k,
        "dof": n_train - k,
        "saturated": n_train - k <= 0,
        "huber_objective": obj,
        "aic": 2 * k + 2 * obj,
        "bic": k * np.log(n_train) + 2 * obj,
        "optimizer_success": fit.success,
        "collapsed_to_single": fit.diagnostics.get("collapsed_to_single", False),
        "median_second_fraction": fit.diagnostics.get("median_second_fraction", fit.diagnostics.get("median_slow_fraction", np.nan)),
    }


def _predict_kind(kind: str, params: dict, part: pd.DataFrame) -> np.ndarray:
    if kind in N_KINDS:
        return predict(kind, params, N=part["N"].to_numpy())
    return predict(kind, params, N=part["N"].to_numpy(), D=part["D"].to_numpy())


def _eval(dataset: str, family: str, kind: str, fit, train: pd.DataFrame, test: pd.DataFrame, pred_rows: list) -> dict:
    yhat = {}
    for split, part in (("train", train), ("test", test)):
        pred = np.full(len(part), np.nan) if fit.failed else _predict_kind(kind, fit.params, part)
        yhat[split] = pred
        for i, (_, row) in enumerate(part.iterrows()):
            pred_rows.append(
                {
                    "dataset": dataset,
                    "family": family,
                    "kind": kind,
                    "split": split,
                    "N": row["N"],
                    "D": row["D"],
                    "loss": row["loss"],
                    "prediction": float(pred[i]),
                }
            )
    te = regression_metrics(test["loss"], yhat["test"])
    tr = regression_metrics(train["loss"], yhat["train"])
    row = {
        "dataset": dataset,
        "family": family,
        "kind": kind,
        "n_train": tr["n"],
        "n_test": te["n"],
        "train_rmse": tr["rmse"],
        "test_rmse": te["rmse"],
        "test_mae": te["mae"],
        "n_nonpositive_pred_test": te["n_nonpositive_pred"],
        **_info(fit, tr["n"]),
    }
    return row


def _n_ladders() -> list[pd.DataFrame]:
    frames = [chinchilla_matched(20.0)]
    frames[0]["dataset"] = "chinchilla_matched_r20"
    for name in ("epoch_chinchilla_matched_r20.csv.gz", "pythia_pile_val_final.csv.gz", "porian_rw_matched_r20.csv.gz"):
        frames.append(pd.read_csv(PUBLIC / name))
    return [_split_by_size(df) for df in frames]


def _joint_tables() -> list[pd.DataFrame]:
    chi = load_chinchilla_runs()
    chi["dataset"] = "chinchilla_runs"
    epoch = pd.read_csv(PUBLIC / "epoch_chinchilla_runs.csv.gz")
    return [_split_by_size(chi), _split_by_size(epoch)]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pred_rows: list = []
    metric_rows: list = []
    for df in _n_ladders():
        dataset = df["dataset"].iloc[0]
        train, test = df[df.split == "train"], df[df.split == "test"]
        print(dataset, "N-only", len(train), len(test), flush=True)
        for kind in N_KINDS:
            fit = fit_power(kind, train["N"], train["loss"], n_starts=12, seed=0)
            metric_rows.append(_eval(dataset, "n_only", kind, fit, train, test, pred_rows))
    for df in _joint_tables():
        dataset = df["dataset"].iloc[0]
        train, test = df[df.split == "train"], df[df.split == "test"]
        print(dataset, "joint", len(train), len(test), flush=True)
        for kind, fitter in (
            ("joint", lambda: fit_joint(train["N"], train["D"], train["loss"], extra=False, n_starts=12, seed=0)),
            ("joint_extra_n", lambda: fit_joint(train["N"], train["D"], train["loss"], extra=True, n_starts=12, seed=0)),
            ("joint_extra_d", lambda: fit_joint_extra_d(train["N"], train["D"], train["loss"], n_starts=12, seed=0)),
        ):
            metric_rows.append(_eval(dataset, "joint", kind, fitter(), train, test, pred_rows))
    preds = pd.DataFrame(pred_rows)
    metrics = pd.DataFrame(metric_rows)
    te = preds[preds.split == "test"]
    recomputed = te.groupby(["dataset", "kind"]).apply(
        lambda g: float(np.sqrt(np.mean((g.prediction - g.loss) ** 2))), include_groups=False
    )
    merged = metrics.merge(recomputed.rename("re_rmse").reset_index(), on=["dataset", "kind"])
    if (merged.test_rmse - merged.re_rmse).abs().max() > 1e-8:
        raise AssertionError("RMSE mismatch")
    preds.to_csv(OUT / "predictions.csv", index=False)
    metrics.to_csv(OUT / "metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    npart = metrics[metrics.family == "n_only"]
    datasets = list(dict.fromkeys(npart.dataset))
    x = np.arange(len(datasets))
    width = 0.18
    colors = {"single_term": "#0072B2", "two_term": "#D55E00", "log_correction": "#009E73", "broken_power": "#CC79A7"}
    for i, kind in enumerate(N_KINDS):
        vals = [float(npart[(npart.dataset == d) & (npart.kind == kind)].test_rmse.iloc[0]) for d in datasets]
        axes[0].bar(x + (i - 1.5) * width, vals, width=width, color=colors[kind], label=kind.replace("_", " "))
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Chinchilla\nD/N~20", "Epoch\nD/N~20", "Pythia\nPile val", "Porian rw"], fontsize=8)
    axes[0].set_ylabel("held-out RMSE")
    axes[0].set_title("N-only forms")
    axes[0].legend(fontsize=7)
    axes[0].grid(True, axis="y", alpha=0.3)
    jpart = metrics[metrics.family == "joint"]
    jsets = list(dict.fromkeys(jpart.dataset))
    x = np.arange(len(jsets))
    jcolors = {"joint": "#0072B2", "joint_extra_n": "#D55E00", "joint_extra_d": "#009E73"}
    for i, kind in enumerate(J_KINDS):
        vals = [float(jpart[(jpart.dataset == d) & (jpart.kind == kind)].test_rmse.iloc[0]) for d in jsets]
        axes[1].bar(x + (i - 1) * 0.25, vals, width=0.25, color=jcolors[kind], label=kind.replace("_", " "))
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Chinchilla runs", "Epoch runs"], fontsize=8)
    axes[1].set_title("Joint forms, large-N holdout")
    axes[1].legend(fontsize=7)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "heldout_rmse.png", dpi=130)
    plt.close(fig)
    cols = ["dataset", "kind", "n_train", "dof", "test_rmse", "aic", "bic", "collapsed_to_single", "optimizer_success"]
    print(metrics[cols].to_string(index=False))


if __name__ == "__main__":
    main()
