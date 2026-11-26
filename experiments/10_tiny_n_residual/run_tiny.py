"""Does a fast term show up as extra loss at the small end?

Fit only on sizes above a threshold. Score the sizes below it, which the fit
never saw. Residual = loss - prediction. A positive mean residual means the
large-N fit underpredicts small-model loss, the signature of an extra positive
fast term A N^{-alpha}.

Thresholds are fixed: the smallest 20% and the smallest 40% of sizes, at least
2, and at least 4 sizes kept for training. Alpha is fast, beta < alpha is slow.
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

from py_src.conjecture_fits import N_PARAMS, fit_power, predict, regression_metrics  # noqa: E402
from py_src.ladders import chinchilla_matched  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
PUBLIC = REPO / "data" / "public_ladders"
FRACS = (0.20, 0.40)
KINDS = ("single_term", "two_term")


def _n_small(n_sizes: int, frac: float) -> int:
    k = int(round(frac * n_sizes))
    return max(2, min(k, n_sizes - 4))


def _ladders() -> list[pd.DataFrame]:
    chi = chinchilla_matched(20.0)
    chi["dataset"] = "chinchilla_matched_r20"
    frames = [chi]
    for name in (
        "epoch_chinchilla_matched_r20.csv.gz",
        "pythia_pile_val_final.csv.gz",
        "porian_rw_matched_r20.csv.gz",
    ):
        frames.append(pd.read_csv(PUBLIC / name))
    return frames


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pred_rows = []
    rows = []
    for df in _ladders():
        dataset = df["dataset"].iloc[0]
        sizes = np.sort(df["N"].unique())
        for frac in FRACS:
            k = _n_small(len(sizes), frac)
            small = set(sizes[:k])
            part_s = df[df["N"].isin(small)]
            part_l = df[~df["N"].isin(small)]
            if part_s["N"].max() >= part_l["N"].min():
                raise AssertionError(dataset)
            for kind in KINDS:
                fit = fit_power(kind, part_l["N"], part_l["loss"], n_starts=12, seed=0)
                yhat = np.full(len(part_s), np.nan) if fit.failed else predict(kind, fit.params, N=part_s["N"].to_numpy())
                resid = part_s["loss"].to_numpy() - yhat
                met = regression_metrics(part_s["loss"], yhat)
                rows.append(
                    {
                        "dataset": dataset,
                        "frac_small": frac,
                        "n_small": int(part_s["N"].nunique()),
                        "n_large": int(part_l["N"].nunique()),
                        "kind": kind,
                        "dof": int(part_l["N"].nunique() - N_PARAMS[kind]),
                        "saturated": part_l["N"].nunique() - N_PARAMS[kind] <= 0,
                        "mean_residual": float(np.mean(resid)) if np.all(np.isfinite(resid)) else np.nan,
                        "median_residual": float(np.median(resid)) if np.all(np.isfinite(resid)) else np.nan,
                        "small_rmse": met["rmse"],
                        "optimizer_success": fit.success,
                        "collapsed_to_single": fit.diagnostics.get("collapsed_to_single", False),
                        "n_nonpositive": met["n_nonpositive_pred"],
                    }
                )
                for i, (_, row) in enumerate(part_s.iterrows()):
                    pred_rows.append(
                        {
                            "dataset": dataset,
                            "frac_small": frac,
                            "kind": kind,
                            "N": row["N"],
                            "loss": row["loss"],
                            "prediction": float(yhat[i]),
                            "residual": float(resid[i]),
                        }
                    )
            print(dataset, frac, "small", k, flush=True)
    metrics = pd.DataFrame(rows)
    preds = pd.DataFrame(pred_rows)
    # Recompute mean residual from the point file.
    chk = preds.groupby(["dataset", "frac_small", "kind"]).residual.mean()
    merged = metrics.merge(chk.rename("re_mean").reset_index(), on=["dataset", "frac_small", "kind"])
    if (merged.mean_residual - merged.re_mean).abs().max() > 1e-8:
        raise AssertionError("residual mismatch")
    metrics.to_csv(OUT / "metrics.csv", index=False)
    preds.to_csv(OUT / "small_end_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    sub = metrics[metrics.frac_small == 0.40]
    datasets = list(dict.fromkeys(sub.dataset))
    labels = ["Chinchilla", "Epoch", "Pythia Pile", "Porian rw"]
    x = np.arange(len(datasets))
    for i, kind, color in ((0, "single_term", "#0072B2"), (1, "two_term", "#D55E00")):
        vals = [float(sub[(sub.dataset == d) & (sub.kind == kind)].mean_residual.iloc[0]) for d in datasets]
        ax.bar(x + (i - 0.5) * 0.35, vals, width=0.35, color=color, label=kind.replace("_", " "))
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean residual on the small end (nats)")
    ax.set_title("Fit on the large 60% of sizes, score the small 40%")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "small_end_residual.png", dpi=130)
    plt.close(fig)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
