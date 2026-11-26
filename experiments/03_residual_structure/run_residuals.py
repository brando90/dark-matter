"""In-sample residual structure after a single-term fit.

This is not a back-test. Lower in-sample error from a five-parameter model is
expected and is not evidence that the model extrapolates. Extrapolation is
experiments 01 and 02.

Naming: here alpha is the fast exponent (larger, dominates at small N) and
beta is the slow exponent, with beta < alpha. The experiments index writes the
same law with those two names swapped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from py_src.conjecture_fits import assert_self_check, fit_joint, fit_power, predict  # noqa: E402
from py_src.ladders import chinchilla_matched, gpt3_binned, load_chinchilla_runs, pythia_lambada  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"


def _quad(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    X = np.column_stack([np.ones(len(x)), x, x ** 2])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ coef
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r, p = spearmanr(x, y)
    return {
        "spearman_r": float(r),
        "spearman_p_iid": float(p),
        "quad_intercept": float(coef[0]),
        "quad_linear": float(coef[1]),
        "quad_square": float(coef[2]),
        "quad_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "residual_rmse": float(np.sqrt(np.mean(y ** 2))),
        "n": int(len(y)),
    }


def _rows_for_curve(name: str, kind: str, frame: pd.DataFrame, yhat: np.ndarray, fit) -> tuple[list, dict]:
    resid = frame["loss"].to_numpy() - yhat
    logn = np.log(frame["N"].to_numpy())
    stats = _quad(logn, resid)
    stats.update(
        {
            "curve": name,
            "kind": kind,
            "optimizer_success": fit.success,
            "collapsed_to_single": fit.diagnostics.get("collapsed_to_single", False),
            "train_rmse": float(np.sqrt(np.mean(resid ** 2))),
            **{f"param_{k}": v for k, v in fit.params.items()},
        }
    )
    rows = []
    for i, row in frame.reset_index(drop=True).iterrows():
        rows.append(
            {
                "curve": name,
                "kind": kind,
                "N": row["N"],
                "D": row["D"],
                "loss": row["loss"],
                "prediction": float(yhat[i]),
                "residual": float(resid[i]),
            }
        )
    return rows, stats


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert_self_check()
    residual_rows = []
    stat_rows = []

    matched = chinchilla_matched(20.0).sort_values("N")
    for kind in ("single_term", "two_term"):
        fit = fit_power(kind, matched["N"], matched["loss"], n_starts=12, seed=0)
        yhat = predict(kind, fit.params, N=matched["N"])
        rows, stats = _rows_for_curve("chinchilla_matched_r20", kind, matched, yhat, fit)
        residual_rows.extend(rows)
        stat_rows.append(stats)

    runs = load_chinchilla_runs()
    joint = fit_joint(runs["N"], runs["D"], runs["loss"], extra=False, n_starts=12, seed=0)
    yhat_j = predict("joint", joint.params, N=runs["N"], D=runs["D"])
    rows, stats = _rows_for_curve("chinchilla_all_runs", "joint", runs, yhat_j, joint)
    residual_rows.extend(rows)
    # Also residual vs log D, stored as a second stat row.
    resid_j = runs["loss"].to_numpy() - yhat_j
    dstats = _quad(np.log(runs["D"].to_numpy()), resid_j)
    dstats.update({"curve": "chinchilla_all_runs", "kind": "joint_vs_logD", "optimizer_success": joint.success, "collapsed_to_single": False, "train_rmse": stats["train_rmse"]})
    stat_rows.append(stats)
    stat_rows.append(dstats)
    # Partial linear association after the joint fit: residual ~ 1 + log N + log D.
    logn = np.log(runs["N"].to_numpy())
    logd = np.log(runs["D"].to_numpy())
    X = np.column_stack([np.ones(len(runs)), logn, logd])
    coef, *_ = np.linalg.lstsq(X, resid_j, rcond=None)
    fitted = X @ coef
    ss_res = float(np.sum((resid_j - fitted) ** 2))
    ss_tot = float(np.sum((resid_j - resid_j.mean()) ** 2))
    stat_rows.append(
        {
            "curve": "chinchilla_all_runs",
            "kind": "joint_residual_ols_logN_logD",
            "quad_intercept": float(coef[0]),
            "quad_linear": float(coef[1]),
            "quad_square": float(coef[2]),
            "quad_r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
            "residual_rmse": float(np.sqrt(np.mean(resid_j ** 2))),
            "n": int(len(runs)),
            "optimizer_success": joint.success,
            "note": "quad_linear is the log N coefficient; quad_square is the log D coefficient",
        }
    )

    gpt = gpt3_binned()
    gpt = gpt[gpt["d_id"] == gpt["d_id"].max()].sort_values("N")
    py = pythia_lambada()
    py = py[py["d_id"] == py["d_id"].max()].sort_values("N")
    for name, frame in (("gpt3_largest_D", gpt), ("pythia_lambada_largest_D", py)):
        fit = fit_power("single_term", frame["N"], frame["loss"], n_starts=12, seed=0)
        yhat = predict("single_term", fit.params, N=frame["N"])
        rows, stats = _rows_for_curve(name, "single_term", frame, yhat, fit)
        residual_rows.extend(rows)
        stat_rows.append(stats)

    residuals = pd.DataFrame(residual_rows)
    stats_df = pd.DataFrame(stat_rows)
    # The joint RMSE is sensitive to a few very small-D runs. Record that
    # sensitivity explicitly so the quadratic R^2 is not quoted alone.
    joint_resid = residuals[(residuals["curve"] == "chinchilla_all_runs") & (residuals["kind"] == "joint")].copy()
    keep = joint_resid[joint_resid["residual"] <= 0.5]
    def _r2(frame: pd.DataFrame, col: str) -> float:
        return float(_quad(np.log(frame[col].to_numpy()), frame["residual"].to_numpy())["quad_r2"])
    sensitivity = pd.DataFrame(
        [
            {
                "curve": "chinchilla_all_runs",
                "subset": "all",
                "n": int(len(joint_resid)),
                "median_abs_residual": float(joint_resid["residual"].abs().median()),
                "n_abs_gt_0.2": int((joint_resid["residual"].abs() > 0.2).sum()),
                "quad_r2_logN": _r2(joint_resid, "N"),
                "quad_r2_logD": _r2(joint_resid, "D"),
            },
            {
                "curve": "chinchilla_all_runs",
                "subset": "drop_residual_above_0.5",
                "n": int(len(keep)),
                "median_abs_residual": float(keep["residual"].abs().median()),
                "n_abs_gt_0.2": int((keep["residual"].abs() > 0.2).sum()),
                "quad_r2_logN": _r2(keep, "N"),
                "quad_r2_logD": _r2(keep, "D"),
            },
        ]
    )
    residuals.to_csv(OUT / "residuals.csv", index=False)
    stats_df.to_csv(OUT / "residual_stats.csv", index=False)
    sensitivity.to_csv(OUT / "joint_residual_sensitivity.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=True)
    sub = residuals[residuals["curve"] == "chinchilla_matched_r20"]
    for ax, kind, title in (
        (axes[0], "single_term", r"single term $E+AN^{-\alpha}$"),
        (axes[1], "two_term", r"two-term, $\alpha$ fast, $\beta<\alpha$ slow"),
    ):
        part = sub[sub["kind"] == kind]
        ax.axhline(0, color="black", lw=1)
        ax.scatter(part["N"], part["residual"], c="#0072B2", s=28)
        ax.set_xscale("log")
        ax.set_xlabel("parameters N")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("loss residual (nats)")
    fig.suptitle("Chinchilla D/N ≈ 20, in-sample residuals (not a holdout)")
    fig.tight_layout()
    fig.savefig(OUT / "chinchilla_matched_residuals.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    part = residuals[(residuals["curve"] == "chinchilla_all_runs") & (residuals["kind"] == "joint")]
    sc = axes[0].scatter(part["N"], part["residual"], c=np.log10(part["D"]), s=16, cmap="viridis")
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("joint residual (nats)")
    axes[0].set_title("Residual vs N, color = log10 D")
    fig.colorbar(sc, ax=axes[0], label="log10 tokens")
    sc2 = axes[1].scatter(part["D"], part["residual"], c=np.log10(part["N"]), s=16, cmap="plasma")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("D (tokens)")
    axes[1].set_title("Residual vs D, color = log10 N")
    fig.colorbar(sc2, ax=axes[1], label="log10 parameters")
    fig.suptitle("In-sample Chinchilla joint-law residuals")
    fig.tight_layout()
    fig.savefig(OUT / "chinchilla_joint_residuals.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for ax, curve, title in (
        (axes[0], "gpt3_largest_D", "GPT-3 val, largest D (n=8)"),
        (axes[1], "pythia_lambada_largest_D", "Pythia LAMBADA, largest D (n=7)"),
    ):
        part = residuals[residuals["curve"] == curve]
        ax.axhline(0, color="black", lw=1)
        ax.scatter(part["N"], part["residual"], c="#D55E00", s=36)
        ax.set_xscale("log")
        ax.set_xlabel("N")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("single-term residual (nats)")
    fig.suptitle("Endpoint ladders are too short for a curvature claim")
    fig.tight_layout()
    fig.savefig(OUT / "endpoint_residuals.png", dpi=130)
    plt.close(fig)
    print(stats_df[["curve", "kind", "n", "residual_rmse", "spearman_r", "quad_square", "quad_r2", "collapsed_to_single"]].to_string(index=False))


if __name__ == "__main__":
    main()
