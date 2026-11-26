"""Synthetic recovery and a false-positive extrapolation check.

Data are drawn from a known additive law with multiplicative log-noise, which
is the noise the Huber-on-log objective is built for. Alpha is the fast
exponent and beta < alpha is the slow one.
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
    assert_self_check,
    fit_joint,
    fit_power,
    holdout_count,
    predict,
)
from py_src.scaling_laws_fit import synth_joint  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
N_REPS = 20
KINDS = ["single_term", "two_term", "log_correction", "broken_power"]
N_GRID = np.geomspace(1e8, 1e11, 36)


def _truth_mean(case: str, n: np.ndarray) -> tuple[np.ndarray, dict]:
    if case == "two_term_separated":
        params = {"E": 1.5, "A": 12649.07, "alpha": 0.50, "B": 25.24, "beta": 0.20}
        mean = params["E"] + params["A"] * n ** (-params["alpha"]) + params["B"] * n ** (-params["beta"])
    elif case == "two_term_close":
        params = {"E": 1.6, "A": 1194.0, "alpha": 0.35, "B": 398.0, "beta": 0.28}
        # At N=1e9 the fast term is about 0.85 and the slow term is about 1.2.
        mean = params["E"] + params["A"] * n ** (-params["alpha"]) + params["B"] * n ** (-params["beta"])
    elif case == "single_term":
        params = {"E": 1.5, "A": 12649.07, "alpha": 0.50}
        mean = params["E"] + params["A"] * n ** (-params["alpha"])
    elif case == "log_correction":
        params = {"E": 1.5, "A": 1990.0, "alpha": 0.40, "gamma": -0.15, "n_ref": 1e9}
        z = np.log(n) - np.log(params["n_ref"])
        mean = params["E"] + params["A"] * n ** (-params["alpha"]) * (1.0 + params["gamma"] * z)
    else:
        raise KeyError(case)
    return mean, params


def _param_error(fit_params: dict, truth: dict) -> dict:
    out = {}
    for key, value in truth.items():
        if key in fit_params:
            out[f"err_{key}"] = float(fit_params[key] - value)
            out[f"abs_err_{key}"] = float(abs(fit_params[key] - value))
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert_self_check()
    k_test = holdout_count(len(N_GRID))
    train_n = N_GRID[:-k_test]
    test_n = N_GRID[-k_test:]
    assert train_n.max() < test_n.min()

    cases = {
        "two_term_separated": 0.02,
        "two_term_close": 0.02,
        "single_term": 0.02,
        "log_correction": 0.02,
    }
    extrap_rows = []
    recover_rows = []
    for case, sigma in cases.items():
        for rep in range(N_REPS):
            rng = np.random.default_rng(10_000 + rep)
            mean, truth = _truth_mean(case, N_GRID)
            y = mean * np.exp(rng.normal(0.0, sigma, size=len(N_GRID)))
            y_train, y_test = y[:-k_test], y[-k_test:]
            for kind in KINDS:
                fit = fit_power(kind, train_n, y_train, n_starts=8, seed=rep)
                if fit.failed:
                    yhat = np.full(len(test_n), np.nan)
                else:
                    yhat = predict(kind, fit.params, N=test_n)
                err = yhat - y_test
                extrap_rows.append(
                    {
                        "case": case,
                        "sigma": sigma,
                        "rep": rep,
                        "kind": kind,
                        "test_rmse": float(np.sqrt(np.mean(err ** 2))) if np.all(np.isfinite(err)) else np.nan,
                        "test_mae": float(np.mean(np.abs(err))) if np.all(np.isfinite(err)) else np.nan,
                        "optimizer_success": fit.success,
                        "collapsed_to_single": fit.diagnostics.get("collapsed_to_single", False),
                        "n_nonpositive": int(np.sum(~(yhat > 0))),
                    }
                )
            if case in ("two_term_separated", "two_term_close", "single_term"):
                kind_rec = "two_term" if case.startswith("two_term") else "single_term"
                fit_all = fit_power(kind_rec, N_GRID, y, n_starts=8, seed=rep)
                recover_rows.append(
                    {
                        "case": case,
                        "sigma": sigma,
                        "rep": rep,
                        "kind": fit_all.kind,
                        "success": fit_all.success,
                        "collapsed_to_single": fit_all.diagnostics.get("collapsed_to_single", False),
                        **fit_all.params,
                        **_param_error(fit_all.params, truth),
                    }
                )
        print("finished", case, flush=True)
    # Quieter separated law, same grid, fit on every size.
    for rep in range(N_REPS):
        rng = np.random.default_rng(20_000 + rep)
        mean, truth = _truth_mean("two_term_separated", N_GRID)
        y = mean * np.exp(rng.normal(0.0, 0.01, size=len(N_GRID)))
        fit_all = fit_power("two_term", N_GRID, y, n_starts=8, seed=rep)
        recover_rows.append(
            {
                "case": "two_term_separated_quiet",
                "sigma": 0.01,
                "rep": rep,
                "kind": fit_all.kind,
                "success": fit_all.success,
                "collapsed_to_single": fit_all.diagnostics.get("collapsed_to_single", False),
                **fit_all.params,
                **_param_error(fit_all.params, truth),
            }
        )

    # Joint recovery from the shared synthetic generator. beta there is the token exponent.
    joint_rows = []
    ns = np.geomspace(1e8, 1e10, 8)
    ds = np.geomspace(2e9, 2e11, 8)
    nn, dd = np.meshgrid(ns, ds)
    nn, dd = nn.ravel(), dd.ravel()
    for rep in range(10):
        yy = synth_joint(nn, dd, E=1.8, A=500.0, alpha=0.35, B=2000.0, beta=0.36, noise=0.02, seed=rep)
        fit = fit_joint(nn, dd, yy, n_starts=8, seed=rep)
        joint_rows.append(
            {
                "rep": rep,
                "success": fit.success,
                "E": fit.params.get("E"),
                "A": fit.params.get("A"),
                "alpha": fit.params.get("alpha"),
                "G": fit.params.get("G"),
                "gamma": fit.params.get("gamma"),
                "abs_err_alpha": abs(fit.params.get("alpha", np.nan) - 0.35),
                "abs_err_gamma": abs(fit.params.get("gamma", np.nan) - 0.36),
                "abs_err_E": abs(fit.params.get("E", np.nan) - 1.8),
            }
        )
    extrap = pd.DataFrame(extrap_rows)
    recover = pd.DataFrame(recover_rows)
    joint = pd.DataFrame(joint_rows)
    extrap.to_csv(OUT / "extrapolation_reps.csv", index=False)
    recover.to_csv(OUT / "parameter_recovery.csv", index=False)
    joint.to_csv(OUT / "joint_recovery.csv", index=False)

    summary_rows = []
    for (case, kind), g in extrap.groupby(["case", "kind"]):
        summary_rows.append(
            {
                "case": case,
                "kind": kind,
                "median_test_rmse": float(g["test_rmse"].median()),
                "mean_test_rmse": float(g["test_rmse"].mean()),
                "success_rate": float(g["optimizer_success"].mean()),
                "collapsed_rate": float(g["collapsed_to_single"].mean()),
                "nonpositive_rate": float((g["n_nonpositive"] > 0).mean()),
            }
        )
    # How often two-term beats single on the same replicate.
    beat_rows = []
    for case, g in extrap.groupby("case"):
        wide = g.pivot(index="rep", columns="kind", values="test_rmse")
        beat_rows.append(
            {
                "case": case,
                "frac_two_term_rmse_lt_single": float(np.mean(wide["two_term"] < wide["single_term"])),
                "median_two_minus_single": float(np.median(wide["two_term"] - wide["single_term"])),
            }
        )
    summary = pd.DataFrame(summary_rows)
    beats = pd.DataFrame(beat_rows)
    summary.to_csv(OUT / "extrapolation_summary.csv", index=False)
    beats.to_csv(OUT / "two_term_vs_single.csv", index=False)

    rec_sum = []
    for case, g in recover.groupby("case"):
        row = {"case": case, "n": int(len(g)), "success_rate": float(g["success"].mean()), "collapsed_rate": float(g["collapsed_to_single"].mean())}
        for col in g.columns:
            if col.startswith("abs_err_"):
                row[f"median_{col}"] = float(g[col].median())
        rec_sum.append(row)
    rec_sum.append(
        {
            "case": "joint_synth",
            "n": int(len(joint)),
            "success_rate": float(joint["success"].mean()),
            "median_abs_err_alpha": float(joint["abs_err_alpha"].median()),
            "median_abs_err_gamma": float(joint["abs_err_gamma"].median()),
            "median_abs_err_E": float(joint["abs_err_E"].median()),
        }
    )
    pd.DataFrame(rec_sum).to_csv(OUT / "recovery_summary.csv", index=False)

    # Plot: median test RMSE by case and model.
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    cases_order = list(cases)
    x = np.arange(len(cases_order))
    width = 0.18
    colors = {"single_term": "#0072B2", "two_term": "#D55E00", "log_correction": "#009E73", "broken_power": "#CC79A7"}
    for i, kind in enumerate(KINDS):
        vals = [float(summary[(summary["case"] == c) & (summary["kind"] == kind)]["median_test_rmse"].iloc[0]) for c in cases_order]
        ax.bar(x + (i - 1.5) * width, vals, width=width, color=colors[kind], label=kind)
    ax.set_xticks(x)
    ax.set_xticklabels(["two-term\nseparated", "two-term\nclose", "single\ntruth", "log-correction\ntruth"], fontsize=8)
    ax.set_ylabel("median held-out RMSE")
    ax.set_title(f"Synthetic extrapolation, {N_REPS} replicates, σ = 0.02")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "synthetic_extrapolation.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sep = recover[recover["case"] == "two_term_separated"]
    close = recover[recover["case"] == "two_term_close"]
    ax.scatter(sep["alpha"], sep["beta"], c="#D55E00", s=28, label="separated, σ=0.02")
    ax.scatter(close["alpha"], close["beta"], c="#0072B2", s=28, label="close, σ=0.02")
    ax.scatter([0.50], [0.20], marker="x", c="black", s=80, label="true exponents")
    ax.scatter([0.35], [0.28], marker="x", c="black", s=80)
    ax.set_xlabel(r"fitted fast $\alpha$")
    ax.set_ylabel(r"fitted slow $\beta$")
    ax.set_title("Parameter recovery, fit on all N")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "parameter_recovery.png", dpi=130)
    plt.close(fig)
    print(summary.to_string(index=False))
    print(beats.to_string(index=False))
    print(pd.DataFrame(rec_sum).to_string(index=False))


if __name__ == "__main__":
    main()
