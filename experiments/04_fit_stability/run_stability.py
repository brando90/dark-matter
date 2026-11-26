"""Identifiability: multi-start dispersion, bootstrap, Huber-delta sweep.

Naming: for the N-only two-term law in this script, alpha is the fast
exponent and beta is the slow one (beta < alpha). The joint law's data
exponent is gamma (Hoffmann's beta) and its data coefficient is G.
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
    literature_joint_start,
    predict,
)
from py_src.ladders import chinchilla_matched, load_chinchilla_runs  # noqa: E402
from py_src.scaling_laws_fit import huber  # noqa: E402

OUT = Path(__file__).resolve().parent / "outputs"
N_BOOT = 30
PUBLISHED = {"E": 1.82, "A": 482.0, "alpha": 0.34, "G": 2085.0, "gamma": 0.36}


def _huber_obj(y, yhat, delta=1e-3) -> float:
    r = np.log(yhat) - np.log(y)
    return float(np.sum(huber(r, delta=delta)))


def _summarize(samples: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in cols:
        v = samples[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) == 0:
            continue
        rows.append(
            {
                "parameter": col,
                "n": int(len(v)),
                "median": float(np.median(v)),
                "p10": float(np.percentile(v, 10)),
                "p90": float(np.percentile(v, 90)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assert_self_check()
    runs = load_chinchilla_runs()
    matched = chinchilla_matched(20.0)

    joint = fit_joint(runs["N"], runs["D"], runs["loss"], n_starts=24, seed=0, record_starts=True)
    starts = pd.DataFrame(joint.diagnostics["starts"])
    starts["near_best"] = starts["objective"] <= joint.objective * 1.01 + 1e-12
    starts.to_csv(OUT / "joint_multistart.csv", index=False)

    published_yhat = predict("joint", PUBLISHED, N=runs["N"], D=runs["D"])
    published_obj = _huber_obj(runs["loss"].to_numpy(), published_yhat)
    from_published = fit_joint(
        runs["N"], runs["D"], runs["loss"], n_starts=0, seed=0, extra_starts=[literature_joint_start()]
    )
    pd.DataFrame(
        [
            {"which": "best_of_24_starts", "objective": joint.objective, **joint.params, "success": joint.success},
            {"which": "published_point_not_optimized", "objective": published_obj, **PUBLISHED, "success": True},
            {
                "which": "optimized_from_published_start",
                "objective": from_published.objective,
                **from_published.params,
                "success": from_published.success,
            },
        ]
    ).to_csv(OUT / "published_comparison.csv", index=False)

    deltas = [1e-4, 1e-3, 1e-2, 1e-1, 10.0]
    delta_rows = []
    for delta in deltas:
        fit = fit_joint(runs["N"], runs["D"], runs["loss"], delta=delta, n_starts=8, seed=0)
        delta_rows.append({"delta": delta, "objective": fit.objective, "success": fit.success, **fit.params})
    pd.DataFrame(delta_rows).to_csv(OUT / "huber_delta_sweep.csv", index=False)

    rng = np.random.default_rng(0)
    boot_rows = []
    n = len(runs)
    N = runs["N"].to_numpy()
    D = runs["D"].to_numpy()
    L = runs["loss"].to_numpy()
    for b in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        fit = fit_joint(N[idx], D[idx], L[idx], n_starts=4, seed=1000 + b)
        boot_rows.append({"replicate": b, "success": fit.success, "objective": fit.objective, **fit.params})
        print(f"joint boot {b+1}/{N_BOOT}", flush=True)
    boot = pd.DataFrame(boot_rows)
    boot.to_csv(OUT / "joint_bootstrap.csv", index=False)
    summary = _summarize(boot, ["E", "A", "alpha", "G", "gamma"])
    summary.insert(0, "model", "joint_chinchilla_runs")
    # Ridge between prefactor and exponent.
    ok = boot[np.isfinite(boot["A"]) & np.isfinite(boot["alpha"]) & (boot["A"] > 0)]
    corr_rows = [
        {
            "pair": "logA_alpha",
            "corr": float(np.corrcoef(np.log(ok["A"]), ok["alpha"])[0, 1]),
            "n": int(len(ok)),
        }
    ]
    okg = boot[np.isfinite(boot["G"]) & np.isfinite(boot["gamma"]) & (boot["G"] > 0)]
    corr_rows.append(
        {
            "pair": "logG_gamma",
            "corr": float(np.corrcoef(np.log(okg["G"]), okg["gamma"])[0, 1]),
            "n": int(len(okg)),
        }
    )

    two_rows = []
    mN = matched["N"].to_numpy()
    mL = matched["loss"].to_numpy()
    for b in range(N_BOOT):
        idx = rng.integers(0, len(matched), size=len(matched))
        fit = fit_power("two_term", mN[idx], mL[idx], n_starts=4, seed=2000 + b)
        a_term = fit.params.get("A", np.nan) * np.median(mN) ** (-fit.params.get("alpha", np.nan))
        b_term = fit.params.get("B", np.nan) * np.median(mN) ** (-fit.params.get("beta", np.nan))
        frac = b_term / (a_term + b_term) if np.isfinite(a_term + b_term) and (a_term + b_term) != 0 else np.nan
        two_rows.append(
            {
                "replicate": b,
                "success": fit.success,
                "collapsed_to_single": fit.diagnostics.get("collapsed_to_single", False),
                "second_fraction_at_median_N": frac,
                **fit.params,
            }
        )
        print(f"two-term boot {b+1}/{N_BOOT}", flush=True)
    two = pd.DataFrame(two_rows)
    two.to_csv(OUT / "two_term_bootstrap.csv", index=False)
    two_sum = _summarize(two, ["E", "A", "alpha", "B", "beta", "second_fraction_at_median_N"])
    two_sum.insert(0, "model", "two_term_chinchilla_matched_r20")
    pd.concat([summary, two_sum], ignore_index=True).to_csv(OUT / "bootstrap_summary.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(OUT / "bootstrap_correlations.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    axes[0].hist(boot["alpha"].dropna(), bins=12, color="#0072B2", edgecolor="white")
    axes[0].axvline(PUBLISHED["alpha"], color="black", ls="--", label="published 0.34")
    axes[0].set_xlabel(r"joint $\alpha$ (parameter exponent)")
    axes[0].set_ylabel("bootstrap replicates")
    axes[0].legend(fontsize=8)
    axes[1].hist(boot["gamma"].dropna(), bins=12, color="#D55E00", edgecolor="white")
    axes[1].axvline(PUBLISHED["gamma"], color="black", ls="--", label="published 0.36")
    axes[1].set_xlabel(r"joint $\gamma$ (token exponent)")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Chinchilla joint fit, {N_BOOT} row-resamples")
    fig.tight_layout()
    fig.savefig(OUT / "joint_bootstrap_exponents.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.scatter(two["alpha"], two["beta"], c=np.where(two["collapsed_to_single"], "#999999", "#D55E00"), s=28)
    lim = max(two["alpha"].max(), two["beta"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label=r"$\beta=\alpha$")
    ax.set_xlabel(r"fast exponent $\alpha$")
    ax.set_ylabel(r"slow exponent $\beta$")
    ax.set_title("Two-term bootstrap on D/N ≈ 20")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "two_term_bootstrap.png", dpi=130)
    plt.close(fig)

    near = starts[starts["near_best"]]
    print("joint best", joint.params, "obj", joint.objective, "success", joint.success)
    print("published obj", published_obj, "from published", from_published.objective, from_published.params)
    print("starts", len(starts), "within 1% obj", int(near.shape[0]), "alpha range", near["alpha"].min(), near["alpha"].max())
    print("boot success", boot.success.mean(), "two collapsed rate", two.collapsed_to_single.mean())
    print(summary.to_string(index=False))
    print(two_sum.to_string(index=False))
    print(pd.DataFrame(corr_rows).to_string(index=False))


if __name__ == "__main__":
    main()
