"""Load the public ladders shipped in this repo and make held-out splits.

Units, throughout:
- N: parameter count as recorded in the table (not recomputed)
- D: tokens seen
- C: training FLOPs. The Chinchilla, GPT-3, and Pythia tables satisfy C = 6 N D.
- loss: cross-entropy in nats, except Pythia, where the table has no pretraining
  loss and `loss` is ln(LAMBADA perplexity) = LAMBADA mean NLL in nats.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from py_src.conjecture_fits import assign_holdout_by_size, holdout_count

REPO = Path(__file__).resolve().parents[1]
HITCH = REPO / "data" / "data_hitchhikers_guide"


def load_chinchilla_runs() -> pd.DataFrame:
    raw = pd.read_csv(
        HITCH / "colpret_chinchilla.csv.gz",
        usecols=["model_name", "num_params", "tokens_seen", "flops", "loss"],
    )
    out = pd.DataFrame(
        {
            "dataset": "chinchilla",
            "model_name": raw["model_name"].astype(str),
            "N": raw["num_params"].to_numpy(dtype=np.float64),
            "D": raw["tokens_seen"].to_numpy(dtype=np.float64),
            "C": raw["flops"].to_numpy(dtype=np.float64),
            "loss": raw["loss"].to_numpy(dtype=np.float64),
        }
    )
    out = out[(out.N > 0) & (out.D > 0) & (out.loss > 0)].reset_index(drop=True)
    out["ratio"] = out["D"] / out["N"]
    return out


def chinchilla_matched(target_ratio: float = 20.0, factor: float = 1.5) -> pd.DataFrame:
    """One run per model size: the D/N closest to `target_ratio`, kept if within `factor`.

    The target is the literature Chinchilla ratio, not a value fit on these losses.
    """
    runs = load_chinchilla_runs()
    rows = []
    log_tol = np.log(factor)
    for n_value, sub in runs.groupby("N"):
        log_gap = np.abs(np.log(sub["ratio"].to_numpy()) - np.log(target_ratio))
        i = int(np.argmin(log_gap))
        if log_gap[i] > log_tol:
            continue
        row = sub.iloc[i]
        rows.append(
            {
                "dataset": f"chinchilla_matched_r{target_ratio:g}",
                "model_name": row["model_name"],
                "N": float(row["N"]),
                "D": float(row["D"]),
                "C": float(row["C"]),
                "loss": float(row["loss"]),
                "ratio": float(row["ratio"]),
                "d_id": "matched",
            }
        )
    out = pd.DataFrame(rows).sort_values("N").reset_index(drop=True)
    return out


def gpt3_binned(window: float = 0.10, min_count: int = 5, n_grid: int = 10) -> pd.DataFrame:
    """Median validation loss in a pre-specified token window around each grid point.

    Grid is log-spaced from 2e10 to 2.5e11 tokens, inside every GPT-3 model's
    observed range. `window` is a relative half-width: keep rows with
    |log D - log D_grid| <= log(1+window). D used below is the median token
    count inside the window, not the grid center.
    """
    raw = pd.read_csv(
        HITCH / "colpret_gpt3.csv.gz",
        usecols=["model_name", "num_params", "tokens_seen", "val_loss"],
    ).dropna(subset=["val_loss"])
    raw = raw[(raw["val_loss"] > 0) & (raw["tokens_seen"] > 0) & (raw["num_params"] > 0)]
    d_grid = np.geomspace(2e10, 2.5e11, n_grid)
    rows = []
    log_tol = np.log(1.0 + window)
    for name, sub in raw.groupby("model_name"):
        n_value = float(sub["num_params"].iloc[0])
        tokens = sub["tokens_seen"].to_numpy(dtype=np.float64)
        loss = sub["val_loss"].to_numpy(dtype=np.float64)
        for d_id, d_target in enumerate(d_grid):
            mask = np.abs(np.log(tokens) - np.log(d_target)) <= log_tol
            if int(mask.sum()) < min_count:
                continue
            d_med = float(np.median(tokens[mask]))
            loss_med = float(np.median(loss[mask]))
            q75, q25 = np.quantile(loss[mask], [0.75, 0.25])
            rows.append(
                {
                    "dataset": "gpt3_val",
                    "model_name": str(name),
                    "N": n_value,
                    "D": d_med,
                    "C": 6.0 * n_value * d_med,
                    "loss": loss_med,
                    "loss_iqr": float(q75 - q25),
                    "n_rows": int(mask.sum()),
                    "d_id": int(d_id),
                    "d_target": float(d_target),
                }
            )
    return pd.DataFrame(rows).sort_values(["d_id", "N"]).reset_index(drop=True)


def pythia_lambada(min_tokens: float = 1e9, max_ppl: float = 1e4) -> pd.DataFrame:
    """Pythia LAMBADA NLL. The ColPret Pythia table has no pretraining cross-entropy."""
    raw = pd.read_csv(
        HITCH / "colpret_pythia.csv.gz",
        usecols=["model_name", "num_params", "tokens_seen", "flops", "lambada_openai ppl"],
    ).dropna(subset=["lambada_openai ppl"])
    raw = raw[(raw["tokens_seen"] >= min_tokens) & (raw["lambada_openai ppl"] > 1.0) & (raw["lambada_openai ppl"] < max_ppl)]
    raw = raw[(raw["num_params"] > 0) & (raw["flops"] > 0)]
    # Shared checkpoints only, so each d_id is the same token count on every model.
    tokens = raw["tokens_seen"].round().astype(np.int64)
    raw = raw.assign(token_key=tokens)
    counts = raw.groupby("token_key")["model_name"].nunique()
    n_models = raw["model_name"].nunique()
    keep = set(counts[counts == n_models].index)
    raw = raw[raw["token_key"].isin(keep)].copy()
    keys = sorted(raw["token_key"].unique())
    key_to_id = {k: i for i, k in enumerate(keys)}
    out = pd.DataFrame(
        {
            "dataset": "pythia_lambada",
            "model_name": raw["model_name"].astype(str).to_numpy(),
            "N": raw["num_params"].to_numpy(dtype=np.float64),
            "D": raw["tokens_seen"].to_numpy(dtype=np.float64),
            "C": raw["flops"].to_numpy(dtype=np.float64),
            "ppl": raw["lambada_openai ppl"].to_numpy(dtype=np.float64),
            "d_id": raw["token_key"].map(key_to_id).to_numpy(),
        }
    )
    out["loss"] = np.log(out["ppl"])
    return out.sort_values(["d_id", "N"]).reset_index(drop=True)


def with_size_holdout(df: pd.DataFrame) -> pd.DataFrame:
    out, _ = assign_holdout_by_size(df, "N")
    return out


def coverage_note(df: pd.DataFrame) -> dict:
    """How many nominal slices contain every model size."""
    if "d_id" not in df.columns:
        return {"n_models": int(df["N"].nunique()), "n_full_slices": 1}
    n_models = int(df["N"].nunique())
    full = []
    for d_id, sub in df.groupby("d_id"):
        if sub["N"].nunique() == n_models:
            full.append(d_id)
    return {
        "n_models": n_models,
        "n_rows": int(len(df)),
        "n_slices": int(df["d_id"].nunique()),
        "n_full_slices": int(len(full)),
        "holdout": int(holdout_count(n_models)),
    }
