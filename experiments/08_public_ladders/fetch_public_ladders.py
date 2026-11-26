"""Download public ladders the ColPret export does not contain, and write csv.gz files.

Sources, all fetched at runtime:
- Epoch AI svg extraction of Hoffmann et al. Figure 4
  https://github.com/epoch-research/analyzing-chinchilla
- Pythia final Pile validation loss from public W&B histories mirrored at
  https://github.com/tecunningham/open-labs (data/losses/eleutherai)
- Porian et al. runs already in data/data_resolving_discrepancies (local pickle)
- mlfoundations/scaling C4 models (104 released runs, c4 validation loss)

N for Porian is the non-embedding proxy 12 * depth * width^2, not an exact count.
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

OUT = REPO / "data" / "public_ladders"
EPOCH_URL = "https://raw.githubusercontent.com/epoch-research/analyzing-chinchilla/main/data/svg_extracted_data.csv"
PYTHIA_MODELS = {
    "pythia-70m": 70e6,
    "pythia-160m": 160e6,
    "pythia-410m": 410e6,
    "pythia-1b": 1.0e9,
    "pythia-1.4b": 1.4e9,
    "pythia-2.8b": 2.8e9,
    "pythia-6.9b": 6.9e9,
    "pythia-12b": 12e9,
}
PYTHIA_BASE = "https://raw.githubusercontent.com/tecunningham/open-labs/main/data/losses/eleutherai"
SCALING_API = "https://api.github.com/repos/mlfoundations/scaling/contents/exp_data/models?ref=main"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "dark-matter-fetch"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _write(df: pd.DataFrame, name: str) -> None:
    path = OUT / name
    df.to_csv(path, index=False, compression="gzip")
    print(f"wrote {path.relative_to(REPO)} rows={len(df)}", flush=True)


def epoch_table() -> pd.DataFrame:
    raw = pd.read_csv(io.BytesIO(_get(EPOCH_URL)))
    n = raw["Model Size"].to_numpy(dtype=np.float64)
    c = raw["Training FLOP"].to_numpy(dtype=np.float64)
    out = pd.DataFrame(
        {
            "dataset": "epoch_chinchilla_figure4",
            "model_name": [f"epoch_{i}" for i in range(len(raw))],
            "N": n,
            "C": c,
            "D": c / (6.0 * n),
            "loss": raw["loss"].to_numpy(dtype=np.float64),
        }
    )
    out = out[(out.N > 0) & (out.D > 0) & (out.loss > 0)].reset_index(drop=True)
    out["ratio"] = out["D"] / out["N"]
    return out


def matched_ratio(df: pd.DataFrame, dataset: str, target: float = 20.0, factor: float = 1.5) -> pd.DataFrame:
    rows = []
    tol = np.log(factor)
    for n_value, sub in df.groupby("N"):
        gap = np.abs(np.log(sub["ratio"].to_numpy()) - np.log(target))
        i = int(np.argmin(gap))
        if gap[i] > tol:
            continue
        row = sub.iloc[i]
        rows.append(
            {
                "dataset": dataset,
                "model_name": row["model_name"],
                "N": float(row["N"]),
                "D": float(row["D"]),
                "C": float(row["C"]),
                "loss": float(row["loss"]),
                "ratio": float(row["ratio"]),
                "d_id": "matched",
            }
        )
    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)


def pythia_final() -> pd.DataFrame:
    rows = []
    for name, n_params in PYTHIA_MODELS.items():
        meta = json.loads(_get(f"{PYTHIA_BASE}/{name}.meta.json"))
        loss = float(meta["final_loss"]["validation/lm_loss"])
        tokens = float(meta["final_tokens"])
        rows.append(
            {
                "dataset": "pythia_pile_val",
                "model_name": name,
                "N": float(n_params),
                "D": tokens,
                "C": 6.0 * float(n_params) * tokens,
                "loss": loss,
                "ratio": tokens / float(n_params),
                "d_id": "final",
                "source_run": meta.get("run_id", name),
            }
        )
    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)


def porian_matched(dataset_name: str) -> pd.DataFrame:
    pickle_path = REPO / "data" / "data_resolving_discrepancies" / "experiment_results.pickle.xz"
    df = pd.read_pickle(pickle_path)
    sub = df[(df["dataset"] == dataset_name) & (df["hparams"] == "base") & (df["decay"] == "chinchilla") & (df["warmup"] == "short")].copy()
    sub["N"] = 12.0 * sub["depth"].astype(float) * sub["width"].astype(float) ** 2
    sub["D"] = sub["max_step"].astype(float) * sub["seq_len"].astype(float) * sub["bs"].astype(float)
    sub["C"] = 6.0 * sub["N"] * sub["D"]
    sub["loss"] = sub["val/loss"].map(lambda s: float(s.dropna().iloc[-1]))
    sub["ratio"] = sub["D"] / sub["N"]
    sub["model_name"] = [f"porian_{dataset_name}_{i}" for i in range(len(sub))]
    return matched_ratio(sub, f"porian_{dataset_name}_matched_r20")


def scaling_c4_matched() -> pd.DataFrame:
    listing = json.loads(_get(SCALING_API))
    rows = []
    for item in listing:
        if not item["name"].startswith("c4_original-"):
            continue
        payload = json.loads(_get(item["download_url"]))
        hp = payload["hyperparameters"]
        c4 = [r for r in payload["results"] if any("c4_val" in str(v) for v in r.get("val_data", []))]
        if not c4:
            continue
        # The long C4 validation shard, not the tiny multiplier probe.
        chosen = max(c4, key=lambda r: float(r.get("tokens", 0)))
        n_params = float(hp["params_no_embed"])
        tokens = float(hp["tokens"])
        rows.append(
            {
                "model_name": payload["name"],
                "N": n_params,
                "D": tokens,
                "C": 6.0 * n_params * tokens,
                "loss": float(chosen["loss"]),
                "ratio": tokens / n_params,
                "multiplier": float(hp.get("chinchilla_multiplier", np.nan)),
            }
        )
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "scaling_c4_all_runs.csv.gz", index=False, compression="gzip")
    print(f"c4 runs {len(raw)} unique N {raw['N'].nunique()}", flush=True)
    return matched_ratio(raw, "scaling_c4_matched_r20")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    epoch = epoch_table()
    _write(epoch, "epoch_chinchilla_runs.csv.gz")
    epoch_matched = matched_ratio(epoch, "epoch_chinchilla_matched_r20")
    _write(epoch_matched, "epoch_chinchilla_matched_r20.csv.gz")
    pythia = pythia_final()
    _write(pythia, "pythia_pile_val_final.csv.gz")
    for name in ("rw", "owt2"):
        matched = porian_matched(name)
        _write(matched, f"porian_{name}_matched_r20.csv.gz")
    c4 = scaling_c4_matched()
    _write(c4, "scaling_c4_matched_r20.csv.gz")


if __name__ == "__main__":
    main()
