#!/usr/bin/env python3
"""Where do the Chinchilla runs under one token per parameter sit in the data? Data-only trace of the 245 rows of
data/data_hitchhikers_guide/colpret_chinchilla.csv.gz. Writes results/trace_*.csv and results/trace_summary.json.

What this can show: structure inside the 245 rows (FLOP columns, rank of N inside a column, loss relative to neighbours,
how tokens relate to FLOPs). What it cannot show: whether any single run was really trained that way. The original run
logs are not public; see provenance.md for the published description of how the 245 rows were obtained.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from py_src.ladders import load_chinchilla_runs  # noqa: E402

GAP_DEX = 0.05  # a gap larger than this in sorted log10(FLOP) starts a new column of runs
RES = HERE / "results"
RES.mkdir(exist_ok=True)

raw = pd.read_csv(REPO / "data" / "data_hitchhikers_guide" / "colpret_chinchilla.csv.gz", usecols=["flops", "num_params", "tokens_seen"])
dev = (raw.flops / (6.0 * raw.num_params * raw.tokens_seen) - 1.0).abs().max()

runs = load_chinchilla_runs().copy()
runs["log10C"] = np.log10(runs["C"])
order = runs.sort_values("log10C")
runs.loc[order.index, "column"] = np.concatenate([[0], np.cumsum(np.diff(order["log10C"].to_numpy()) > GAP_DEX)]).astype(int)
runs["column"] = runs["column"].astype(int)
runs["N_rank_in_column"] = runs.groupby("column")["N"].rank(ascending=False, method="first").astype(int)  # 1 = largest N
col = runs.groupby("column").agg(n_runs=("N", "size"), n_sizes=("model_name", "nunique"), C_min=("C", "min"), C_max=("C", "max"),
                                 N_min=("N", "min"), N_max=("N", "max"), min_ratio=("ratio", "min"),
                                 n_under_1=("ratio", lambda s: int((s < 1).sum()))).reset_index()
col["spread_dex"] = np.log10(col.C_max / col.C_min)
col.to_csv(RES / "trace_columns.csv", index=False)

short = runs[runs.ratio < 1.0].sort_values("ratio").copy()


def neighbour_loss_ratio(row):
    near = runs[(np.abs(runs.log10C - row.log10C) <= 0.1) & (runs.ratio >= 1.0)]
    return float(row.loss / near.loss.median()) if len(near) else np.nan


short["loss_over_median_loss_of_runs_within_0.1dex"] = short.apply(neighbour_loss_ratio, axis=1)
short = short.merge(col[["column", "n_runs", "C_min", "C_max"]].rename(columns={"n_runs": "column_n_runs", "C_min": "column_C_min", "C_max": "column_C_max"}), on="column")
short[["model_name", "N", "D", "C", "ratio", "loss", "column", "column_n_runs", "N_rank_in_column", "loss_over_median_loss_of_runs_within_0.1dex"]].to_csv(RES / "trace_short_runs.csv", index=False)

# Prediction 6 as written in README.md
c1e19 = col[(np.log10(col.C_min) >= 18.9) & (np.log10(col.C_max) <= 19.1)]
assert len(c1e19) == 1, c1e19
c1e19 = int(c1e19.column.iloc[0])
in_1e19 = short[short.column == c1e19]
top_n = runs[runs.column == c1e19].nlargest(len(in_1e19), "N").index
ten_are_largest = bool(set(in_1e19.index) == set(short.merge(runs[["model_name", "N", "D"]], on=["model_name", "N", "D"]).index) or True)
largest_ids = set(runs.loc[top_n, ["model_name", "D"]].apply(tuple, axis=1))
short_ids = set(in_1e19[["model_name", "D"]].apply(tuple, axis=1))
rest = short[short.column != c1e19]
p6 = {
    "short_runs": int(len(short)), "in_1e19_column": int(len(in_1e19)), "in_1e19_column_are_its_largest_N": bool(short_ids == largest_ids),
    "1e19_column_runs": int((runs.column == c1e19).sum()), "1e19_column_C_range": [float(runs[runs.column == c1e19].C.min()), float(runs[runs.column == c1e19].C.max())],
    "1e19_column_N_range_of_short": [float(in_1e19.N.min()), float(in_1e19.N.max())],
    "others": [{"N": float(r.N), "C": float(r.C), "ratio": float(r.ratio), "column_C_range": [float(r.column_C_min), float(r.column_C_max)], "N_rank_in_column": int(r.N_rank_in_column)}
               for r in rest.itertuples()],
}
p6["passed"] = bool(p6["in_1e19_column"] == 10 and p6["in_1e19_column_are_its_largest_N"] and len(rest) == 1
                    and abs(rest.iloc[0].N - 1.257e10) < 1e6 and rest.iloc[0].N_rank_in_column == 1
                    and 20.7 <= np.log10(rest.iloc[0].C) <= 20.9)
five = runs.nsmallest(5, "ratio")
summary = {
    "flops_equals_6ND_max_rel_dev": float(dev), "n_rows": int(len(runs)), "n_columns": int(len(col)), "gap_dex": GAP_DEX,
    "n_columns_with_8_or_more_runs": int((col.n_runs >= 8).sum()), "runs_in_those_columns": int(col[col.n_runs >= 8].n_runs.sum()),
    "D_min": float(runs.D.min()), "D_max": float(runs.D.max()), "n_D_below_5e9": int((runs.D < 5e9).sum()),
    "n_D_below_5e9_and_ratio_ge_1": int(((runs.D < 5e9) & (runs.ratio >= 1)).sum()), "n_short_D_below_5e9": int((short.D < 5e9).sum()),
    "short_D_min": float(short.D.min()), "short_D_max": float(short.D.max()),
    "five_lowest_ratios": [round(float(x), 4) for x in five.ratio], "five_lowest_N": [float(x) for x in five.N], "five_lowest_loss": [round(float(x), 3) for x in five.loss],
    "five_lowest_all_in_1e19_column": bool((five.column == c1e19).all()),
    "max_loss_over_neighbour_median_of_five": float(short.nsmallest(5, "ratio")["loss_over_median_loss_of_runs_within_0.1dex"].max()),
    "short_ratio_max": float(short.ratio.max()), "prediction_6": p6,
}
(RES / "trace_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2)); print(col.to_string()); print(short[["N", "D", "C", "ratio", "loss", "column", "N_rank_in_column", "loss_over_median_loss_of_runs_within_0.1dex"]].to_string())
