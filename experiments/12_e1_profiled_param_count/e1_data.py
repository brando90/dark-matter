"""E1 ladders (PREREG.md, 11-25-2026): one (N_nonemb, N_emb, y) row per size for every ladder with embedding counts.

    python e1_data.py        # writes results/ladders.csv and results/pythia_counts.csv
"""
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DM2 = os.path.join(ROOT, "experiments", "dm2_real_ladders")
PYTHIA = os.path.join(ROOT, "data", "public_ladders", "pythia_pile_val_final.csv.gz")
PYTHIA_CFG = os.path.join(HERE, "data", "pythia_hf_configs.csv")
OUT = os.path.join(HERE, "results", "ladders.csv")
PORIAN_V = 50432
C_STD = {"porian-rw": 2.0, "porian-owt2": 2.0, "chinchilla": 1.0, "pythia": 2.0}
LADDERS = list(C_STD)


def pythia_counts():
    """GPT-NeoX counts from the HF configs: per layer 12 d^2 + 13 d (qkv, dense, 4d MLP, two LayerNorms), final
    LayerNorm 2 d; embedding block V d (untied: input embedding and output head are each V d)."""
    cfg = pd.read_csv(PYTHIA_CFG)
    assert (cfg.intermediate_size == 4 * cfg.hidden_size).all()
    d, L, V = cfg.hidden_size.astype(float), cfg.num_hidden_layers.astype(float), cfg.vocab_size.astype(float)
    cfg["N_nonemb"] = L * (12 * d * d + 13 * d) + 2 * d
    cfg["N_emb"] = V * d
    cfg["N_total_untied"] = cfg.N_nonemb + 2 * cfg.N_emb
    return cfg


def build():
    cur = pd.read_csv(os.path.join(DM2, "results", "curves.csv"))
    cur = cur[cur["slice"] == "D=20N"].dropna(subset=["y"])
    rows = []
    for ds in ("rw", "owt2"):
        g = cur[cur.curve == f"porian/{ds}/tuned-short-const"]
        for _, r in g.iterrows():
            rows.append(dict(ladder=f"porian-{ds}", name=f"w{int(r.width)}", N_nonemb=r.N_nonemb,
                             N_emb=PORIAN_V * r.width, y=r.y, slice="D=20N (DM-2 N_total)"))
    g = cur[cur.curve == "chinchilla/hoffmann-epoch"]
    for _, r in g.iterrows():
        rows.append(dict(ladder="chinchilla", name=f"{r.N_name:.4g}", N_nonemb=r.N_nonemb,
                         N_emb=r.N_total - r.N_nonemb, y=r.y, slice="D=20N interpolated (DM-2)"))
    cfg = pythia_counts()
    py = pd.read_csv(PYTHIA).merge(cfg, left_on="model_name", right_on="model", how="left", validate="1:1")
    assert py.N_nonemb.notna().all()
    for _, r in py.iterrows():
        rows.append(dict(ladder="pythia", name=r.model_name, N_nonemb=r.N_nonemb, N_emb=r.N_emb, y=r.loss,
                         slice="fixed D = 2.999e11 tokens, final Pile val loss"))
    df = pd.DataFrame(rows)
    df["c_std"] = df.ladder.map(C_STD)
    df["N_std"] = df.N_nonemb + df.c_std * df.N_emb
    df = df.sort_values(["ladder", "N_std"]).reset_index(drop=True)
    cfg = cfg.merge(pd.read_csv(PYTHIA)[["model_name", "N"]], left_on="model", right_on="model_name")
    cfg["total_over_nominal"] = cfg.N_total_untied / cfg.N
    return df, cfg


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df, cfg = build()
    df.to_csv(OUT, index=False)
    cfg.to_csv(os.path.join(HERE, "results", "pythia_counts.csv"), index=False)
    print(df.groupby("ladder").agg(n=("y", "size"), Nstd_min=("N_std", "min"), Nstd_max=("N_std", "max"),
                                   emb_share_min=("N_emb", "min")).to_string())
    for lad, g in df.groupby("ladder"):
        print(lad, "embedding share of N_std (smallest, largest):",
              np.round((g.c_std * g.N_emb / g.N_std).values[[0, -1]], 3),
              "order by N_nonemb == order by N_std:", bool((np.argsort(g.N_nonemb.values) == np.arange(len(g))).all()))
    print(cfg[["model", "N_nonemb", "N_emb", "N_total_untied", "N", "total_over_nominal"]].to_string())
