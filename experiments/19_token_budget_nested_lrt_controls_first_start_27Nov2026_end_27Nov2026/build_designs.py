"""Extract the tuned-constant (TC) design grids of Porian et al. (rw and owt2) for folder 19.

One row per (size, tokens-per-parameter slice): tied parameter count, tokens, the observed validation loss, and the noise sd
used as weight. The loss is read exactly as folder 14 reads it (14_e2_*/e2lib.py ladder(), recipe TS, slices 5, 10, 20, 40, 80).
The sd is DM-2's seed sd (dm2_real_ladders/results/seed_sigma.csv, 7 seeds of the tuned recipe) interpolated linearly in
log width on log sd between the measured widths 96 to 480, and held at the width-480 value above 480 (folder 14 does the
same for widths 640 and 832). Nothing is fitted here. Run from anywhere: python3 -I build_designs.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
E14 = os.path.join(HERE, "..", "14_e2_token_budget_hparams_small_end")
sys.path.insert(0, E14)
import e2lib as L  # noqa: E402

SLICES = [5, 10, 20, 40, 80]


def seed_sd_interp(ds):
    t = pd.read_csv(os.path.join(L.DM2, "results", "seed_sigma.csv"))
    t = t[t.dataset == ds].sort_values("width")
    lw, ls = np.log(t.width.values), np.log(t.sd.values)
    return lambda w: float(np.exp(np.interp(np.log(w), lw, ls)))


def main():
    out = os.path.join(HERE, "results")
    for ds in ("rw", "owt2"):
        sd = seed_sd_interp(ds)
        rows = []
        for k in SLICES:
            for r in L.ladder(ds, "TS", k).itertuples():
                rows.append(dict(dataset=ds, k=k, width=int(r.width), N_tied=r.N_tied, D=r.D, y=r.y, sd=sd(r.width)))
        df = pd.DataFrame(rows).sort_values(["k", "N_tied"]).reset_index(drop=True)
        df.to_csv(os.path.join(out, f"design_{ds}.csv"), index=False)
        print(ds, len(df), "points; N range", df.N_tied.min(), df.N_tied.max(), "; sd range", df.sd.min(), df.sd.max())


if __name__ == "__main__":
    main()
