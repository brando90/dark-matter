"""Shared E2/E4 helpers (11-25-2026): Porian ladders under every convention and count, 1-term fits, reverse ladder.

Reuses (import only): dm2_real_ladders/data.py (loader, counts, curve readers) and fitlib.py (joint fit),
dm1_power_analysis/dm1b.py via 12_e1_profiled_param_count/e1lib.py (DM-1b fitters).
Law convention: alpha is the FAST exponent. DM-1b's fit output names the fast exponent `beta`; e1lib.two_term_named
translates.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DM2 = os.path.join(ROOT, "experiments", "dm2_real_ladders")
E1 = os.path.join(ROOT, "experiments", "12_e1_profiled_param_count")
for p in (DM2, E1):
    if p not in sys.path:
        sys.path.insert(0, p)
import data as D  # noqa: E402
import fitlib as F  # noqa: E402
import e1lib  # noqa: E402  (also puts dm5_small_to_large on sys.path: dm1, dm1b, dm5)
import dm1b  # noqa: E402

V = D.PORIAN_VOCAB
N_REF = 1e6
REF_MIN = 1e8  # reference sizes for the reverse ladder: tied N >= 1e8
SMALL = {96: "5M", 128: "7M", 160: "9M", 224: "15M"}
VERDICT_SIZES = [96, 160, 224]
COUNTS = {"untied": 2.0, "nonemb": 0.0, "tied": 1.0}  # c in N = N_nonemb + c V w
RECIPES = {"TS": ("tuned", "short", "const"), "TL": ("tuned", "long", "const"),
           "BC": ("base", "short", "chinchilla"), "BSK": ("base", "short", "kaplan"),
           "BLK": ("base", "long", "kaplan")}
SLICES = {"TS": [5, 10, 20, 40, 80, "final"], "TL": [5, 10, 20, 40, 80, "final"], "BC": [5, 10, 20, 40, "final"],
          "BSK": ["final"], "BLK": ["final"]}


def count(c, ne, w):
    return np.asarray(ne, float) + c * V * np.asarray(w, float)


_P = None


def porian():
    global _P
    if _P is None:
        _P = D.load_porian()
        _P["tpp"] = _P.final_tokens / _P.N_total
    return _P


def ladder(ds, recipe, sl):
    """One y per width for (dataset, recipe, slice); D/N on the tied count (DM-2 N_total). Rows with no value are
    dropped. Columns: width, N_nonemb, N_tied, y, D (tokens at the point), how."""
    P = porian()
    hp, wu, dec = RECIPES[recipe]
    sub = P[(P.dataset == ds) & (P.hparams == hp) & (P.warmup == wu) & (P.decay == dec)]
    rows = []
    for w, g in sub.groupby("width"):
        r0 = g.iloc[0]
        longest = g.loc[g.final_tokens.idxmax()]
        if sl == "final":
            y, Dtok = longest.final_loss, longest.final_tokens
        elif dec == "chinchilla":
            y, Dtok = D.interp_log(sl, g.tpp, g.final_loss), sl * r0.N_total
        elif dec == "const":
            assert len(g) == 1
            y, Dtok = D.porian_curve_value(longest, ratio=sl), sl * r0.N_total
        else:
            raise ValueError("kaplan runs have no matched-D/N slice")
        if np.isfinite(y):
            rows.append(dict(width=float(w), N_nonemb=r0.N_nonemb, N_tied=r0.N_total, y=float(y), D=float(Dtok)))
    return pd.DataFrame(rows).sort_values("N_tied").reset_index(drop=True)


def conventions():
    return [(ds, rc, sl) for ds in ("rw", "owt2") for rc in RECIPES for sl in SLICES[rc]]


def seed_sd():
    """{(dataset, width): seed sd} from DM-2's seed runs (7 seeds of the tuned recipe); w640 / w832 get w480's."""
    s = pd.read_csv(os.path.join(DM2, "results", "seed_sigma.csv"))
    out = {(r.dataset, float(r.width)): float(r.sd) for r in s.itertuples()}
    for ds in ("rw", "owt2"):
        for w in (640.0, 832.0):
            out[(ds, w)] = out[(ds, 480.0)]
    return out


def fit1(n, y):
    return e1lib.fit1(np.asarray(n, float), np.asarray(y, float))


def fit1_many(n, Y):
    return e1lib.fit1_vec(np.asarray(n, float), np.asarray(Y, float))


def pred1(f, n):
    return e1lib.pred1(f, n)


def reverse(N, y, ref_mask):
    """1-term on the reference sizes, residual y - yhat at every size; None if < 4 reference sizes."""
    if ref_mask.sum() < 4:
        return None, None
    n = np.asarray(N, float) / N_REF
    f = fit1(n[ref_mask], y[ref_mask])
    return y - pred1(f, n), f
