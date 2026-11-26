"""E1 reproduction gate (PREREG.md): DM-3's Porian A3 profile with DM-3's grid and fitter, and with E1's fitter.

    python repro_dm3.py      # writes results/repro_dm3_profile.csv and results/repro_dm3.csv, prints PASS/FAIL
"""
import os
import sys

import numpy as np
import pandas as pd

import e1lib as L

DM2 = os.path.join(L.ROOT, "experiments", "dm2_real_ladders")
sys.path.insert(0, DM2)
import dm3  # noqa: E402
import fitlib as F  # noqa: E402

OUT = os.path.join(L.HERE, "results")
REF = {"rw": dict(c1=2.0, set=[(0.05, 0.11), (1.55, 2.0)], dbic=19.5),
       "owt2": dict(c1=0.045, set=[(0.03, 0.08), (1.83, 2.0)], dbic=20.8)}


def main():
    lad = pd.read_csv(os.path.join(OUT, "ladders.csv"))
    prof_ref = pd.read_csv(os.path.join(DM2, "results", "dm3_profileA.csv"))
    rows, prof = [], []
    for ds in ("rw", "owt2"):
        g = lad[lad.ladder == f"porian-{ds}"].sort_values("N_std")
        ne, nemb, y = g.N_nonemb.values, g.N_emb.values, g.y.values
        nt = ne + nemb
        npts = len(y)
        r1, r1v, r2 = [], [], []
        for c in dm3.C_GRID:
            n = dm3.n_of(c, nt, ne)
            f1 = F.fit1(n, y)
            r1.append(f1["rss"])
            r2.append(F.fit2(n, y, f1)["rss"])
            r1v.append(L.fit1(L.n_of(c, ne, nemb), y)["rss"])
        r1, r1v, r2 = map(np.array, (r1, r1v, r2))
        ref = prof_ref[prof_ref.dataset == ds].sort_values("c")
        assert np.allclose(ref.c.values, dm3.C_GRID)
        rel = np.abs(r1 / ref.rss1.values - 1)
        relv = np.abs(r1v / ref.rss1.values - 1)
        c1, v1 = L.refine(lambda c: F.fit1(dm3.n_of(c, nt, ne), y)["rss"], dm3.C_GRID, r1)
        c2, v2 = L.refine(lambda c: F.fit2(dm3.n_of(c, nt, ne), y, F.fit1(dm3.n_of(c, nt, ne), y))["rss"],
                          dm3.C_GRID, r2)
        s = L.profile_set(dm3.C_GRID, r1, v1, npts)
        dbic = F.bic(v1, npts, 4) - F.bic(v2, npts, 6)
        edges = [x for a, b, _ in s["intervals"] for x in (a, b)]
        ref_edges = [x for a, b in REF[ds]["set"] for x in (a, b)]
        edge_err = max(abs(a - b) for a, b in zip(edges, ref_edges)) if len(edges) == len(ref_edges) else np.inf
        checks = dict(rss1_fitlib_max_rel=float(rel.max()), rss1_e1fitter_max_rel=float(relv.max()),
                      c1=c1, c1_ref=REF[ds]["c1"], c1_set=s["text"], set_edge_max_err=float(edge_err),
                      dbic_free=dbic, dbic_ref=REF[ds]["dbic"], c2=c2)
        checks["pass"] = bool(rel.max() <= 1e-6 and relv.max() <= 1e-4 and abs(c1 - REF[ds]["c1"]) <= 0.01
                              and edge_err <= 0.02 and abs(dbic - REF[ds]["dbic"]) <= 0.5)
        rows.append(dict(dataset=ds, **checks))
        prof.append(pd.DataFrame(dict(dataset=ds, c=dm3.C_GRID, rss1_fitlib=r1, rss1_e1=r1v, rss2_fitlib=r2,
                                      rss1_dm3=ref.rss1.values)))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT, "repro_dm3.csv"), index=False)
    pd.concat(prof).to_csv(os.path.join(OUT, "repro_dm3_profile.csv"), index=False)
    print(out.to_string())
    print("REPRODUCTION", "PASS" if out["pass"].all() else "FAIL")


if __name__ == "__main__":
    main()
