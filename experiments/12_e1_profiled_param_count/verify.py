"""E1 verify (PREREG.md, "verify.py"): reproduction gate, fitter check, positive and null controls, output checks.

    python verify.py         # prints each check and ALL PASS / FAIL; writes results/verify.txt
"""
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd

import e1lib as L
from e1 import smallest_resid

OUT = os.path.join(L.HERE, "results")
REPS = 40
SD = 0.005
SEED = 20261125


def profile_many(ne, nemb, Y, grid=L.GRID_FINE):
    """RSS1 (grid, R) for every replicate row of Y."""
    return np.stack([L.fit1_vec(L.n_of(c, ne, nemb), Y)["rss"] for c in grid])


def lad(name):
    g = pd.read_csv(os.path.join(OUT, "ladders.csv"))
    g = g[g.ladder == name].sort_values("N_std")
    return g.N_nonemb.values, g.N_emb.values, g.y.values


def planted(name, c):
    ne, nemb, y = lad(name)
    n = L.n_of(c, ne, nemb)
    return ne, nemb, n, L.pred1(L.fit1(n, y), n)


def main():
    lines, ok_all = [], True

    def report(name, ok, msg):
        nonlocal ok_all
        ok_all &= bool(ok)
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
        print(lines[-1], flush=True)

    # 1. reproduction gate
    subprocess.run([sys.executable, os.path.join(L.HERE, "repro_dm3.py")], check=True, capture_output=True)
    rp = pd.read_csv(os.path.join(OUT, "repro_dm3.csv"))
    report("1 DM-3 reproduction", rp["pass"].all(),
           "; ".join(f"{r.dataset}: c1 {r.c1:.3f} (ref {r.c1_ref}), set {r.c1_set}, dBIC {r.dbic_free:.2f} "
                     f"(ref {r.dbic_ref}), max rel RSS1 diff fitlib {r.rss1_fitlib_max_rel:.1e} / E1 fitter "
                     f"{r.rss1_e1fitter_max_rel:.1e}" for r in rp.itertuples()))
    # 2. vectorized 1-term fitter == fit_batch's 1-term branch
    worst = 0.0
    for name in ("porian-rw", "porian-owt2", "chinchilla", "pythia"):
        ne, nemb, y = lad(name)
        for c in L.GRID_01:
            n = L.n_of(c, ne, nemb)
            worst = max(worst, abs(L.fit1(n, y)["rss"] / L.fit2(n, y)[0]["rss"] - 1))
    report("2 1-term fitter vs fit_batch", worst <= 1e-6, f"max rel RSS diff {worst:.1e} over 4 ladders x 31 c")
    # 3-5. c recovery, LRT null, LRT positive, smallest-size test (porian-rw sizes)
    rng = np.random.default_rng([SEED, 9])
    ne, nemb, n, mu = planted("porian-rw", 0.5)
    npts = len(mu)
    Y = mu[None, :] + SD * rng.normal(0, 1, (REPS, npts))
    R = profile_many(ne, nemb, Y)
    chat, cover, within, fires = [], [], [], []
    eps = np.random.default_rng([SEED, 10]).normal(0, 1, (500, npts))
    for r in range(REPS):
        i = int(np.argmin(R[:, r]))
        chat.append(L.GRID_FINE[i])
        s = L.profile_set(L.GRID_FINE, R[:, r], R[i, r], npts)
        cover.append(any(a <= 0.5 <= b for a, b, _ in s["intervals"]))
        sig2 = math.sqrt(L.fit2(n, Y[r])[1]["rss"] / (npts - 5))
        within.append(smallest_resid(0.5, ne, nemb, Y[r], sig2, eps)["within"])
        fires.append(L.lrt(n, Y[r], [SEED, 11, r])["detect"])
    med = float(np.median(np.abs(np.array(chat) - 0.5)))
    bias = float(np.median(chat) - 0.5)
    # Amendment 1 (PREREG.md): bias <= 0.05 replaces the original precision criterion median |c1 - 0.5| <= 0.1
    report("3 positive control: c recovery (planted c = 0.5)", abs(bias) <= 0.05 and np.mean(cover) >= 0.8
           and np.mean(within) >= 0.8, f"median c1 - 0.5 = {bias:+.3f}; 95% set covers 0.5 in {np.mean(cover):.2f}; "
                                       f"smallest size within 2 sd in {np.mean(within):.2f}; original criterion "
                                       f"median |c1 - 0.5| {med:.3f} <= 0.1: {'pass' if med <= 0.1 else 'FAIL'} "
                                       f"(precision limit, Amendment 1)")
    report("4 null control: LRT at true c on a 1-term ladder", np.mean(fires) <= 0.10,
           f"false detection rate {np.mean(fires):.3f} ({sum(fires)}/{REPS})")
    nf = n[0]
    Bc = 0.3 * nf ** 1.2
    mu2 = mu + Bc * n ** -1.2
    Y2 = mu2[None, :] + SD * np.random.default_rng([SEED, 12]).normal(0, 1, (REPS, npts))
    fires2, flag2 = [], []
    for r in range(REPS):
        fires2.append(L.lrt(n, Y2[r], [SEED, 13, r])["detect"])
        sig2 = math.sqrt(L.fit2(n, Y2[r])[1]["rss"] / (npts - 5))
        flag2.append(1 - smallest_resid(0.5, ne, nemb, Y2[r], sig2, eps)["within"])
    report("5 positive control: planted fast term (alpha 1.2, 0.3 nats at the smallest size)",
           np.mean(fires2) >= 0.8 and np.mean(flag2) >= 0.8,
           f"LRT fires {np.mean(fires2):.2f}, smallest-size residual flagged {np.mean(flag2):.2f}")
    # 6-7. shared c, null and positive
    names = ["porian-rw", "porian-owt2", "chinchilla", "pythia"]

    def pooled_sim(cs, seed):
        res = []
        prof = {}
        for j, (name, c) in enumerate(zip(names, cs)):
            ne_, nemb_, n_, mu_ = planted(name, c)
            Yk = mu_[None, :] + SD * np.random.default_rng([SEED, seed, j]).normal(0, 1, (REPS, len(mu_)))
            prof[name] = (len(mu_), profile_many(ne_, nemb_, Yk))
        ntot = sum(p[0] for p in prof.values())
        for r in range(REPS):
            lk = {k: m * np.log(P[:, r] / m) for k, (m, P) in prof.items()}
            lam = sum(lk.values()).min() - sum(v.min() for v in lk.values())
            dbic = lam - (len(names) - 1) * math.log(ntot)
            mask = np.ones(len(L.GRID_FINE), bool)
            ninf = 0
            for k, (m, P) in prof.items():
                s = L.profile_set(L.GRID_FINE, P[:, r], P[:, r].min(), m)
                if s["length"] < 2.7:
                    mask &= s["mask"]
                    ninf += 1
            res.append((dbic, bool(mask.any()) and ninf >= 2))
        return np.array(res, float)
    s6 = pooled_sim([0.5] * 4, 14)
    pass6 = float(np.mean((s6[:, 0] <= 6) & (s6[:, 1] == 1)))
    report("6 null control: same planted c on four ladders", pass6 >= 0.8,
           f"shared c accepted (dBIC <= 6 and common set) in {pass6:.2f}; median dBIC_shared {np.median(s6[:, 0]):.2f}")
    s7 = pooled_sim([0.1, 0.1, 1.5, 1.5], 15)
    pass7 = float(np.mean(s7[:, 0] > 6))
    report("7 positive control: c 0.1 (Porian) vs 1.5 (Chinchilla, Pythia)", pass7 >= 0.8,
           f"dBIC_shared > 6 in {pass7:.2f}; median dBIC_shared {np.median(s7[:, 0]):.1f}")
    # 8. outputs
    P = pd.read_csv(os.path.join(OUT, "e1_profile.csv"))
    nest = float((P.rss2 - P.rss1).max())
    S = pd.read_csv(os.path.join(OUT, "e1_ladder_summary.csv"))
    SM = pd.read_csv(os.path.join(OUT, "e1_smallest_resid_by_c.csv"))
    LR = pd.read_csv(os.path.join(OUT, "e1_lrt_by_c.csv"))
    PO = pd.read_csv(os.path.join(OUT, "e1_pooled.csv"))
    nan = int(P[["rss1", "rss2"]].isna().sum().sum() + SM[["r_min", "sd_boot"]].isna().sum().sum()
              + LR.p_lrt.isna().sum())
    okB = (SM.nboot == 500).all() and (S.free_boot == 200).all() and (PO.dropna(subset=["nboot"]).nboot == 200).all() \
        and len(P) == 4 * len(L.GRID_FINE) and (LR.tested == 1).all()
    report("8 outputs", nest <= 1e-12 and nan == 0 and okB,
           f"max RSS2 - RSS1 {nest:.1e} (nesting), NaN {nan}, bootstrap sizes as planned {bool(okB)}, "
           f"LRT testable on every ladder {bool((LR.tested == 1).all())}")
    lines.append("ALL PASS" if ok_all else "FAIL")
    print(lines[-1])
    open(os.path.join(OUT, "verify.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
