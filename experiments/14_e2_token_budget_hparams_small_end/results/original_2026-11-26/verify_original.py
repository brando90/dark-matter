"""E2 verify (PREREG.md, 11-25-2026): reproduction, positive and null controls, arithmetic. Prints PASS/FAIL per check.

    python verify.py      # writes results/verify.txt
"""
import os

import numpy as np
import pandas as pd

import e2
import e2lib as L

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
# control truth: additive Chinchilla law, no second N term; its data term alone gives ~+0.20 nats of N-only excess at 5M
TRUTH = dict(E=2.0, A=3.0, alpha=0.6, B=1.0, beta=0.2)
FAST = dict(size=0.3, alpha=1.4)  # planted fast N term: +0.3 nats at 5M (w96), exponent 1.4
NOISE, REPS, BN, BJ = 0.003, 20, 300, 100


def truth(N, D, fast):
    y = TRUTH["E"] + TRUTH["A"] * (N / 1e6) ** -TRUTH["alpha"] + TRUTH["B"] * (D / 1e9) ** -TRUTH["beta"]
    if fast:
        n5 = 5173248 / 1e6
        y = y + FAST["size"] * (N / 1e6 / n5) ** -FAST["alpha"]
    return y


def control(fast, seed):
    sd = {w: s for (d_, w), s in L.seed_sd().items() if d_ == "rw"}
    rng = np.random.default_rng([e2.SEED, 99, int(fast), seed])
    lad = L.ladder("rw", "TS", 20).copy()
    lad["y"] = truth(lad.N_tied.values, 20 * lad.N_tied.values, fast) + rng.normal(0, NOISE, len(lad))
    J = e2.joint_points_tc("rw").copy()
    J["y"] = truth(J.N_tied.values, J.D.values, fast) + rng.normal(0, NOISE, len(J))
    eps = {w: rng.normal(0, 1, BN) for w in L.SMALL}
    X = e2.excess_boot(lad, "tied", sd, rng, BN, eps)
    small = lad[lad.width.isin(list(L.SMALL))]
    XJ = e2.joint_boot(J, small, "tied", sd, rng, BJ, eps)
    w = 96.0
    C = X["obs"][w] - XJ["obs"][w]
    lo, hi = e2.ci(X["boot"][w][:BJ] - XJ["boot"][w])
    vJ = e2.verdict(C, lo, hi, X["obs"][w], 2 * sd[w])
    mlo, mhi = e2.ci(XJ["boot"][w])
    vM = e2.ret_verdict(XJ["obs"][w], mlo, mhi, 2 * sd[w])
    return dict(fast=int(fast), rep=seed, X=X["obs"][w], XJ=XJ["obs"][w], J=C, J_lo=lo, J_hi=hi, vJ=vJ, vM=vM)


def main():
    out, ok_all = [], True

    def report(name, ok, msg):
        nonlocal ok_all
        ok_all &= bool(ok)
        out.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
        print(out[-1], flush=True)

    # 1. reproduction of DM-2 (reverse-ladder 5M residual) and curves.csv values
    cur = pd.read_csv(os.path.join(L.DM2, "results", "curves.csv"))
    X = pd.read_csv(os.path.join(RES, "e2_excess_tied.csv"))
    msgs, ok = [], True
    for ds, ref in (("rw", 0.347), ("owt2", 0.371)):
        x = X[(X.dataset == ds) & (X.cond == "TC20") & (X["size"] == "5M")].X.iloc[0]
        ok &= abs(x - ref) <= 0.005
        msgs.append(f"{ds} X_5M(TC20) {x:+.4f} vs DM-2 {ref:+.3f}")
        for c, curve in (("TC20", "tuned-short-const"), ("BC20", "base-short-chinchilla")):
            lad = L.ladder(ds, *e2.COND[c])
            ref_y = cur[(cur.curve == f"porian/{ds}/{curve}") & (cur.slice == "D=20N")].dropna(subset=["y"])
            m = lad.merge(ref_y[["width", "y"]], on="width", suffixes=("", "_dm2"))
            dmax = float(np.abs(m.y - m.y_dm2).max())
            ok &= dmax <= 1e-9 and len(m) == len(lad) == len(ref_y)
            msgs.append(f"{ds} {c} vs curves.csv max |dy| {dmax:.1e} ({len(m)} sizes)")
    report("1 reproduction", ok, "; ".join(msgs))

    # 2 / 3. positive (data term only) and null (planted fast N term) controls
    rows = [control(f, r) for f in (False, True) for r in range(REPS)]
    C = pd.DataFrame(rows)
    C.to_csv(os.path.join(RES, "verify_controls.csv"), index=False)
    p = C[C.fast == 0]
    a, b = (p.vJ == "EXPLAINS").mean(), (p.vM == "RETURNS").mean()
    report("2 positive control (data term only)", a >= 0.8 and b >= 0.8,
           f"J EXPLAINS at 5M in {a:.2f}, M_5M RETURNS in {b:.2f} of {REPS} (median X {p.X.median():+.3f}, "
           f"X^J {p.XJ.median():+.3f})")
    n = C[C.fast == 1]
    a, b = (n.vJ != "EXPLAINS").mean(), (n.vM == "STAYS OFF").mean()
    report("3 null control (planted fast N term)", a >= 0.8 and b >= 0.8,
           f"J not EXPLAINS in {a:.2f}, M_5M STAYS OFF in {b:.2f} of {REPS} (median X {n.X.median():+.3f}, "
           f"X^J {n.XJ.median():+.3f})")

    # 4. arithmetic
    F = pd.read_csv(os.path.join(RES, "e2_factors_tied.csv"))
    errs = []
    for ds in ("rw", "owt2"):
        for s in ("5M", "7M", "9M", "15M"):
            g = lambda c: X[(X.dataset == ds) & (X.cond == c) & (X["size"] == s)].X  # noqa: E731
            f = lambda nm: F[(F.dataset == ds) & (F["size"] == s) & (F.factor.str.startswith(nm))].C  # noqa: E731
            R = f("R (")
            if len(R):
                errs.append(abs(R.iloc[0] - (g("BC20").iloc[0] - g("TC20").iloc[0])))
            T = f("T (TC")
            if len(T):
                errs.append(abs(T.iloc[0] - (g("TCfin").iloc[0] - g("TC20").iloc[0])))
            H, S = f("H ("), f("S (")
            if len(H):
                errs.append(abs(S.iloc[0] - (R.iloc[0] - H.iloc[0])))
    report("4 arithmetic", max(errs) < 1e-12, f"max |error| {max(errs):.1e} over {len(errs)} identities")
    out.append("ALL PASS" if ok_all else "SOME FAIL")
    print(out[-1])
    open(os.path.join(RES, "verify.txt"), "w").write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
