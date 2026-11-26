"""E3 verify (PREREG.md): fold counts and NaN, positive control (wrong count -> e1_strict wins), null control (right
count -> no false win), no peeking at held-out sizes.

    python verify.py --workers 12      # prints each check and ALL PASS / FAIL; writes results/verify.txt
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

import e3

OUT = os.path.join(e3.HERE, "results")
REPS = 20
SD = 0.005
SEED = 20261128


def synthetic(c_true, tag):
    g = e3.load()[0]["porian-rw"]
    ne, nemb, y = g.N_nonemb.values, g.N_emb.values, g.y.values
    n = e3.L.n_of(c_true, ne, nemb)
    mu = e3.L.pred1(e3.L.fit1(n, y), n)
    jobs, jid = [], 0
    for r in range(REPS):
        ys = mu + SD * np.random.default_rng([SEED, tag, r]).normal(0, 1, len(mu))
        for k in range(e3.MIN_FIT, len(ys)):
            jobs.append((f"syn{r}", ne, nemb, ys, 2.0, 2.0, k, 100000 * tag + jid))
            jid += 1
    return jobs


def gain10(rows):
    df = pd.DataFrame(rows)
    a = df[(df.arm == "e1_strict") & (df.defined == 1) & (df.ratio >= 10)]
    b = df[(df.arm == "one") & (df.ratio >= 10)]
    m = a.merge(b, on=["ladder", "k", "target_rank"], suffixes=("", "_one"))
    return float((m.abs_err_one - m.abs_err).median()), len(m), float(m.abs_err.median()), float(m.abs_err_one.median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    lines, ok_all = [], True

    def report(name, ok, msg):
        nonlocal ok_all
        ok_all &= bool(ok)
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}: {msg}")
        print(lines[-1], flush=True)

    lads, c_sh = e3.load()
    df = pd.read_csv(os.path.join(OUT, "folds.csv"))
    nfolds = sum(len(g) - 4 for g in lads.values())
    ntg = sum(sum(len(g) - k for k in range(4, len(g))) for g in lads.values())
    got = df.groupby(["ladder", "k"]).ngroups
    nan = int(df[df.defined == 1].pred.isna().sum())
    rows_ok = len(df) == ntg * len(e3.ARMS)
    defined_ok = bool((df[(df.arm == "e1_strict")].defined == (df[df.arm == "e1_strict"].k >= 5)).all())
    report("1 fold counts / NaN", got == nfolds and rows_ok and nan == 0 and defined_ok,
           f"{got} folds (expected {nfolds}), {len(df)} rows (expected {ntg * len(e3.ARMS)}), NaN in defined "
           f"predictions {nan}, e1_strict defined exactly for k >= 5: {defined_ok}")
    with Pool(min(a.workers, 16)) as pool:
        pos = [r for o in pool.map(e3.run_fold, synthetic(0.3, 1), chunksize=1) for r in o]
        null = [r for o in pool.map(e3.run_fold, synthetic(2.0, 2), chunksize=1) for r in o]
    g, m, ea, eo = gain10(pos)
    report("2 positive control: planted c = 0.3, arm 1 uses c = 2", g > 0.005,
           f"median gain of e1_strict at >= 10x {g:.4f} nats over {m} pairs (|err| {ea:.4f} vs one {eo:.4f})")
    g, m, ea, eo = gain10(null)
    report("3 null control: planted c = 2 = c_std", g < 0.005,
           f"median gain {g:.4f} nats over {m} pairs (|err| {ea:.4f} vs one {eo:.4f})")
    # 4. no peeking: perturb held-out y; training-fold choices and predictions must not move
    g = lads["porian-rw"]
    ne, nemb, y = g.N_nonemb.values, g.N_emb.values, g.y.values.copy()
    same = True
    for k in (5, 8, 11):
        y2 = y.copy()
        y2[k:] += np.random.default_rng([SEED, k]).normal(0, 0.5, len(y) - k)
        r1 = pd.DataFrame(e3.run_fold(("x", ne, nemb, y, 2.0, c_sh, k, 7)))
        r2 = pd.DataFrame(e3.run_fold(("x", ne, nemb, y2, 2.0, c_sh, k, 7)))
        same &= bool(np.allclose(r1.pred.fillna(-1), r2.pred.fillna(-1), rtol=0, atol=1e-12))
        same &= bool(np.allclose(r1.info_c.fillna(-1), r2.info_c.fillna(-1), rtol=0, atol=0))
    report("4 no peeking (held-out y perturbed by sd 0.5)", same, f"predictions and chosen c unchanged: {same}")
    lines.append("ALL PASS" if ok_all else "FAIL")
    print(lines[-1])
    open(os.path.join(OUT, "verify.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
