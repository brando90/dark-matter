"""Unit tests for py_src/nested_joint_lrt.py (folder 19). Run: python3 -I test_nested_lrt.py  -> prints PASS lines, exits 1 on failure.
They are software checks on a real design grid with synthetic responses; they are not the gate controls."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
from py_src.nested_joint_lrt import Design  # noqa: E402
from py_src.scaling_laws_fit import InsufficientDataError  # noqa: E402

df = pd.read_csv(os.path.join(HERE, "results", "design_owt2.csv"))
N, D, SD = df.N_tied.values, df.D.values, df.sd.values
truth = lambda N, D: 2.0 + 3.0 * (N / 1e6) ** -0.6 + 1.0 * (D / 1e9) ** -0.2  # noqa: E731
des = Design(N, D, SD)
fails = []


def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (" " + detail if detail else ""))
    if not ok:
        fails.append(name)


rng = np.random.default_rng(0)
y = truth(N, D) + 1e-5 * rng.standard_normal(len(N))
T, f1, f2 = des.statistic(y)
check("M1 recovers exponents at tiny noise", abs(f1.exps[0] - 0.6) < 0.01 and abs(f1.exps[1] - 0.2) < 0.01, str(np.round(f1.exps, 4)))
check("M1 recovers coefficients at tiny noise", np.allclose(f1.coef, [2.0, 3.0, 1.0], atol=0.02), str(np.round(f1.coef, 4)))
check("nested: RSS_2 clipped so T >= 0", T >= 0 and f2.rss >= 0)
planted = truth(N, D) + 0.3 * (N / 5173248.0) ** -1.4 + 1e-5 * rng.standard_normal(len(N))
Tp, g1, g2 = des.statistic(planted)
check("planted fast term: M2 beats M1 by a wide margin", g2.rss < 0.01 * g1.rss and Tp > 100, f"T={Tp:.1f} gamma={g2.exps[2]:.2f}")
check("planted fast term: gamma > alpha + gap", g2.exps[2] >= g2.exps[0] + 0.15 - 1e-9)
a = des.bootstrap_lrt(planted, B=5, rng=np.random.default_rng(3))
b = des.bootstrap_lrt(planted, B=5, rng=np.random.default_rng(3))
check("bootstrap reproducible for a fixed generator", np.array_equal(a["T_boot"], b["T_boot"]) and a["p"] == b["p"])
check("p-value is at least 1/(B+1)", a["p"] >= 1 / 6 - 1e-12)
for bad, msg in ((lambda: Design(N[:6], D[:6], SD[:6]), "too few points"),
                 (lambda: Design(np.full(20, 1e7), D[:20], SD[:20]), "one distinct size")):
    try:
        bad()
        check(f"raises InsufficientDataError: {msg}", False)
    except InsufficientDataError:
        check(f"raises InsufficientDataError: {msg}", True)
sys.exit(1 if fails else 0)
