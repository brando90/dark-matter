"""Regression test for py_src/scaling_ext.fit: 11 fits, each compared with a stored golden result.

Why it exists: the Mega QA pass of 11-27-2026 split the 150-line fit() into small helpers (finding F7). The golden file
results/golden_scaling_ext_fit.json was captured from fit() BEFORE that change (commit c49cd63), so a pass here shows that
the refactor changed no fitted number in these 11 calls (every fitting mode: 1- and 2-term N laws, the joint law, three
dark-term joint laws, warm starts, frozen exponents, both objectives, exp_max).

    python test_scaling_ext_golden.py            compare with the golden file (exit 1 on any difference)
    python test_scaling_ext_golden.py --capture  overwrite the golden file with the CURRENT code's results (only do this on
                                                 code you have already shown to be correct)

The comparison is exact equality of every float on the machine that wrote the golden file (CPython 3.13.18, numpy 2.5.3,
scipy 1.18.1). On another platform BLAS rounding can differ in the last digits, so the pass criterion is a relative
difference of at most 1e-9 (absolute 1e-12); the log also says how many numbers were exactly equal. The log is saved to
results/test_scaling_ext_golden.txt.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from py_src import scaling_ext as se  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
GOLDEN = RES / "golden_scaling_ext_fit.json"
RTOL, ATOL = 1e-9, 1e-12

rng = np.random.default_rng(123)
N = np.geomspace(1e7, 1e10, 14)
D = N * 20 * np.exp(0.3 * rng.standard_normal(14))
L = (1.7 + 120 / N ** 0.30 + 400 / D ** 0.28) * np.exp(0.005 * rng.standard_normal(14))
L1 = (1.7 + 120 / N ** 0.30) * np.exp(0.005 * rng.standard_normal(14))
K = dict(n_starts=60, n_refine=3, max_nfev=600)


def pack(r):
    return dict(E=float(r.E), c=[float(x) for x in r.c], e=[float(x) for x in r.e], objective=float(r.objective),
                rss=float(r.rss), n=int(r.n), ok=bool(r.ok))


def run():
    out = {}
    r1 = se.fit(se.ONE_N, N, L1, seed=0, **K)
    out["one_n_huber"] = pack(r1)
    out["two_n_huber"] = pack(se.fit(se.TWO_N, N, L1, seed=1, **K))
    out["two_n_warm"] = pack(se.fit(se.TWO_N, N, L1, seed=2, warm=r1, **K))
    out["two_n_half_ls"] = pack(se.fit(se.TWO_N_HALF, N, L1, objective="ls", seed=3, **K))
    j = se.fit(se.JOINT, N, L, D, seed=0, **K)
    out["joint_huber"] = pack(j)
    jn = se.fit(se.JOINT_N_DARK, N, L, D, seed=4, warm=j, **K)
    out["joint_ndark_warm"] = pack(jn)
    out["joint_both_dark_warm"] = pack(se.fit(se.JOINT_BOTH_DARK, N, L, D, seed=5, warm=jn, **K))
    out["joint_fixmain"] = pack(se.fit(se.JOINT, N, L, D, seed=6, fix_main={0: 0.35}, **K))
    out["one_n_ls_expmax"] = pack(se.fit(se.ONE_N, N, L1, objective="ls", seed=7, exp_max=1.5, **K))
    out["joint_ddark_warm_ls"] = pack(se.fit(se.JOINT_D_DARK, N, L, D, objective="ls", seed=8, warm=j, **K))
    out["joint_both_dark_warm_fix"] = pack(se.fit(se.JOINT_BOTH_DARK, N, L, D, seed=9, warm=jn, fix_main={1: 0.3}, **K))
    return out


def flat(d):
    vals = []
    for k in ("E", "objective", "rss", "n"):
        vals.append(float(d[k]))
    vals += [float(x) for x in d["c"]] + [float(x) for x in d["e"]] + [float(d["ok"])]
    return np.array(vals)


def main():
    t0 = time.time()
    out = run()
    lines = [f"fits run: {len(out)} in {time.time() - t0:.1f} seconds"]
    if "--capture" in sys.argv:
        GOLDEN.write_text(json.dumps(out, indent=1) + "\n")
        lines.append(f"golden file written: {GOLDEN.name}")
        print("\n".join(lines))
        return 0
    gold = json.loads(GOLDEN.read_text())
    bad, exact_numbers, numbers = [], 0, 0
    for k, g in gold.items():
        a, b = flat(out[k]), flat(g)
        if a.shape != b.shape or not np.allclose(a, b, rtol=RTOL, atol=ATOL):
            bad.append(k)
            continue
        numbers += a.size
        exact_numbers += int(np.sum(a == b))
    lines.append(f"golden fits compared: {len(gold)}; numbers compared: {numbers}; exactly equal: {exact_numbers}")
    lines.append(f"fits outside rtol {RTOL:g} / atol {ATOL:g}: {bad or 'none'}")
    lines.append("RESULT: " + ("FAIL" if bad else "PASS"))
    text = "\n".join(lines)
    print(text)
    (RES / "test_scaling_ext_golden.txt").write_text(text + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
