"""Differential test: scaling_ext.fit before and after the Mega QA refactor, on a random battery of calls.

The old fit() is the one at commit c49cd63 (150 lines, read from git). Both versions are called with the same inputs, and every
returned number must be exactly equal (same E, coefficients, exponents, objective, rss, n and ok flag), or both must raise the same
exception. The battery covers all seven standard model specs, both objectives, warm starts, frozen exponents, exp_max, e_range and
delta. It needs the git history of this repository.

    python test_scaling_ext_vs_c49cd63.py      (about 15 seconds; exit 1 on any difference)

The log is saved to results/test_scaling_ext_vs_c49cd63.txt.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from py_src import scaling_ext as new  # noqa: E402

RES = Path(__file__).resolve().parent / "results"
OLD_COMMIT = "c49cd63"
OLD_FILES = ("scaling_ext.py", "scaling_laws_fit.py", "two_term_lrt.py")


def load_old():
    """The py_src modules at OLD_COMMIT, written to a temporary package called pkg_old and imported from there."""
    root = Path(tempfile.mkdtemp())
    pkg = root / "pkg_old"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    for name in OLD_FILES:
        text = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{OLD_COMMIT}:py_src/{name}"],
                              capture_output=True, text=True, check=True).stdout
        (pkg / name).write_text(text)
    sys.path.insert(0, str(root))
    return importlib.import_module("pkg_old.scaling_ext")


SPECS = ["ONE_N", "TWO_N", "TWO_N_HALF", "JOINT", "JOINT_N_DARK", "JOINT_D_DARK", "JOINT_BOTH_DARK"]
PREFIX = {"TWO_N": "ONE_N", "TWO_N_HALF": "ONE_N", "JOINT_N_DARK": "JOINT", "JOINT_D_DARK": "JOINT", "JOINT_BOTH_DARK": "JOINT_N_DARK"}


def same(a, b):
    return (a.E == b.E and np.array_equal(a.c, b.c) and np.array_equal(a.e, b.e) and a.objective == b.objective
            and a.rss == b.rss and a.n == b.n and a.ok == b.ok)


def main():
    old = load_old()
    rng = np.random.default_rng(2026)
    lines = [f"old code: commit {OLD_COMMIT}; new code: working tree"]
    bad = ncalls = 0
    t0 = time.time()

    def data(joint, n):
        N = np.geomspace(1e7, 1e10, n)
        D = N * 20 * np.exp(0.3 * rng.standard_normal(n))
        noise = np.exp(0.004 * rng.standard_normal(n))
        return N, D, (1.7 + 150 / N ** 0.31 + (300 / D ** 0.27 if joint else 0)) * noise

    for trial in range(36):
        name = SPECS[trial % len(SPECS)]
        joint = name.startswith("JOINT")
        N, D, L = data(joint, int(rng.integers(9, 16)))
        kw = dict(n_starts=int(rng.integers(8, 50)), n_refine=int(rng.integers(1, 4)), max_nfev=int(rng.integers(100, 400)),
                  objective=str(rng.choice(["huber", "ls"])), seed=int(rng.integers(0, 1000)))
        if rng.random() < 0.4:
            kw["exp_max"] = float(rng.choice([0.8, 1.5, 3.0]))
        if rng.random() < 0.4:
            kw["e_range"] = (float(rng.choice([0.02, 0.05])), float(rng.choice([1.0, 2.0])))
        if rng.random() < 0.3:
            kw["delta"] = float(rng.choice([1e-4, 1e-2]))
        if rng.random() < 0.3:
            kw["fix_main"] = {0: float(rng.choice([0.25, 0.4]))}
        args = (N, L, D) if joint else (N, L)
        warm = None
        if name in PREFIX and "fix_main" not in kw and rng.random() < 0.8:
            p = PREFIX[name]
            pargs = (N, L, D) if p.startswith("JOINT") else (N, L)
            wk = dict(n_starts=12, n_refine=1, max_nfev=150, seed=7)
            warm = (old.fit(getattr(old, p), *pargs, **wk), new.fit(getattr(new, p), *pargs, **wk))
            assert same(*warm), "the warm-start base fits already differ"
        out = []
        for mod, w in ((old, None if warm is None else warm[0]), (new, None if warm is None else warm[1])):
            try:
                out.append((mod.fit(getattr(mod, name), *args, **kw, **({} if w is None else {"warm": w})), None))
            except Exception as ex:  # same exception type on both sides counts as equal
                out.append((None, type(ex).__name__))
        ncalls += 1
        if out[0][1] != out[1][1] or (out[0][0] is not None and not same(out[0][0], out[1][0])):
            bad += 1
            lines.append(f"DIFF trial {trial} {name} {kw} {out[0][1]} {out[1][1]}")
    for label, fn in (("joint spec without D", lambda m: m.fit(m.JOINT, [1e8, 1e9], [2.0, 1.9])),
                      ("negative N", lambda m: m.fit(m.ONE_N, [1e8, -1.0], [2.0, 1.9]))):
        res = []
        for m in (old, new):
            try:
                fn(m)
                res.append(None)
            except Exception as ex:
                res.append((type(ex).__name__, str(ex)))
        ncalls += 1
        if res[0] != res[1]:
            bad += 1
            lines.append(f"DIFF validation ({label}): {res}")
    lines.append(f"calls compared: {ncalls}; differing: {bad}; seconds {time.time() - t0:.0f}")
    lines.append("RESULT: " + ("FAIL" if bad else "PASS"))
    text = "\n".join(lines)
    print(text)
    (RES / "test_scaling_ext_vs_c49cd63.txt").write_text(text + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
