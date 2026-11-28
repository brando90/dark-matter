"""Checks behind the stage-3 Mega QA finding F1 (a possible infeasible Nelder-Mead start in py_src/nested_joint_lrt.py).

Part 1 (probe, old code): runs the grid ranking of the code as it was when the results were produced (commit c49cd63, read with
`git show`) on every response vector that the folder 19 pipeline fits: each control's outer response and its 200 bootstrap draws
(200 control replicates), and the real owt2 response with its 1000 independent-noise and 500 correlated-noise bootstrap draws.
Draws are regenerated with the pipeline's seeds, so they are the same vectors; only the polishing step is skipped. It counts start
cells that violate gamma >= alpha + GAP, start cells that carry the negative-coefficient penalty, short pick lists, and the
smallest number of feasible cells in any ranking. It also checks the exponent constraints on every saved outer fit.
Part 2 (regression, current code): recomputes replicates 0 and 1 of every dataset and control (20 of the 200) and the real owt2
stage with the current, hardened code and compares every saved number exactly (==, not allclose).
Usage: python3 -I verify_f1_saved_workload.py   (about 3 minutes on two cores; needs the git history for part 1).
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import controls as C  # noqa: E402
import real_data as RD  # noqa: E402

OLD_COMMIT = "c49cd63"
OLD = None
FIELDS = ["T", "p", "detect", "B", "sigma_hat", "m1_alpha", "m1_beta", "m2_alpha", "m2_beta", "m2_gamma", "m2_C", "fast_at_5M"]


def load_old():
    src = subprocess.run(["git", "-C", REPO, "show", f"{OLD_COMMIT}:py_src/nested_joint_lrt.py"],
                         capture_output=True, text=True, check=True).stdout
    path = os.path.join(tempfile.mkdtemp(), "nested_old.py")
    with open(path, "w") as fh:
        fh.write(src)
    spec = importlib.util.spec_from_file_location("nested_old", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nested_old"] = mod   # the dataclass decorator looks its module up here
    spec.loader.exec_module(mod)
    return mod


def empty():
    return dict(vectors=0, picks=0, gap_infeasible_picks=0, penalized_picks=0, short_pick_lists=0, min_feasible_cells=10 ** 9)


def probe_y(design, y, st):
    for two in (False, True):
        rss = design._grid_rss(y, two)
        k = 3 if two else 2
        cells = design._distinct_cells(rss, k)
        st["vectors"] += 1
        st["picks"] += len(cells)
        st["short_pick_lists"] += int(len(cells) < k)
        st["min_feasible_cells"] = min(st["min_feasible_cells"], int((rss < OLD.INFEASIBLE).sum()))
        for c in cells:
            st["penalized_picks"] += int(rss[c] >= OLD.INFEASIBLE)
            if two:
                st["gap_infeasible_picks"] += int(OLD.GAMMA_GRID[c[2]] < OLD.ALPHA_GRID[c[0]] + OLD.GAP)


def draws(design, y, B, rng, corr_width=None):
    """The pipeline's bootstrap responses (bootstrap_lrt / corr_bootstrap_p), without the polishing of each draw."""
    _, f1, _ = design.statistic(y)
    scale = float(np.sqrt(f1.rss / (design.npts - 5))) / design._sw
    if corr_width is not None:
        ids = {w: i for i, w in enumerate(np.unique(corr_width))}
        idx = np.array([ids[w] for w in corr_width])
    for _ in range(B):
        if corr_width is None:
            yield f1.fitted + scale * rng.standard_normal(design.npts)
        else:
            u = rng.standard_normal(len(ids))[idx]
            yield f1.fitted + scale * (np.sqrt(C.RHO) * u + np.sqrt(1 - C.RHO) * rng.standard_normal(design.npts))


def probe_control(task):
    ds, control, rep = task
    df = pd.read_csv(os.path.join(HERE, "results", f"design_{ds}.csv"))
    ds_i, c_i = C.DATASETS.index(ds), C.ALL.index(control)
    y = C.synthetic(df, control, np.random.default_rng([C.SEED, ds_i, c_i, rep]))
    design = OLD.Design(df.N_tied.values, df.D.values, df.sd.values)
    st = empty()
    probe_y(design, y, st)
    for ys in draws(design, y, C.B, np.random.default_rng([C.SEED, ds_i, c_i, rep, 1])):
        probe_y(design, ys, st)
    return st


def probe_real(kind):
    df = pd.read_csv(os.path.join(HERE, "results", "design_owt2.csv"))
    design = OLD.Design(df.N_tied.values, df.D.values, df.sd.values)
    st = empty()
    probe_y(design, df.y.values, st)
    gen = (draws(design, df.y.values, 1000, np.random.default_rng([C.SEED, 101])) if kind == "indep" else
           draws(design, df.y.values, 500, np.random.default_rng([C.SEED, 201]), corr_width=df.width.values))
    for ys in gen:
        probe_y(design, ys, st)
    return st


def exact_equal(a, b):
    return float(a) == float(b)


def main():
    global OLD
    t0 = time.time()
    OLD = load_old()
    print(f"Part 1: probe of the old code ({OLD_COMMIT}) on the saved workload")
    tasks = [(ds, c, r) for c in C.ALL for ds in C.DATASETS for r in range(C.REPS)]
    with Pool(2) as pool:
        sts = pool.map(probe_control, tasks, chunksize=4) + pool.map(probe_real, ["indep", "corr"])
    tot = empty()
    for st in sts:
        for k in tot:
            tot[k] = min(tot[k], st[k]) if k == "min_feasible_cells" else tot[k] + st[k]
    print(f"  response vectors ranked (each for M1 and M2): {tot['vectors'] // 2}; start cells picked: {tot['picks']}")
    print(f"  start cells violating gamma >= alpha + GAP: {tot['gap_infeasible_picks']}")
    print(f"  start cells carrying the negative-coefficient penalty: {tot['penalized_picks']}; short pick lists: {tot['short_pick_lists']}")
    print(f"  fewest feasible cells in any single ranking: {tot['min_feasible_cells']}")
    bad = rows = 0
    for ds in C.DATASETS:
        d = pd.read_csv(os.path.join(HERE, "results", f"controls_{ds}.csv"), keep_default_na=False, float_precision="round_trip")
        for r in d.itertuples():
            rows += 1
            if float(r.m2_C) > 0 and not float(r.m2_gamma) >= float(r.m2_alpha) + OLD.GAP - 1e-12:
                bad += 1
            if not (0.02 <= r.m1_alpha <= 2.0 and 0.02 <= r.m1_beta <= 1.5 and 0.02 <= r.m2_alpha <= 2.0
                    and 0.02 <= r.m2_beta <= 1.5 and 0.2 <= r.m2_gamma <= 6.0):
                bad += 1
    real = json.load(open(os.path.join(HERE, "results", "real_results.json")))["owt2"]
    rows += 1
    bad += int(real["m2_exps"] is not None and not real["m2_exps"][2] >= real["m2_exps"][0] + OLD.GAP - 1e-12)
    print(f"  saved outer fits checked: {rows}; exponent-constraint violations: {bad}")

    print("Part 2: regression of the current code against the saved numbers (exact equality)")
    todo = [(ds, c, r) for c in C.ALL for ds in C.DATASETS for r in (0, 1)]
    # float_precision="round_trip" is required for an exact comparison: pandas' default CSV float parser can be one unit in the
    # last place off for 17-digit strings, which would flag identical numbers as different.
    saved = {ds: pd.read_csv(os.path.join(HERE, "results", f"controls_{ds}.csv"), keep_default_na=False, float_precision="round_trip")
             for ds in C.DATASETS}
    mismatches = 0
    with Pool(2) as pool:
        for row in pool.imap_unordered(C.one, todo):
            s = saved[row["dataset"]]
            s = s[(s["control"] == row["control"]) & (s["rep"] == row["rep"])].iloc[0]
            diff = [f for f in FIELDS if not exact_equal(row[f], s[f])]
            mismatches += bool(diff)
            if diff:
                print("  MISMATCH", row["dataset"], row["control"], row["rep"], diff)
    print(f"  control replicates recomputed: {len(todo)} of {C.REPS * len(C.ALL) * len(C.DATASETS)}; replicates with any mismatch: {mismatches}")
    new = json.loads(json.dumps(RD.fit_dataset(pd.read_csv(os.path.join(HERE, "results", "design_owt2.csv")), 1)))
    rdiff = [k for k in real if new[k] != real[k]]
    print(f"  real owt2 stage recomputed (B=1000 plus the B=500 correlated-noise run); fields differing from real_results.json: {rdiff or 'none'}")
    print(f"seconds {time.time() - t0:.0f}")
    sys.exit(1 if (tot["gap_infeasible_picks"] or bad or mismatches or rdiff) else 0)


if __name__ == "__main__":
    main()
