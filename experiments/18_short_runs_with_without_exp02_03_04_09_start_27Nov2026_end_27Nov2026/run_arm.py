#!/usr/bin/env python3
"""Run one original experiment script, unchanged, on one arm of the Chinchilla table.

    python run_arm.py --exp 02 --arm all245
    python run_arm.py --exp 09 --arm drop_DN_lt_1

Arms: all245 keeps every run (245); drop_DN_lt_1 keeps runs with D/N >= 1 (the 11 runs with fewer than one token per
parameter are removed, 234 remain). The wrapper only (a) replaces the experiment module's `load_chinchilla_runs`, (b)
points its `OUT` at outputs/<exp>/<arm>/ and (c) deletes the PNG files the script writes, then calls its main().
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

EXPS = {
    "02": ("02_compute_and_joint_extrapolation", "run_joint_backtest.py"),
    "03": ("03_residual_structure", "run_residuals.py"),
    "04": ("04_fit_stability", "run_stability.py"),
    "09": ("09_alternative_forms", "run_forms.py"),
}
ARMS = ("all245", "drop_DN_lt_1")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--exp", required=True, choices=sorted(EXPS))
    ap.add_argument("--arm", required=True, choices=ARMS)
    a = ap.parse_args()
    folder, script = EXPS[a.exp]
    path = REPO / "experiments" / folder / script
    spec = importlib.util.spec_from_file_location(f"exp{a.exp}_{a.arm}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    original = mod.load_chinchilla_runs

    def load_arm():
        df = original()
        if a.arm == "drop_DN_lt_1":
            df = df[df["ratio"] >= 1.0].reset_index(drop=True)
        print(f"[arm {a.arm}] Chinchilla runs: {len(df)}", flush=True)
        return df

    mod.load_chinchilla_runs = load_arm
    out = HERE / "outputs" / a.exp / a.arm
    out.mkdir(parents=True, exist_ok=True)
    mod.OUT = out
    t0 = time.time()
    mod.main()
    for png in out.glob("*.png"):
        png.unlink()
    print(f"[exp {a.exp} arm {a.arm}] done in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
