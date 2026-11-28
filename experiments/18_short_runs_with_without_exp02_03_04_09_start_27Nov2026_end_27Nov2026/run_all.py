#!/usr/bin/env python3
"""Run the eight (experiment, arm) pairs, a few at a time. Each runs in its own process; logs go to results/logs/."""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAIRS = [(e, a) for e in ("02", "04", "09", "03") for a in ("all245", "drop_DN_lt_1")]


def run(pair):
    exp, arm = pair
    log = HERE / "results" / "logs" / f"exp{exp}_{arm}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as fh:
        rc = subprocess.call([sys.executable, str(HERE / "run_arm.py"), "--exp", exp, "--arm", arm], stdout=fh,
                             stderr=subprocess.STDOUT, cwd=str(HERE.parents[1]))
    print(f"exp {exp} arm {arm}: exit {rc}", flush=True)
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    with ThreadPoolExecutor(args.workers) as ex:
        rcs = list(ex.map(run, PAIRS))
    sys.exit(1 if any(rcs) else 0)
