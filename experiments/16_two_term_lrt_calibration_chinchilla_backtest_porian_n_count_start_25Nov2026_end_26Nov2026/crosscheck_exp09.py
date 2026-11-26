"""Cross-check: how much of experiment 09's joint size-holdout test error comes from runs with D/N below 0.35?

Input: experiments/09_alternative_forms/outputs/predictions.csv from brando90/dark-matter main, passed on the command
line (read a copy with `git show origin/main:<path>`; this script never touches the repo). Columns used: dataset,
family, kind, split, N, D, loss, prediction.

For each joint cloud and joint model it recomputes the test RMSE in loss units (the quantity experiment 09 reports),
the number of runs with D/N < 0.35 in the test and train splits, the share of the test squared error those runs
carry, and the test RMSE without them. Output: results/crosscheck_exp09.json
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

THRESHOLD = 0.35  # same post hoc cut-off as E2b, chosen from the residuals of the Chinchilla points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--source", default="", help="provenance note: repo commit and blob of the file")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.predictions)))
    out = {"source": args.source, "threshold_D_over_N": THRESHOLD, "cells": {}}
    for ds in sorted({r["dataset"] for r in rows if r["family"] == "joint"}):
        for kind in ("joint", "joint_extra_n", "joint_extra_d"):
            sel = [r for r in rows if r["dataset"] == ds and r["family"] == "joint" and r["kind"] == kind]
            test = [r for r in sel if r["split"] == "test"]
            train = [r for r in sel if r["split"] == "train"]
            if not test:
                continue
            err = [(float(r["loss"]) - float(r["prediction"]), float(r["D"]) / float(r["N"])) for r in test]
            tot = sum(e * e for e, _ in err)
            short = [e for e, dn in err if dn < THRESHOLD]
            rest = [e for e, dn in err if dn >= THRESHOLD]
            out["cells"][f"{ds}|{kind}"] = {
                "n_test": len(test),
                "n_train": len(train),
                "rmse_all": math.sqrt(tot / len(err)),
                "n_short_in_test": len(short),
                "n_short_in_train": sum(1 for r in train if float(r["D"]) / float(r["N"]) < THRESHOLD),
                "share_sq_err_short": sum(e * e for e in short) / tot,
                "rmse_without_short": math.sqrt(sum(e * e for e in rest) / len(rest)),
            }
    dst = HERE / "results" / "crosscheck_exp09.json"
    dst.write_text(json.dumps(out, indent=1))
    for k, v in out["cells"].items():
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
    print("wrote", dst)


if __name__ == "__main__":
    main()
