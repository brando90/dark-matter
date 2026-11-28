"""Gate decision for folder 19 from results/controls_<ds>.csv: writes results/controls_summary.json and .csv.
G1 null: detections <= 4 of 20. G2 fast03: detections >= 16 of 20. G3 fast01: detections >= 16 of 20 (per dataset).
A dataset passes iff all three gates pass; stress controls are reported only. Usage: python3 -I summarize_controls.py"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPS = 20
RULES = {"null": ("<=", 4), "fast03": (">=", 16), "fast01": (">=", 16)}


def main():
    rows, summary = [], {}
    for ds in ("rw", "owt2"):
        path = os.path.join(HERE, "results", f"controls_{ds}.csv")
        df = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["control", "rep", "detect", "p", "T", "fast_at_5M"])
        gates = {}
        for c in ("null", "fast03", "fast01", "corr_noise", "dterm_misfit"):
            s = df[df.control == c]
            n, k = len(s), int(s.detect.sum())
            row = dict(dataset=ds, control=c, replicates=n, detections=k, median_p=float(s.p.median()) if n else None,
                       median_T=float(s.T.median()) if n else None, median_fast_at_5M=float(s.fast_at_5M.median()) if n else None)
            if c in RULES:
                op, thr = RULES[c]
                ok = n == REPS and ((k <= thr) if op == "<=" else (k >= thr))
                row.update(rule=f"detections {op} {thr} of {REPS}", gate_passed=bool(ok))
                gates[c] = bool(ok)
            rows.append(row)
        summary[ds] = dict(gates=gates, all_gates_passed=len(gates) == 3 and all(gates.values()))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "controls_summary.csv"), index=False)
    json.dump(summary, open(os.path.join(HERE, "results", "controls_summary.json"), "w"), indent=1)
    print(pd.DataFrame(rows).to_string())
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
