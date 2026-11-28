#!/usr/bin/env python3
"""Print the numbered items that results.md cites, from the saved files in results/ (standard library only).

    python3 -I report_numbers.py > results/report_numbers.txt

Reads results/design_<ds>.csv, results/controls_<ds>.csv, results/controls_summary.json, results/real_results.json and
results/real_not_run.json. No number is typed by hand; the gate rules and the preregistered thresholds are the ones in
summarize_controls.py and README.md.
"""
import csv
import json
import math
import statistics
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
DS = ("rw", "owt2")
REPS = 20
RULE = {"null": ("<=", 4), "fast03": (">=", 16), "fast01": (">=", 16)}
NAME = {"null": "G1 null (joint law only)", "fast03": "G2 planted +0.3 fast term", "fast01": "G3 planted +0.1 fast term",
        "corr_noise": "S1 correlated noise (0.5 within a width)", "dterm_misfit": "S2 misspecified data term"}
LINES, N_ITEM = [], [0]


def section(t):
    LINES.append(f"\n## {t}")


def item(t):
    N_ITEM[0] += 1
    LINES.append(f"{N_ITEM[0]}. {t}")


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def ctrl(ds):
    out = {}
    for r in rows(R / f"controls_{ds}.csv"):
        out.setdefault(r["control"], []).append(r)
    return out


def g(x, nd=4):
    return f"{float(x):.{nd}f}"


summary = json.loads((R / "controls_summary.json").read_text())
C = {ds: ctrl(ds) for ds in DS}
det = {ds: {c: sum(r["detect"].lower() in ("true", "1") for r in rs) for c, rs in C[ds].items()} for ds in DS}

section("Design: the real ladders the controls are built on")
for ds in DS:
    d = rows(R / f"design_{ds}.csv")
    N = [float(r["N_tied"]) for r in d]
    sd = [float(r["sd"]) for r in d]
    item(f"{ds}: {len(d)} points, tied parameter counts {min(N):.4g} to {max(N):.4g}, tokens per parameter slices {sorted({int(float(r['k'])) for r in d})}, "
         f"per-point sd {min(sd):.4f} to {max(sd):.4f}, sd at the smallest model {sd[N.index(min(N))]:.4f}.")

section("Gates and stress controls per dataset (20 replicates each, 200 bootstrap draws each, detect = p at most 0.05)")
for ds in DS:
    for c in ("null", "fast03", "fast01", "corr_noise", "dterm_misfit"):
        rs = C[ds].get(c, [])
        if not rs:
            continue
        k = det[ds][c]
        lo, hi = wilson(k, len(rs))
        T = [float(r["T"]) for r in rs]
        P = [float(r["p"]) for r in rs]
        F = [float(r["fast_at_5M"]) for r in rs]
        rule = (f"; gate rule: detections {RULE[c][0]} {RULE[c][1]} of {REPS}, "
                + ("PASSED" if summary[ds]["gates"][c] else "FAILED")) if c in RULE else "; report only, no gate"
        item(f"{ds}, {NAME[c]}: {k} of {len(rs)} detections (rate {k / len(rs):.2f}, 95% Wilson interval {lo:.2f} to {hi:.2f}); median p {statistics.median(P):.4f}, "
             f"median T {statistics.median(T):.3f}, median fitted fast term at the smallest model {statistics.median(F):.3f} nats{rule}.")

section("Gate verdict per dataset")
for ds in DS:
    gs = summary[ds]["gates"]
    item(f"{ds}: G1 {'passed' if gs['null'] else 'failed'}, G2 {'passed' if gs['fast03'] else 'failed'}, G3 {'passed' if gs['fast01'] else 'failed'}; "
         f"all three gates passed: {summary[ds]['all_gates_passed']}; detections {det[ds]['null']}, {det[ds]['fast03']} and {det[ds]['fast01']} of {REPS}.")

section("Real data (only for a dataset whose three gates passed)")
real = json.loads((R / "real_results.json").read_text()) if (R / "real_results.json").exists() else {}
notrun = json.loads((R / "real_not_run.json").read_text()) if (R / "real_not_run.json").exists() else {}
for ds in DS:
    if ds in real:
        r = real[ds]
        item(f"{ds}: statistic T {r['T']:.3f}, bootstrap p-value {r['p']:.6f} with {r['B']} draws, verdict {r['verdict']}; noise scale sigma_hat {r['sigma_hat']:.3f} times the per-point sd; "
             f"fast term at the smallest model {r['fast_at_5M']:.3f} nats.")
        item(f"{ds}: joint law fit M1 exponents alpha {r['m1_exps'][0]:.4f}, beta {r['m1_exps'][1]:.4f}, coefficients E {r['m1_coef'][0]:.4f}, A {r['m1_coef'][1]:.4f}, B {r['m1_coef'][2]:.4f}; "
             f"with the fast term M2 exponents alpha {r['m2_exps'][0]:.4f}, beta {r['m2_exps'][1]:.4f}, gamma {r['m2_exps'][2]:.4f}, "
             f"coefficients E {r['m2_coef'][0]:.4f}, A {r['m2_coef'][1]:.4f}, B {r['m2_coef'][2]:.4f}, C {r['m2_coef'][3]:.4f}.")
        item(f"{ds}: sensitivity bootstrap p-value with within-width correlated noise (correlation 0.5, 500 draws): {r['p_correlated_noise']:.6f}.")
    elif ds in notrun:
        item(f"{ds}: real-data test not run, because the gates did not all pass (failed: {', '.join(k for k, v in notrun[ds]['gates'].items() if not v)}).")

section("Preregistered predictions")
p1 = all(summary[ds]["gates"]["null"] for ds in DS)
p2 = all(summary[ds]["gates"]["fast03"] for ds in DS)
p3 = all(summary[ds]["gates"]["fast01"] for ds in DS)
item(f"Prediction 1 (G1 passes on both datasets): {'PASSED' if p1 else 'FAILED'}; null detections {det['rw']['null']} and {det['owt2']['null']} of {REPS}, limit 4.")
item(f"Prediction 2 (G2 passes on both datasets): {'PASSED' if p2 else 'FAILED'}; detections {det['rw']['fast03']} and {det['owt2']['fast03']} of {REPS}, minimum 16.")
item(f"Prediction 3 (G3 passes on both datasets): {'PASSED' if p3 else 'FAILED'}; detections {det['rw']['fast01']} and {det['owt2']['fast01']} of {REPS}, minimum 16.")
if all("dterm_misfit" in C[ds] and "corr_noise" in C[ds] for ds in DS):
    p4 = max(det[ds]["dterm_misfit"] for ds in DS) >= 5
    p5 = any(det[ds]["corr_noise"] > det[ds]["null"] for ds in DS)
    item(f"Prediction 4 (S2 gives at least 5 of {REPS} false detections on at least one dataset): {'PASSED' if p4 else 'FAILED'}; S2 detections {det['rw']['dterm_misfit']} and {det['owt2']['dterm_misfit']} of {REPS}.")
    item(f"Prediction 5 (S1 gives more false detections than G1 on at least one dataset): {'PASSED' if p5 else 'FAILED'}; S1 detections {det['rw']['corr_noise']} and {det['owt2']['corr_noise']} against G1 {det['rw']['null']} and {det['owt2']['null']} of {REPS}.")

section("Descriptive fast-term size and its limits (post hoc, not preregistered)")
SLOW_AT_MIN = 3.0 * (5173248.0 / 1e6) ** -0.6  # the truth of every control: A = 3.0, alpha = 0.6, at the smallest tied size
for ds in DS:
    for c in ("null", "fast03", "fast01"):
        rs = C[ds].get(c, [])
        if not rs:
            continue
        big = [r for r in rs if float(r["fast_at_5M"]) > 1.0]
        weak = [r for r in big if float(r["T"]) < 1.0 and float(r["p"]) > 0.05]
        item(f"{ds}, {NAME[c]}: {len(big)} of {len(rs)} replicates report a fitted fast term above 1.0 nat at the smallest model; the true slow term A n^-alpha is {SLOW_AT_MIN:.3f} nats there, "
             f"so these report the slow term relabelled as fast; {len(weak)} of those {len(big)} have T below 1.0 and p above 0.05"
             + (f"; their median fitted gamma is {statistics.median(float(r['m2_gamma']) for r in big):.3f} against the true alpha 0.6." if big else "."))
miss = sorted(float(r["p"]) for r in C["rw"].get("fast01", []) if r["detect"].lower() not in ("true", "1"))
if miss:
    item(f"rw, G3 planted +0.1 fast term: the {len(miss)} replicates that were not detected have p-values " + ", ".join(f"{p:.4f}" for p in miss) + "; the gate needed 16 detections.")
for ds in DS:
    s = [float(r["sigma_hat"]) for r in C[ds]["null"]]
    item(f"{ds}, G1 null: sigma_hat (noise scale as a multiple of the per-point sd) has median {statistics.median(s):.3f} and range {min(s):.3f} to {max(s):.3f} over {len(s)} replicates.")
if "owt2" in real:
    r = real["owt2"]
    sizes = sorted({float(x["N_tied"]) for x in rows(R / "design_owt2.csv")})
    gam, coef = r["m2_exps"][2], r["m2_coef"][3]
    vals = [(n, coef * (n / 1e6) ** -gam) for n in sizes]
    first = ", ".join(f"{n / 1e6:.2f}M: {v:.3f}" for n, v in vals[:5])
    below = next(n for n, v in vals if v < 0.01)
    item(f"owt2 real ladder: fitted fast term C n^-gamma (gamma {gam:.3f}) in nats at the five smallest model sizes: {first}; it is below 0.01 nats from {below / 1e6:.1f} million parameters upward; "
         f"the real sigma_hat is {r['sigma_hat']:.3f}, {r['sigma_hat'] / statistics.median([float(x['sigma_hat']) for x in C['owt2']['null']]):.1f} times the median null-control value.")

section("Cost of the control run")
for ds in DS:
    sec = [float(r["seconds"]) for rs in C[ds].values() for r in rs]
    item(f"{ds}: {len(sec)} control replicates took {sum(sec) / 3600:.2f} CPU hours in total, a median of {statistics.median(sec):.1f} seconds each (range {min(sec):.1f} to {max(sec):.1f}).")

print("\n".join(LINES).lstrip("\n"))
