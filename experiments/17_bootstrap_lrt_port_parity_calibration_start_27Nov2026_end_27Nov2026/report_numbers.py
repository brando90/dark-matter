#!/usr/bin/env python3
"""Print the numbered items that results.md cites, from the saved files in results/ (standard library only).

    python3 -I report_numbers.py > results/report_numbers.txt
"""
import csv
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
par = json.loads((R / "parity_summary.json").read_text())
cal = json.loads((R / "calibration_summary.json").read_text())
cells = cal["cells"]
nul = [(k, v) for k, v in cells.items() if v["role"] == "null"]
ctl = cells["power_control_1e5_1e9_8pd_sd0.005_s1"]
weak = cells["power_weak_1e7_1e9_4pd_sd0.01_s0.1"]
with open(R / "calibration.csv") as fh:
    rows = list(csv.DictReader(fh))
total = len(rows)
ph = json.loads((R / "posthoc_multiplicative_summary.json").read_text())
tests = (R / "test_fail_loud.txt").read_text().strip().splitlines()[-1]

items = []


def item(text):
    items.append(text)


item(f"Source identity: the fitter block in py_src/two_term_lrt.py is {'identical' if par['source_identical'] else 'DIFFERENT'} "
     f"to dm1b.py's block after three renames ({par['source_lines_compared']} lines compared).")
item(f"Row parity on committed DM-1b rows: {par['rows_all_ok']} of {par['rows']} sampled rows reproduce k_exceed, refits used, "
     f"the decision, T and p; mismatches: k {par['k_mismatch']}, refits used {par['b_mismatch']}, decision {par['det_mismatch']}, "
     f"T {par['T_mismatch']}, p {par['p_mismatch']}; largest relative difference in T {par['max_T_rel_diff']:.1e}; "
     f"detections {par['detections_lib']} against {par['detections_ref']}; rows with a non-finite bootstrap statistic "
     f"{par['rows_with_nonfinite_bootstrap']}; rows from ladders of more than 17 points {par['rows_large_ladders']}.")
for k, v in nul:
    item(f"Null cell {k} ({v['npts']} points): {v['detections']} of {v['n']} ladders rejected, rate {v['rate']}, "
         f"exact 95% interval {v['exact_95ci'][0]} to {v['exact_95ci'][1]}.")
p = cal["null_pooled"]
item(f"Pooled null rejection: {p['detections']} of {p['n']} ladders, rate {p['rate']}, exact 95% interval "
     f"{p['exact_95ci'][0]} to {p['exact_95ci'][1]}; worst cell {cal['null_worst_cell_rate']}; gates {cal['gates']['pooled_max']} "
     f"pooled and {cal['gates']['cell_max']} per cell; DM-1b measured 0.049 pooled and 0.070 worst cell.")
item(f"Power, positive control (33 points, second term of relative size 1): {ctl['detections']} of {ctl['n']} ladders detected, "
     f"exact 95% interval {ctl['exact_95ci'][0]} to {ctl['exact_95ci'][1]}, median T {ctl['median_T']:.1f}; gate at least "
     f"{cal['gates']['power_min_detections']}.")
item(f"Power, weak cell (9 points, second term of relative size 0.1): {weak['detections']} of {weak['n']} ladders detected, "
     f"rate {weak['rate']}, exact 95% interval {weak['exact_95ci'][0]} to {weak['exact_95ci'][1]}; DM-1b's power for this design is 0.05.")
item(f"Criteria: 1 {'pass' if par['criteria_1_pass'] else 'FAIL'}, 2 {'pass' if par['criteria_2_pass'] else 'FAIL'}, "
     f"3 {'pass' if cal['criterion_3_pass'] else 'FAIL'}, 4 {'pass' if cal['criterion_4_pass'] else 'FAIL'}.")
item(f"Fail-loud tests (pytest on test_fail_loud.py): {tests}.")
item(f"Run time: {total} fresh ladders took on average {sum(float(r['seconds']) for r in rows) / total:.2f} seconds each; "
     f"the {par['rows']} parity rows took {par['seconds']} seconds in all.")
item(f"Environment: Python {cal['python']}, numpy {cal['numpy']}, two CPU cores; the repository pins Python 3.11.")
for k, v in ph["cells"].items():
    item(f"Post hoc multiplicative noise, cell {k} ({v['npts']} points, loss {v['loss_range'][0]} to {v['loss_range'][1]}): "
         f"{v['detections']} of {v['n']} one-term ladders rejected, rate {v['rate']}, exact 95% interval "
         f"{v['exact_95ci'][0]} to {v['exact_95ci'][1]}.")
item(f"Post hoc multiplicative noise, pooled: {ph['pooled']['detections']} of {ph['pooled']['n']} ladders rejected, rate "
     f"{ph['pooled']['rate']}, exact 95% interval {ph['pooled']['exact_95ci'][0]} to {ph['pooled']['exact_95ci'][1]}; relative noise sd "
     f"{ph['rel_sd']}; not preregistered.")
for i, t in enumerate(items, 1):
    print(f"{i}. {t}")
