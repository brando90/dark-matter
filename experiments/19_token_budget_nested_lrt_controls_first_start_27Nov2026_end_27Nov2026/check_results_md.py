#!/usr/bin/env python3
"""Provenance check for results.md: every number in a table row must appear in the
numbered items of results/report_numbers.txt that the row cites in its last column.

Standard library only. Run from anywhere:
    python3 -I experiments/19_token_budget_nested_lrt_controls_first_start_27Nov2026_end_27Nov2026/check_results_md.py
Exit code 0 means every number token in every table row was found in its cited items.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Numbered items of results/report_numbers.txt: "12. Data: ..." -> items[12] = "Data: ..."
items = {}
for line in (HERE / "results" / "report_numbers.txt").read_text(encoding="utf-8").splitlines():
    m = re.match(r"^(\d+)\. (.*)$", line)
    if m:
        items[int(m.group(1))] = m.group(2)

# A number token: optional sign, digits with optional thousands commas, decimals, exponent, percent.
TOKEN = re.compile(r"(?<![\w.])[-+]?\d[\d,]*(?:\.\d+)?(?:e[-+]?\d+)?%?")


def numbers(text):
    out = []
    for tok in TOKEN.findall(text):
        out.append(tok.lstrip("+").rstrip("%").rstrip(","))  # only "+" is dropped: a minus sign must match the cited item verbatim
    return out


def present(token, haystack):
    return re.search(r"(?<![\w.])" + re.escape(token) + r"(?!\d)", haystack) is not None


rows = tokens_checked = failures = 0
for line in (HERE / "results.md").read_text(encoding="utf-8").splitlines():
    if not line.startswith("|"):
        continue
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    cited = cells[-1]
    if not re.fullmatch(r"\d+(, \d+)*", cited):
        continue  # header and separator rows
    ids = [int(x) for x in cited.split(", ")]
    missing_items = [i for i in ids if i not in items]
    if missing_items:
        print(f"FAIL row cites missing item(s) {missing_items}: {cells[0][:70]}")
        failures += 1
        continue
    haystack = " ".join(items[i] for i in ids)
    rows += 1
    for cell in cells[:-1]:
        for tok in numbers(cell):
            tokens_checked += 1
            if not present(tok, haystack):
                failures += 1
                print(f"FAIL number {tok!r} not in item(s) {ids}: {cells[0][:70]}")

print(f"rows checked {rows}, number tokens checked {tokens_checked}, failures {failures}")
sys.exit(1 if failures else 0)
