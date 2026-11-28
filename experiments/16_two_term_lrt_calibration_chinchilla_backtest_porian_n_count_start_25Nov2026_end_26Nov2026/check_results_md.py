#!/usr/bin/env python3
"""Provenance check for results.md: every number in a table row must appear in the
numbered items of results/report_numbers.txt that the row cites in its last column.

Standard library only. Run from anywhere:
    python3 -I experiments/16_two_term_lrt_calibration_chinchilla_backtest_porian_n_count_start_25Nov2026_end_26Nov2026/check_results_md.py
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
    """True if the token occurs as a whole number: not as the leading part of a longer one ("1" is not found in "1.897",
    "1e-06" or "1,200")."""
    return re.search(r"(?<![\w.])" + re.escape(token) + r"(?!\d|\.\d|e[-+]?\d|,\d)", haystack) is not None


SEPARATOR = re.compile(r"^\|[\s:|-]+\|?$")
rows = tokens_checked = failures = 0
md_lines = (HERE / "results.md").read_text(encoding="utf-8").splitlines()
for k, line in enumerate(md_lines):
    if not line.startswith("|") or SEPARATOR.match(line):
        continue
    if k + 1 < len(md_lines) and SEPARATOR.match(md_lines[k + 1]):
        continue  # header row: the row directly above a separator
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    cited = cells[-1]
    if not re.fullmatch(r"\d+(, \d+)*", cited):
        print(f"FAIL last cell is not a list of item numbers ({cited[:30]!r}): {cells[0][:70]}")
        failures += 1
        continue
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
