# The bootstrap likelihood-ratio test is now lr_test, and the two-term fitter fails loudly: results (11-27-2026)

**Doc link:** <https://github.com/brando90/dark-matter/blob/main/experiments/17_bootstrap_lrt_port_parity_calibration_start_27Nov2026_end_27Nov2026/results.md>

**TLDR:** The port reproduces DM-1b exactly, keeps its calibration on fresh ladders when the noise is additive, finds a strong second term every time, and the fitter now refuses empty or too-small selections. With multiplicative noise and a wide loss range, which real ladders resemble, the same test rejects about twice as often as its nominal level, so a detection is weaker evidence than five percent suggests.

Date: 11-27-2026. As of: 11-27-2026 17:30 PST.

Status: COMPLETE. Verdict: PORT VERIFIED, ALL FIVE CRITERIA PASS; POST HOC, THE TEST OVER-REJECTS UNDER WIDE-RANGE MULTIPLICATIVE NOISE. Criteria 1 to 4 were written before they were run; criterion 5 was not prospective (see README).

Owner: Claude Code, Claude Sonnet 5.5, on CPU. Overview and criteria: [README.md](README.md).

## Bottom line

`lr_test(N, L)` in `py_src/scaling_laws_fit.py` now runs the calibrated bootstrap test that DM-1b already had, instead of a chi-square test that could never reject. The fitter block is the same text as in `dm1b.py` after three renames, and 220 of 220 sampled committed DM-1b rows come out identical in every decision field. On six hundred fresh one-term ladders with additive noise it rejects 33 times, a rate of 0.055, inside its 0.07 gate, and it finds the strong second term in 40 of 40 ladders. It cannot see the weak second term that DM-1b says the data cannot resolve, and it says so, 2 of 40. `fit_two_term_N_only` and `lr_test` now raise on an empty, too-small, non-finite or non-positive selection, and the repo smoke test, which used to fit all 400 grid points of a table whose data size also varied after selecting zero points, now fits a genuine twenty-point N-ladder. One result goes against easy use: when I repeated the null check with multiplicative noise, the rejection rate rose to 0.115 pooled and to 0.175 where the loss spans about two to ten. That check was not planned, but it is the one that matters for real ladders.

## Results

Every row is copied from results/report_numbers.txt, which report_numbers.py prints from the saved files in results/. The last column is the item number in that file. Run check_results_md.py to confirm that every number in a row appears in the items the row cites.

### Port parity (criteria 1 and 2)

| Metric | Value | Item |
|---|---|---|
| Fitter block identical to dm1b.py after three renames, lines compared | identical, 187 | 1 |
| Sampled committed DM-1b rows reproduced in every decision field | 220 of 220 | 2 |
| Mismatches in k, refits used, decision, T and p | 0, 0, 0, 0, 0 | 2 |
| Largest relative difference in T | 1.8e-16 | 2 |
| Detections, port against committed rows | 118 against 118 | 2 |

### Calibration and power on fresh ladders, additive noise (criteria 3 and 4)

| Metric | Value | Item |
|---|---|---|
| Null cell, 17 points: ladders rejected, rate, exact 95% interval | 6 of 200, 0.03, 0.0111 to 0.0642 | 3 |
| Null cell, 13 points | 10 of 200, 0.05, 0.0242 to 0.09 | 4 |
| Null cell, 9 points | 17 of 200, 0.085, 0.0503 to 0.1326 | 5 |
| Pooled null rejection, gates 0.07 pooled and 0.1 per cell | 33 of 600, 0.055, 0.0382 to 0.0764 | 6 |
| Power, positive control, 33 points, gate at least 36 | 40 of 40, 0.9119 to 1.0 | 7 |
| Power, weak cell, 9 points, DM-1b's power 0.05 | 2 of 40, 0.05, 0.0061 to 0.1692 | 8 |
| Criteria 1, 2, 3 and 4 | pass, pass, pass, pass | 9 |

### Post hoc: multiplicative noise (not preregistered)

| Metric | Value | Item |
|---|---|---|
| 17 points, loss 2.02 to 9.94: one-term ladders rejected | 35 of 200, 0.175, 0.125 to 0.2349 | 13 |
| 13 points, loss 2.02 to 5.49 | 24 of 200, 0.12, 0.0784 to 0.1733 | 14 |
| 9 points, loss 2.01 to 3.43 | 10 of 200, 0.05, 0.0242 to 0.09 | 15 |
| Pooled, relative noise sd 0.005 | 69 of 600, 0.115, 0.0906 to 0.1433 | 16 |

### Loud failure (criterion 5)

| Metric | Value | Item |
|---|---|---|
| pytest on test_fail_loud.py | 17 passed in 4.45s | 10 |

## Caveats

1. The additive-noise calibration is on DM-1's own generator: one-term law with E = 1.69 and exponent 0.34, ladders of 9 to 33 points between 1e5 and 1e9 parameters. It says nothing about ladders of other shapes.
2. The worst additive cell, 9 points with noise sd 0.02, came out at 0.085. DM-1b measured 0.070 for the same design. The interval 0.0503 to 0.1326 covers both, so this run neither confirms nor rules out a mild over-rejection on short ladders. It is inside the preregistered cell gate of 0.1.
3. The multiplicative-noise check has one relative noise level and one law. It shows the direction and rough size of the problem, not a calibration curve. Applying the port to the Porian grid, as folder 15 did, inherits this caveat, and I did not re-run folder 15.
4. Criterion 5 was not prospective: the tests were written and passed before the criteria were committed.
5. The fitter block is the exact DM-1b code. The same weak spots apply: DM-1b's fits can land on the bounds, and the bootstrap draws Gaussian noise whose sd comes from the one-term fit alone.
6. This ran in Python 3.13 with numpy 2.5.3, not the Python 3.11 the repository pins. The committed DM-1b rows were produced on a Slurm cluster (Sherlock job 48518244); I do not know its Python or numpy versions, and the match to 1.8e-16 holds across whatever the difference was.

## Appendix: what ran

1. Library: `py_src/two_term_lrt.py` (new; DM-1b's fitter block plus `bootstrap_lr_test`, `fit_one_and_two_term`, the dataclasses and the error classes) and `py_src/scaling_laws_fit.py` (`lr_test` wrapper, `lr_test_chi2_uncalibrated` kept with a warning, `fit_two_term_N_only` validation and convergence record, repaired smoke test).
2. Scripts: parity_dm1b.py, calibration_power.py, posthoc_multiplicative_noise.py, test_fail_loud.py, report_numbers.py, check_results_md.py. Every extra term is additive: `E + a n^-p + b n^-(p+d)` on the loss.
3. Harness and model: Claude Code, claude-sonnet-5-5, CPU, two cores. The QA reviewers that read this folder, and what they found, are recorded in `experiments/MEGA_QA_11-27-2026.md`; `check_results_md.py` was tightened after the second pass (it now fails on a malformed citation cell and on an integer that only matches the leading digits of a longer number) and still passes on this folder. No number in this folder changed.
4. Seeds: data `[cell, rep, 20261127]`, bootstrap the same with a trailing 1, parity sample 20261127.
5. Naming: in `two_term.alpha` and `two_term.beta`, alpha is the fast exponent (the larger one, matters at small N) and beta the slow one. DM-1b's own fit dictionary names them the other way round.
