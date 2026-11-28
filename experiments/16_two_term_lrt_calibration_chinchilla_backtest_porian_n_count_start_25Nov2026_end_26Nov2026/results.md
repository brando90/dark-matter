# CPU tests of the two-term scaling conjecture: the repo's likelihood-ratio test, extra terms on Chinchilla, and Porian parameter counts, results (11-26-2026)

**Doc link:** <https://github.com/brando90/dark-matter/blob/main/experiments/16_two_term_lrt_calibration_chinchilla_backtest_porian_n_count_start_25Nov2026_end_26Nov2026/results.md>

**TLDR:** The repo's two-term likelihood-ratio test cannot reject anything, and on the Chinchilla, Pythia, OPT and Porian data a second power of model size is neither supported nor excluded. The striking second data term on Chinchilla shrinks when the shortest runs are dropped, and the Porian parameter-count definition moves the compute-optimal exponent between about 0.52 and 0.75.

Date: 11-26-2026. Experiments run: 11-25-2026. As of: 11-27-2026 18:46 PST (mega QA corrections, appendix items 7 and 8).

Status: COMPLETE, with no pending phase. Verdict: NO EVIDENCE FOR OR AGAINST A SECOND N TERM; THE REPO'S TEST CANNOT REJECT. Nothing here was preregistered.

Owner: Claude Code, Claude Sonnet 5.5, on CPU. Full report: [REPORT.md](REPORT.md). Overview: [README.md](README.md).

## Bottom line

I found no evidence that a second, shallower power law in model size is a real feature of these data, and I cannot rule one out either. The likelihood-ratio test the repo ships never fires: on simulated ten-size ladders its largest statistic sits between about forty-three thousand and ninety-four thousand times below the five percent cutoff, so its p-value of one carries no information. A Gaussian version is roughly calibrated at ten sizes, with false-positive rates of five point three and two point seven percent, but even at ten sizes its power is only sixty percent at zero point three percent noise and eighteen percent at one percent noise (on the seven-size training ladders only its false-positive rate was measured, ten and two point seven percent), and the two-term fit lands on a boundary in about four of ten one-term ladders and half of the two-term ladders. On the Chinchilla points a second data term lowers the fit objective by thirty-seven and a half percent, but its gain in token extrapolation falls from seventy-nine to sixty-six percent when the four shortest runs are dropped and to nineteen percent when all eleven runs under one token per parameter are dropped; both cut-offs were chosen after seeing the residuals. On the Porian grid the way parameters are counted moves the compute-optimal exponent between about zero point five two and zero point seven five, and the apparent second size term at the small end lowers the fit objective by about ten point five to ten point eight percent when embeddings are counted and by at most about two point two five percent when they are not.

## Results

Every row is copied from results/report_numbers.txt, which report_numbers.py prints from the saved files in results/. The last column is the item number in that file. Run check_results_md.py to confirm that every number in a row appears in the items the row cites. Pairs written as "x vs y" read one-term versus two-term. Wins, ties and losses are written [wins, ties, losses] against the joint fit, and a held-out difference within 1e-6 is a tie.

### E1: calibration and power of the two-term test on simulated ladders

| Metric | 0.3% noise | 1% noise | Item |
|---|---|---|---|
| Repo statistic 2 x (Huber loss gain), largest value over all 250 ladders (5% cutoff 5.99) | 6.40e-05 | 1.38e-04 | 2 |
| Repo statistic, cutoff divided by largest value | 93,597 | 43,482 | 2 |
| Repo statistic, nominal false-positive rate and nominal power | 0% and 0% | 0% and 0% | 2 |
| Ladders where the two-term fit gains exactly nothing, one-term truth and two-term truth | 46% and 3% | 51% and 20% | 3 |
| Repo statistic, ranking quality (AUC) | 0.899 | 0.728 | 4 |
| Repo statistic, power at the 95th percentile of its own null | 66% | 36% | 4 |
| Gaussian statistic, 10 sizes, false-positive rate | 5.3% | 2.7% | 5 |
| Gaussian statistic, 10 sizes, nominal power | 60% | 18% | 5 |
| Gaussian statistic, 10 sizes, AUC | 0.900 | 0.770 | 5 |
| Gaussian statistic, 7 sizes, false-positive rate | 10.0% | 2.7% | 5 |
| Two-term fit on a boundary, one-term truth and two-term truth | 41% and 50% | 37% and 49% | 6 |
| Median fitted beta (truth 0.15) and share within 0.05 of truth | 0.23 and 8% | 0.21 and 11% | 7 |
| Forecast error, two-term truth, median, one-term vs two-term | 0.00844 vs 0.00688 | 0.01236 vs 0.01474 | 8 |
| Forecast error, one-term truth, median, one-term vs two-term | 0.00277 vs 0.00488 | 0.01447 vs 0.01774 | 9 |
| Gated forecast error, two-term truth, median | 0.00797 | 0.01241 | 10 |

### E2 and E2b: the joint law and extra terms on the 245 Chinchilla points

| Metric | Value | Item |
|---|---|---|
| Repo smoke test: statistic, p-value, points selected, joint points fitted, alpha and beta returned | -0.00, 1.000, 0, 400, 0.4832, 0.2105 | 11 |
| Multi-start joint fit: E, A, alpha, B, beta | 1.897, 529.4, 0.3529, 1.358e+04, 0.4558 | 13 |
| Multi-start joint fit: Huber objective, residual sum of squares | 0.001828, 0.1374 | 13 |
| Repo single-start fit: Huber objective, residual sum of squares | 0.001850, 0.17496 | 13 |
| Compute-optimal exponent beta/(alpha+beta): multi-start, repo single-start | 0.5636, 0.5491 | 13 |
| Second N term: Huber objective | 0.001828 | 14 |
| Second D term: Huber objective, residual sum of squares | 0.001143 (37.5% lower), 0.0526 (62% lower) | 14 |
| Held-out median error, larger models: joint / +N / +D / both / repo | 0.0441 / 0.0475 / 0.0347 / 0.0406 / 0.0452 | 15 |
| Held-out median error, more tokens: joint / +N / +D / both / repo | 0.0265 / 0.0265 / 0.0056 / 0.0056 / 0.0226 | 16 |
| Held-out median error, more compute: joint / +N / +D / both / repo | 0.0175 / 0.0180 / 0.0086 / 0.0092 / 0.0156 | 17 |
| Median over 12 splits: joint / +N / +D / both / repo | 0.0265 / 0.0265 / 0.0086 / 0.0092 / 0.0226 | 18 |
| Wins, ties, losses over 12 splits: +D / +N / both | [11, 1, 0] / [1, 7, 4] / [9, 0, 3] | 18 |
| Boundary-flagged fits among 12 splits: joint / +N / +D / both | 0 / 5 / 7 / 10 | 19 |
| Profile band for data exponent beta within 1% and within 5% | 0.420 to 0.492 (exponent 0.545 to 0.584) and 0.366 to 0.530 (exponent 0.505 to 0.604) | 20 |
| Profile band for size exponent alpha within 1% and within 5% | 0.320 to 0.380 (exponent 0.545 to 0.587) and 0.285 to 0.410 (exponent 0.529 to 0.625) | 21 |
| Four shortest runs (D/N below 0.35): share of residual sum of squares, share of Huber objective | 90%, 35% | 35 |
| Root-mean-square residual, other 241 runs vs all 245 runs | 0.0075 vs 0.0237 | 35 |
| Joint fit compute-optimal exponent: all runs / without the 4 shortest / without the 11 shortest | 0.5636 / 0.5132 / 0.4922 | 36 |
| Joint fit Huber objective: all runs / without 4 / without 11 | 1.828e-03 / 1.075e-03 / 8.371e-04 | 36 |
| Second D term, Huber objective lower by: all runs / without 4 / without 11 | 37.5% / 22.4% / 10.6% | 37 |
| Token-axis held-out median error, joint to +D: all runs | 0.0265 to 0.0056 | 38 |
| Token-axis held-out median error, joint to +D: without the 4 shortest | 0.0180 to 0.0062 | 38 |
| Token-axis held-out median error, joint to +D: without the 11 shortest | 0.0100 to 0.0081 | 38 |
| Wins, ties, losses for +D over twelve cuts: all runs / without 4 / without 11 | [11, 1, 0] / [11, 1, 0] / [5, 6, 1] | 40 |
| Median over twelve cuts, repo single-start against multi-start joint: all runs / without 4 / without 11 | 0.0226 against 0.0265 / 0.0104 against 0.0147 / 0.0095 against 0.0096 | 41 |
| Huber objective of the two-decimal AGENTS.md constants: all runs / without 4 / without 11 | 1.190e-02 / 1.129e-02 / 1.112e-02 | 42 |

### E3: real fixed-data ladders (loss is the natural log of LAMBADA perplexity at final checkpoints)

| Metric | Value | Item |
|---|---|---|
| Bootstrap p-value, 200 draws: pythia / pythia-deduped / opt | 0.542 / 0.214 / 1.000 | 22, 23, 24 |
| Gaussian likelihood-ratio statistic: pythia / pythia-deduped / opt | 0.006 / 2.003 / 0.000 | 22, 23, 24 |
| Rolling-origin forecasts, two-term against one-term: wins, ties, losses | 0, 12, 1 | 25 |
| Mean absolute log error over the 13 forecasts, one-term vs two-term | 0.02852 vs 0.02868 | 25 |

### E4: the Porian grid, parameter-count definition and the small-model bend

| Metric | Value | Item |
|---|---|---|
| RefinedWeb compute-optimal exponent by N definition: width times depth / exact label / 12ld2 / 12ld2+emb | 0.667 / 0.587 / 0.752 / 0.566 | 27 |
| OpenWebText2 compute-optimal exponent by N definition: width times depth / exact label / 12ld2 / 12ld2+emb | 0.638 / 0.532 / 0.726 / 0.520 | 28 |
| Width times depth minus exact label, exponent: RefinedWeb / OpenWebText2 | +0.080 / +0.106 | 27, 28 |
| Width times depth minus exact label, irreducible loss E: RefinedWeb / OpenWebText2 | -0.417 / -0.467 | 27, 28 |
| RefinedWeb exponent, width times depth vs exact label, minimum steps 1 / 500 / 1000 / 2000 | 0.666 vs 0.494 / 0.702 vs 0.585 / 0.667 vs 0.587 / 0.707 vs 0.550 | 29 |
| OpenWebText2 exponent, width times depth vs exact label, minimum steps 1 / 500 / 1000 / 2000 | 0.681 vs 0.515 / 0.690 vs 0.580 / 0.638 vs 0.532 / 0.676 vs 0.547 | 29 |
| RefinedWeb second N term, fit-objective reduction on all 15 sizes: width times depth / exact label / no embeddings / plus embeddings | 1.36% / 8.51% / 2.25% / 10.83% | 32 |
| OpenWebText2 second N term, fit-objective reduction on all 15 sizes: width times depth / exact label / no embeddings / plus embeddings | 0.07% / 13.14% / 0.03% / 10.47% | 33 |
| RefinedWeb second N term, exact label, as the smallest kept model grows: all / 30M / 60M / 120M and up | 8.5% / 0.0% / 0.0% / 0.4% | 30 |
| OpenWebText2 second N term, exact label, as the smallest kept model grows: all / 30M / 60M / 120M and up | 13.1% / 0.0% / 0.0% / 0.9% | 31 |
| Porian backtests over 8 splits, median held-out error at cap 1.5: joint / +N / +D / both | 0.0170 / 0.0159 / 0.0132 / 0.0124 | 34 |
| Porian backtests, wins, ties, losses at cap 1.5: +N / +D / both | [6, 0, 2] / [6, 0, 2] / [6, 0, 2] | 34 |
| Porian backtests, boundary-flagged fits out of 8: joint / +N / +D / both | 0 / 7 / 7 / 8 | 34 |

### X: cross-check of experiment 09 from its saved predictions

| Metric | Value | Item |
|---|---|---|
| Experiment 09 size-holdout test root-mean-square error, Chinchilla runs, joint law | 0.2501 | 43 |
| Share of that squared error from the 3 test runs with D/N below 0.35, and the error without them | 98.3%, 0.0336 | 43 |

## Caveats

1. The simulation settings are my choices: the sizes, the two noise levels, one two-term truth, the seven-and-three forecast split, the exponent caps and the fit budgets. A truth with beta closer to alpha or a larger shallow amplitude would change the power numbers.
2. The real ladders have seven or eight sizes and thirteen forecasts in total, and the metric is LAMBADA perplexity at final checkpoints, not a pre-training loss, so they have low power.
3. The "within one percent" and "within five percent" bands are heuristics on the objective, not confidence intervals, and the point estimate moves by more than the width of the one percent band when a handful of runs are dropped.
4. The 245 Chinchilla points come from the repo's data folder, and I did not verify their provenance. Why the four shortest runs sit above the fitted curve is untested, and I tested only the low side of tokens per parameter.
5. On the Porian grid there is one learning rate and batch size. The minimum-steps filter is a choice, and the embedding explanation for the small-model bend is a hypothesis that this folder does not confirm.
6. None of this was preregistered. The E2b cut-offs (D/N of 0.35 and 1) were chosen after seeing the residuals, so every win count and percentage from E2b is descriptive, not a test.
7. scaling_ext.py and safe_pickle.py are new code, checked against the repo's fitter, by seed stability and by fifty-five loader tests, and read by one four-reviewer pass.

## Appendix: methods, harness, model and context

1. Harness and model: Claude Code cloud session, Claude Sonnet 5.5 (claude-sonnet-5-5). Every number is computed by the scripts in this folder; no language model judged any result. CPU only, two cores, no GPU, no cluster and no paid API.
2. Environment: CPython 3.13.18, which is outside the repo's 3.11 pin, with numpy 2.5.3, scipy 1.18.1, pandas 3.0.5 and matplotlib 3.11.2. The commands to rerun every experiment are in the Reproduce section of REPORT.md.
3. Data: the 245 Chinchilla points and the Pythia, deduplicated Pythia and OPT ladders from the repo's Hitchhiker's Guide files, and the Porian et al. grid, loaded through py_src/safe_pickle.py with a pinned SHA-256 digest.
4. Review: a four-reviewer pass by Claude Opus 5.5 at maximum effort found no critical issues, and a fresh re-verification found no critical or major issue. All fixes are applied in REPORT.md.
5. Relation to other folders as of commit 3626ba0: folder 15 (ub9sc) finds the second term convention dependent, with 1 of 38 bootstrap detections when embeddings are left out of the count and 13 of 38 and 12 of 38 with the tied and untied counts. My E4 points the same way in direction only, on a different slice and without a calibrated test. Folder 14 (ub9sc) is scientifically unresolved because its control gates fail, and it does not test the Chinchilla finding above. Folder 11, the embedding ladder, has only its README on main.
6. Provenance check: check_results_md.py reads the tables above and fails if any number in a row is missing from the items that row cites.
7. Corrections on 11-27-2026, after a mega QA pass: the Gaussian statistic's nominal power of 60% and 18% and its area under the curve of 0.900 and 0.770 were measured on the ten-size ladders and are now labelled that way (they had been labelled seven-size, where only the false-positive rate was computed); the bootstrap p-value of the OPT ladder is 1.000, not 0.647, because its collapsed two-term fit gave a statistic of minus 9e-16 that is now clamped at zero and the OPT test was recomputed with the same seeds. No other number changed.
8. Mega QA record, 11-27-2026: the reviewer Carbon read this folder and the library code at commits 3221308 and c49cd63, and every finding that touched this folder was fixed or rejected with evidence in `experiments/MEGA_QA_11-27-2026.md`. After the second pass the 150-line `scaling_ext.fit` was split into helpers; `test_scaling_ext_golden.py` shows that 11 fits (every fitting mode) return the same 107 numbers as before, and `test_scaling_ext_vs_c49cd63.py` finds 0 differences in 38 random calls against commit c49cd63. `check_results_md.py` now also fails on an integer that only matches the leading digits of a longer number and on a table row whose last cell is not a list of item numbers; it passes on this folder. No number in this folder changed.
