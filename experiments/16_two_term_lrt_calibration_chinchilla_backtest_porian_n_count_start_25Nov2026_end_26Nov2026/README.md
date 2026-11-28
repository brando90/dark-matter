# Is there a second, shallower power law in model size? CPU tests of the two-term conjecture (11-26-2026)

**Doc link:** <https://github.com/brando90/dark-matter/blob/main/experiments/16_two_term_lrt_calibration_chinchilla_backtest_porian_n_count_start_25Nov2026_end_26Nov2026/README.md>

**TLDR:** Four CPU experiments and one robustness check ask whether the repo's two-term likelihood-ratio test works, and whether a second power of model size or of data size is a real feature of the Chinchilla, Pythia, OPT and Porian data. The test cannot reject anything, and the data neither support nor exclude a second size term.

Date: 11-26-2026. Experiments run: 11-25-2026. Status: COMPLETE, with no pending phase. Owner: Claude Code, Claude Sonnet 5.5, CPU only. Nothing here was preregistered.

## Hypothesis

The conjecture is that loss follows L(N) = E + A/N^alpha + B/N^beta with beta below alpha, so that a second, shallower power of model size N hides behind the first, and that the joint Chinchilla law E + A/N^alpha + B/D^beta also has room for an extra additive term. The additive form is the only one used here, and no multiplicative form appears anywhere. Because nothing was preregistered, the hypotheses are written as questions.

1. Can the repo's two-term likelihood-ratio test, lr_test in py_src/scaling_laws_fit.py, detect a second N term, and is it calibrated?
2. On real data, does an extra additive N term or D term improve held-out extrapolation, and does the answer survive dropping a few runs?
3. Does the way parameters are counted (width times depth, the exact label, with or without embeddings) change the compute-optimal exponent or the apparent second term?

## Method

Everything ran on CPU, in minutes, on two cores. Every extra term is additive.

1. E1 simulates one hundred and fifty one-term ladders and one hundred two-term ladders at each of two noise levels, zero point three and one percent, over ten model sizes from ten million to twelve billion parameters. It compares the repo's Huber-gain statistic with a Gaussian one for calibration, power, boundary fits and forecast error.
2. E2 fits the joint law, with and without a second N term or a second D term, to the two hundred and forty-five Chinchilla points in the repo's Hitchhiker's Guide data, with twelve train-small test-large splits, profiles of both exponents and full-data fits.
3. E2b repeats E2 without the four shortest runs and without the eleven runs under one token per parameter. These cut-offs were chosen after seeing the residuals, so this check is post hoc.
4. E3 runs a parametric bootstrap for the second term and thirteen rolling forecasts on Pythia, deduplicated Pythia and OPT, with loss taken as the natural log of LAMBADA perplexity at final checkpoints.
5. E4 refits the Porian grid under four definitions of N and tests whether the second N term survives when the smallest models are dropped.
6. A cross-check recomputes experiment 09's held-out error from its saved predictions.

## Result

Verdict: no evidence for or against a second N term, and the repo's test cannot reject. The numbers are in results.md and the full argument is in REPORT.md.

1. The repo's test is off by more than four orders of magnitude. On simulated ten-size ladders its largest statistic is between about forty-three thousand and ninety-four thousand times below the five percent cutoff, and its own smoke test returns a statistic of zero and a p-value of one. A Gaussian version is roughly calibrated at ten sizes, but even there it has only sixty percent power at low noise and eighteen percent at one percent noise.
2. The two-term fit is weakly identified. It lands on a boundary in about four of ten one-term ladders and half of the two-term ladders, and it recovers the shallow exponent to within zero point zero five in only eight to eleven percent of the two-term ladders.
3. On the Chinchilla points the fit switches a second N term off. A second D term lowers the fit objective by thirty-seven and a half percent, but its gain in token extrapolation falls from seventy-nine to sixty-six percent without the four shortest runs and to nineteen percent without the eleven runs under one token per parameter. The compute-optimal exponent moves from zero point five six to zero point five one and zero point four nine across the same three subsets.
4. On Pythia, deduplicated Pythia and OPT the bootstrap p-values are zero point five four, zero point two one and one (the OPT two-term fit collapses onto the one-term fit), and twelve of the thirteen rolling forecasts are ties between the one-term and two-term models.
5. On the Porian grid the definition of N moves the compute-optimal exponent between about zero point five two and zero point seven five. The apparent second size term lives at the small end: it lowers the fit objective by about ten point five to ten point eight percent when embeddings are counted and by at most about two point two five percent when they are not. Whether embedding accounting explains it is not settled here.
6. Experiment 09's held-out error on the Chinchilla runs is dominated by three runs with fewer than zero point three five tokens per parameter, which produce ninety-eight percent of its squared error.

## What would move this next

1. Port the bootstrap likelihood-ratio test already on main (folders dm1_power_analysis and 15_e4_porian_lrt_both_counts) into lr_test, and make fit_two_term_N_only fail loudly when its narrow-D selection is empty.
2. Re-run experiment 09 and experiments 02 to 04 with and without the runs under one token per parameter.
3. Analyze the embedding ladder in folder 11 when it lands, with the bootstrap test and the exact-count definitions.

## Relation to other folders on main

1. Folder 15 (ub9sc) finds the second term convention dependent: one of thirty-eight bootstrap tests detects it when embeddings are left out of the count, against thirteen of thirty-eight and twelve of thirty-eight with the tied and untied counts. My E4 points the same way in direction only, on a different slice and without a calibrated test.
2. Folder 14 (ub9sc) is scientifically unresolved because its preregistered control gates fail, and it does not test the Chinchilla finding in item 3 above.
3. py_src/scaling_ext.py overlaps py_src/conjecture_fits.py. I kept it in py_src because the scripts here import it from there; comparing the two fitters is still open.

## Files

1. REPORT.md: the full report, with the reproduce commands and the numbered reference list.
2. results.md: the bottom line, the results tables, caveats and the appendix. check_results_md.py confirms that every number in its tables appears in results/report_numbers.txt.
3. e1_lrt_calibration.py, e2_chinchilla_joint_backtest.py, e2b_short_run_robustness.py, e3_real_ladders.py, e4_porian_n_proxy.py and crosscheck_exp09.py: the experiments.
4. validate_scaling_ext.py, test_scaling_ext_golden.py (fit() against a golden file, results/golden_scaling_ext_fit.json), test_scaling_ext_vs_c49cd63.py (fit() against the commit before the 11-27-2026 refactor), test_safe_pickle.py, make_figures.py and report_numbers.py: checks, figures and the numbers file.
5. results/: the saved CSV, JSON and text outputs and four figures, about one and a half megabytes in total.
6. py_src/scaling_ext.py and py_src/safe_pickle.py: the two new library modules these scripts import.

## Rerun

Use the commands in the Reproduce section of REPORT.md. The shortest check is `python3 -I experiments/16_two_term_lrt_calibration_chinchilla_backtest_porian_n_count_start_25Nov2026_end_26Nov2026/test_safe_pickle.py`, which runs fifty-five loader tests, followed by `python3 -I experiments/16_two_term_lrt_calibration_chinchilla_backtest_porian_n_count_start_25Nov2026_end_26Nov2026/check_results_md.py`.
