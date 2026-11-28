# Experiment 19 (11-27-2026): token-budget test redesigned as a calibrated nested test, controls first

**Doc link:** <https://github.com/brando90/dark-matter/blob/main/experiments/19_token_budget_nested_lrt_controls_first_start_27Nov2026_end_27Nov2026/README.md>

**TLDR:** Folder 14 asked whether the additive data term (the token budget) explains the small-end excess of Porian et al.'s tuned-constant ladders, but its extrapolation estimator failed its own controls. This folder replaces it with a nested likelihood-ratio test on all sizes at once and fixes three control gates (null at most 4 of 20 detections, planted +0.3 and +0.1 at least 16 of 20) before any control or real fit is run; the real ladders are analysed only for a dataset whose three gates pass.

Date: 11-27-2026. Owner: Claude Code, Claude Sonnet 5.5 (not the pinned Fable 5.1 max), Python CPU in a two-core sandbox, no cluster.
Status: **DONE 11-27-2026.** Preregistered in commit 0683e8a; the controls and the gated real-data stage were run afterwards and no rule, grid, seed or control was changed. Result: section at the end and [results.md](results.md).

## Dark-matter question

Why do the tiniest models live off the scaling curve, and what correction brings them back? Here: does the additive Chinchilla data term B D^-beta, which grows as tokens shrink with model size at matched tokens per parameter, absorb the small-end excess, or does a second, faster term in N remain after it is included? Law forms are additive throughout: E + A n^-alpha + B d^-beta, never multiplicative.

## Why folder 14's estimator is replaced

Folder 14 (E2, ub9sc) fitted the joint law on reference sizes of at least 108 million parameters and read the excess of the 5, 9 and 15 million models against its extrapolation, 20 times down in N. Its own fixed controls failed both versions (`14_*/results.md`, lines 115 to 118): positive control J EXPLAINS 4 of 20 (amended 3 of 20) and M RETURNS 17 of 20 (amended 0 of 20); planted-fast-term control J not EXPLAINS 17 of 20 and M STAYS OFF 10 of 20 (amended 9 of 20); the gate was 16 of 20 each. Recomputed from `14_*/results/verify_controls.csv` (amended run, 20 replicates per condition): with no fast term the median absolute matched excess is 0.0981 nats against the 0.006 nat threshold (two synthetic sd), and the median half-width of its 95% interval is 0.249 nats; the noiseless diagnostic `deterministic_diagnostic.csv` shows that with the +0.3 nat fast term planted, the joint extrapolation already absorbs 53.4% of the excess. An extrapolation over a factor of 20 in N cannot separate the data term from a fast N term at these noise levels, whatever the thresholds. Folder 14 and its results are left unchanged.

## New estimand and estimator

Models, with n = N / 1e6 and d = D / 1e9: M1 is the joint law y = E + A n^-alpha + B d^-beta; M2 adds one fast N term, y = E + A n^-alpha + B d^-beta + C n^-gamma with gamma at least alpha + 0.15; all coefficients are non-negative. M1 is nested in M2 (C = 0). The question becomes: given the data term, is a fast N term still needed? "Detected" means yes, so the token budget alone does not explain the small end; "not detected" with demonstrated power means the data term suffices at the planted effect sizes below.

Estimator (`py_src/nested_joint_lrt.py`, tests in `test_nested_lrt.py`): weighted least squares with weights 1 over sd squared; for fixed exponents the coefficients solve a non-negative least-squares problem; exponents are searched on a grid (alpha 40 values from 0.05 to 1.5, beta 32 from 0.05 to 1.0, gamma 20 from 0.3 to 4.0, geometric) with batched normal equations, and the best 2 (M1) or 3 (M2) distinct grid cells are polished by Nelder-Mead on the exact non-negative RSS. Statistic T = npts ln(RSS1 / RSS2) with RSS2 clipped at RSS1 so T is at least 0. p-value from a parametric bootstrap under the fitted M1 with noise scale sqrt(RSS1 / (npts - 5)) times each point's relative sd, every draw refitted by both models with the same procedure; p = (1 + number of bootstrap T at least T) / (B + 1); detect iff p is at most 0.05. The statistic and the bootstrap are scale free in the weights, so only the relative sd across points matters. The test assumes independent noise across points.

## Data and weights

`build_designs.py` extracts, for rw and owt2 separately, the tuned-constant recipe read at 5, 10, 20, 40 and 80 tokens per parameter exactly as folder 14's `ladder()` does (tied parameter count, 65 points per dataset, sizes 5.2 million to 902 million). Weights use DM-2's seed sd (7 seeds of the tuned recipe), interpolated linearly in log width on log sd between the measured widths 96 to 480 and held at the width-480 value above it (folder 14 holds widths 640 and 832 the same way). The real losses in `results/design_<ds>.csv` were extracted without being fitted. The untied count is not run.

## Controls (fixed now)

For each dataset, synthetic responses use that dataset's 65 real (N, D) points and real per-point sd, with folder 14's truth y = 2.0 + 3.0 n^-0.6 + 1.0 d^-0.2 (the planted fast term is c (N / 5173248)^-1.4). Twenty replicates per control, B = 200 bootstrap draws each, seeds `default_rng([20261127, dataset index, control index, replicate])`.

| Control | Truth and noise | Gate |
|---|---|---|
| G1 null | joint law only, independent noise with the real sd | detections at most 4 of 20 |
| G2 fast03 | plus 0.3 x (N / 5173248)^-1.4 | detections at least 16 of 20 |
| G3 fast01 | plus 0.1 x (N / 5173248)^-1.4 | detections at least 16 of 20 |
| S1 corr_noise | joint law only, noise shared within a width with correlation 0.5 | report only |
| S2 dterm_misfit | joint law plus 0.1 x min(d) x d^-1.0 (a misspecified data term, 0.1 nats at the smallest d) | report only |

A dataset passes iff G1, G2 and G3 all pass for that dataset. S1 and S2 are stress tests of the two assumptions most likely to fail on real data (independent noise; a correct power-law data term); they gate nothing but are reported beside every real-data verdict.

## Real-data stage (gated; `real_data.py`)

Only for a dataset whose three gates passed: bootstrap LRT on the real ladder with B = 1000 and a fixed seed. Verdict words: FAST TERM DETECTED (p at most 0.05) or NOT DETECTED. Descriptive: fitted fast-term size at 5.17 million parameters, fitted exponents, and a sensitivity p-value from a bootstrap with within-width correlated noise (correlation 0.5, B = 500). If a dataset fails any gate, no real-data fit is run for it and the failed gate is the result; no threshold, grid, seed or control is changed afterwards. If G1 passes but G2 or G3 fails, the estimator is reported as underpowered at that size.

## Preregistered predictions (to be judged in results.md)

1. G1 passes on both datasets (at most 4 of 20 false detections).
2. G2 passes on both datasets (at least 16 of 20 at +0.3).
3. G3 passes on both datasets (at least 16 of 20 at +0.1); the doubt is rw, whose 5 million sd is 0.062.
4. S2 (misspecified data term) produces at least 5 of 20 false detections on at least one dataset, so a detection on real data is not by itself proof of a fast N term.
5. S1 (correlated noise) produces more false detections than G1 on at least one dataset.

## What is and is not blind

Porian's small-end excess on these ladders is known from DM-2, DM-3 and folder 14 (+0.347 rw and +0.371 owt2 at 5 million against the one-term reverse ladder), so the real data are not blind. This estimator was not applied to them before the controls. Before this commit the library was run on a few synthetic responses on the rw grid (one tiny-noise, one null, one planted +0.3, and a 20-draw bootstrap) for debugging and timing, with seeds that no gate replicate uses; no gate control was run and no real loss was fitted.

## Disclosures

1. The estimand differs from folder 14's (a likelihood-ratio test on all sizes, not the extrapolated excess of the smallest sizes), so this folder does not rescue folder 14's result; it asks a related question with a calibrated tool.
2. Control noise follows the real per-size seed sd, not folder 14's constant 0.003; the real sd at 5 million is 0.062 on rw.
3. Environment: Python 3.13, numpy 2.5, scipy 1.18 in a two-core sandbox; the repository pins Python 3.11.
4. The model that ran is Claude Sonnet 5.5; the suggested model for this task was Claude Code Fable 5.1 at max effort, which was not used.
5. Grid-plus-polish fitting finds local optima in a flat landscape; the bootstrap applies the same procedure to every draw, so calibration holds for the procedure, not for a global optimum.

## Files and rerun

`build_designs.py`, `controls.py`, `summarize_controls.py`, `real_data.py` (gated), `test_nested_lrt.py`, library `py_src/nested_joint_lrt.py`.

```bash
python3 -I experiments/19_token_budget_nested_lrt_controls_first_start_27Nov2026_end_27Nov2026/test_nested_lrt.py
python3 -I experiments/19_token_budget_nested_lrt_controls_first_start_27Nov2026_end_27Nov2026/controls.py --workers 2
python3 -I experiments/19_token_budget_nested_lrt_controls_first_start_27Nov2026_end_27Nov2026/summarize_controls.py
python3 -I experiments/19_token_budget_nested_lrt_controls_first_start_27Nov2026_end_27Nov2026/real_data.py
```

## Result

Verdict: **owt2: FAST TERM DETECTED (T 4.652, p 0.034965 with 1000 draws), not robust to correlated noise (p 0.121756 at correlation 0.5). rw: no verdict, because gate G3 failed at 15 of 20 (16 needed), so no real-data fit was run.** Gates: null 2 of 20 false detections on both datasets, planted +0.3 detected 20 of 20 on both, planted +0.1 detected 15 of 20 on rw and 20 of 20 on owt2. Preregistered predictions 1, 2 and 5 passed; 3 (rw 15 of 20) and 4 (the misspecified data term gave 0 of 20 false detections, not at least 5) failed. Tables, the post hoc caveat that the fitted fast-term size is not readable alone (under the null a fitted fast term is often the slow term relabelled), and the provenance-checked numbers are in [results.md](results.md), `results/report_numbers.txt` and `check_results_md.py`.
