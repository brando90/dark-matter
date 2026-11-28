# Experiment 19 (11-27-2026): nested joint-law test of a fast N term, controls first, results

**Doc link:** <https://github.com/brando90/dark-matter/blob/main/experiments/19_token_budget_nested_lrt_controls_first_start_27Nov2026_end_27Nov2026/results.md>

**TLDR:** The redesigned nested likelihood-ratio test keeps its size under independent noise (2 of 20 false detections on each dataset) and finds a planted 0.3 nat fast term every time (20 of 20), but it finds a planted 0.1 nat term on rw only 15 of 20 times against a gate of 16, so rw gets no real-data fit; on owt2, which passed all three gates, it detects a fast term (p = 0.034965), and that detection does not survive correlated noise (p = 0.121756). Three of the five preregistered predictions passed and two failed.

Date: 11-27-2026. Owner: Claude Code, Claude Sonnet 5.5 (not the pinned Fable 5.1 max), Python CPU in a two-core sandbox, no cluster. Folder created from preregistration commit 0683e8a, before any control or real fit was run.

## Bottom line

Verdict: **OWT2: FAST TERM DETECTED (T 4.652, p 0.034965), NOT ROBUST TO CORRELATED NOISE (p 0.121756). RW: NO VERDICT, GATE G3 FAILED AT 15 OF 20 (16 NEEDED), SO NO REAL-DATA FIT WAS RUN.**

The test behaved as designed wherever the gates could judge it. With the joint law as the truth and independent noise it falsely detected a fast term in 2 of 20 replicates on both datasets, against a limit of 4 (items 3, 8). With a planted fast term of 0.3 nats at the smallest model it detected it in 20 of 20 on both (items 4, 9). With 0.1 nats it detected it in 20 of 20 on owt2 but in 15 of 20 on rw, one short of the preregistered 16 (items 5, 10). The preregistration named rw as the doubt, because its per-point sd at the smallest model is 0.0619 against 0.0123 on owt2 (items 1, 2). The five rw misses had p-values of 0.0547, 0.0746, 0.2836, 0.3383 and 0.3582 (item 30), and the interval on 15 of 20, 0.53 to 0.89, contains 16 of 20, so rw is better read as power at 0.1 nats not shown than as power shown to be low. No gate, grid, seed or control was changed after seeing this.

On owt2, the only dataset whose three gates passed, the real-ladder test detects a fast term: T = 4.652 and p = 0.034965 from 1000 bootstrap draws, a fitted fast term of 0.194 nats at the 5.17 million parameter model, with the additive data term already in the joint law (items 16, 33). Read as preregistered, that says a fast N term is still needed after the token budget, so the token budget alone does not explain owt2's small-end excess. The evidence is weak. The p-value sits just under 0.05. With noise correlated within a width (correlation 0.5, 500 draws) it is 0.121756 (item 18), and the report-only control S1 shows how much that assumption matters: 4 of 20 and 6 of 20 false detections on rw and owt2 against 2 of 20 and 2 of 20 for independent noise (item 23). The fit's noise scale is 9.512 times the per-point seed sd, where the null controls give a median of 1.028 (items 16, 32), so, if DM-2's seed sd is right, most of the scatter around the joint law is something other than seed noise, and the independent-noise bootstrap treats it as if it were. The fitted term is steep (gamma 2.019) and lives in the smallest models: 0.194 nats at 5.17 million parameters, 0.091 at 7.50 million, 0.053 at 9.81 million, and below 0.01 from 22.5 million upward (item 33).

## Preregistered predictions

| # | Prediction | Result | Verdict | Items |
|---|---|---|---|---|
| 1 | G1 passes on both datasets (at most 4 of 20 false detections) | rw 2 of 20, owt2 2 of 20 | PASSED | 19 |
| 2 | G2 passes on both datasets (at least 16 of 20 at the planted +0.3 term) | 20 of 20 on both | PASSED | 20, 4, 9 |
| 3 | G3 passes on both datasets (at least 16 of 20 at the planted +0.1 term; the doubt is rw) | rw 15 of 20 (95% Wilson interval 0.53 to 0.89), owt2 20 of 20 | FAILED | 21, 5, 10 |
| 4 | S2 (misspecified data term) gives at least 5 of 20 false detections on at least one dataset | 0 of 20 on both | FAILED | 22 |
| 5 | S1 (correlated noise) gives more false detections than G1 on at least one dataset | rw 4 against 2, owt2 6 against 2 | PASSED | 23 |

Prediction 4 failed in the direction that favours the test: the misspecified data term produced no false detections. It was one misspecification of one size, so it does not show robustness to others.

## Gates and stress controls

Twenty replicates per control, 200 bootstrap draws each, detect means p at most 0.05. A median p of 0.0050 is the smallest value 200 draws can give. Synthetic responses use each dataset's 65 real (N, D) points and real per-point sd.

| Dataset | Control | Detections | 95% Wilson interval | Median p | Median T | Gate rule | Verdict | Items |
|---|---|---|---|---|---|---|---|---|
| rw | G1 null (joint law only) | 2 of 20 | 0.03 to 0.30 | 0.2090 | 1.265 | at most 4 of 20 | PASSED | 3 |
| rw | G2 planted +0.3 | 20 of 20 | 0.84 to 1.00 | 0.0050 | 39.582 | at least 16 of 20 | PASSED | 4 |
| rw | G3 planted +0.1 | 15 of 20 | 0.53 to 0.89 | 0.0274 | 6.258 | at least 16 of 20 | FAILED | 5 |
| rw | S1 correlated noise | 4 of 20 | 0.08 to 0.42 | 0.6592 | 0.000 | report only | none | 6 |
| rw | S2 misspecified data term | 0 of 20 | 0.00 to 0.16 | 0.7090 | 0.000 | report only | none | 7 |
| owt2 | G1 null (joint law only) | 2 of 20 | 0.03 to 0.30 | 0.5522 | 0.025 | at most 4 of 20 | PASSED | 8 |
| owt2 | G2 planted +0.3 | 20 of 20 | 0.84 to 1.00 | 0.0050 | 56.910 | at least 16 of 20 | PASSED | 9 |
| owt2 | G3 planted +0.1 | 20 of 20 | 0.84 to 1.00 | 0.0050 | 12.305 | at least 16 of 20 | PASSED | 10 |
| owt2 | S1 correlated noise | 6 of 20 | 0.15 to 0.52 | 0.1791 | 1.395 | report only | none | 11 |
| owt2 | S2 misspecified data term | 0 of 20 | 0.00 to 0.16 | 0.9279 | 0.000 | report only | none | 12 |

A dataset passes iff G1, G2 and G3 all pass for it: owt2 passed all three and rw failed G3 (items 13, 14). By the preregistered rule the real-data stage ran for owt2 only (item 15).

## Real-data stage (owt2 only, 65 points)

| Quantity | Value | Items |
|---|---|---|
| Statistic T | 4.652 | 16 |
| Bootstrap p-value, independent noise, 1000 draws | 0.034965 | 16 |
| Preregistered verdict | FAST TERM DETECTED | 16 |
| Sensitivity p-value, correlation 0.5 within a width, 500 draws | 0.121756 | 18 |
| Noise scale sigma_hat as a multiple of the per-point sd | 9.512 | 16 |
| Fitted fast term at the 5.17 million parameter model, nats | 0.194 | 16, 33 |
| Fast-term exponent gamma | 2.019 | 17, 33 |
| Joint law (M1) exponents alpha and beta | 0.2069 and 0.3323 | 17 |
| With the fast term (M2) exponents alpha and beta | 0.1488 and 0.3320 | 17 |
| M2 coefficients E, A, B and C | 1.4713, 2.5560, 0.6485 and 5.3503 | 17 |

Both laws are additive, E + A n^-alpha + B d^-beta (+ C n^-gamma in M2), as everywhere in this repository.

## Reading the fitted fast-term size (post hoc, not preregistered)

Under the null, the fitted M2 often reports a large fast term that is only the true slow term relabelled. In 10 of 20 owt2 replicates and 7 of 20 rw replicates the reported fast term at the smallest model is above one nat, where the true slow term A n^-alpha is 1.119 nats; their median fitted gamma is 0.604 and 0.616, against a true alpha of 0.6, and for owt2 8 of the 10 have T below 1.0 and p above 0.05 (items 27, 24). T and p are not affected, because every bootstrap draw is refitted by the same procedure, but the descriptive size and exponents should be read only when the test detects. The real owt2 fit is not of that kind: its fast-term exponent is 2.019 rather than about 0.6, and its 0.194 nats lies between the recovered medians for the planted 0.1 (0.101) and 0.3 (0.251) controls (items 33, 10, 9).

## Caveats

1. The estimand differs from folder 14's (a likelihood-ratio test on all sizes, not the extrapolated excess of the smallest sizes). This folder does not rescue folder 14, whose results stay as published and unresolved.
2. The verdict is conditional on independent noise. The real fit's sigma_hat is 9.512 times the seed sd against null-control medians of 1.003 on rw and 1.028 on owt2 (items 31, 32, 16), S1 inflates false detections on both datasets (item 23), and the owt2 p-value rises from 0.034965 to 0.121756 at correlation 0.5. The sensitivity run is one correlation value, not a calibrated alternative.
3. One test, one dataset, p just under 0.05. A post hoc remark: with a Bonferroni split of 0.05 over the two datasets that were candidates for the real-data stage, the threshold would be 0.025, which p = 0.034965 does not meet.
4. The detection rests on the smallest models (item 33). Any systematic effect at those sizes, such as a recipe artefact, would be absorbed as a fast term. A leave-the-smallest-size-out refit was not run.
5. rw has no verdict. The miss was narrow (15 of 20 against 16, interval 0.53 to 0.89), and the estimator was not shown to have power at 0.1 nats at rw's noise level; that is the result for rw.
6. Weights use DM-2's seed sd, interpolated in log width between measured widths 96 and 480 and held at the width-480 value above, so the sd of the larger models is an extrapolation.
7. S2 was one misspecified data term (0.1 nats at the smallest d, exponent 1.0), and it produced 0 of 20 false detections on both datasets, so prediction 4 failed. Other misspecifications, including a different noise structure, are untested.
8. Grid-plus-polish fitting finds local optima in a flat landscape; the bootstrap applies the same procedure to every draw, so calibration holds for the procedure, not for a global optimum.
9. Environment: Python 3.13.18, numpy 2.5.3 and scipy 1.18.1 in a two-core sandbox; the repository pins Python 3.11. The model that ran is Claude Sonnet 5.5, not the suggested Claude Code Fable 5.1 at max effort.
10. Two reporting-script bugs surfaced at the first summary run and were fixed without touching any gate rule or control result: `DataFrame.T` is the transpose, not the statistic column, and `read_csv` turned the control name "null" into a missing value. The same fix went into the resume-read of `controls.py`; the control files hold exactly 20 distinct replicates per control.
11. The real-data stage ran after the gate controls and while the report-only stress controls were still running; the stress controls gate nothing, so the order does not affect any verdict.
12. Cost: the 100 control replicates of each dataset took 0.22 CPU hours in total, a median of 7.8 seconds each on both (items 34, 35).
13. Mega QA record, 11-27-2026 (reviewer Carbon, at commit c49cd63; see `experiments/MEGA_QA_11-27-2026.md`): its one major finding was that `Design.fit` could in principle return a fit that breaks the constraint gamma at least alpha plus 0.15. `verify_f1_saved_workload.py` found no such start cell among the 208,510 picked on this folder's workload, and it reproduced 20 of the 200 control replicates and the real owt2 stage exactly, so no number here changed. The library now masks such cells, returns an infinite sum of squares for a polished point that breaks a constraint, and raises `InsufficientDataError` when no feasible fit exists; the unit tests grew from 9 to 17 checks. Carbon's last pass was not repeated after these fixes.

## What would move this next

1. Rerun the owt2 test with a bootstrap that carries a random effect per width (or resamples whole widths) and repeat G1 to G3 under it; sigma_hat of 9.512 says the joint law misses structure that seed noise does not explain.
2. Preregister a larger control (for example 100 replicates per control) so rw's power at 0.1 nats has an interval narrower than 0.53 to 0.89, and decide rw then.
3. Leave the smallest one, two and three sizes out of the owt2 fit and see whether the detection survives, labelled post hoc.
4. Apply the same gated procedure to an independent ladder, so a detection does not rest on one dataset at p = 0.034965.

## Rerun

`python3 -I experiments/19_token_budget_nested_lrt_controls_first_start_27Nov2026_end_27Nov2026/test_nested_lrt.py`, then `controls.py --workers 2 --only gates`, `summarize_controls.py`, `real_data.py` (reads `results/controls_summary.json` and refuses a dataset whose gates failed), `controls.py --workers 2 --only stress`, `summarize_controls.py` again, `python3 -I report_numbers.py > results/report_numbers.txt` and `python3 -I check_results_md.py`.
