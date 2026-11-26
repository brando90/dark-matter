# E3 results (11-25-2026): does the E1 parameter-count correction forecast larger models better?

As of 11-25-2026. Run by ub9sc with **Claude Code, Opus 5.5 (claude-opus-5-5), fits in Python (numpy/scipy) on
skampere2, CPU only**. PREREG.md was committed and pushed before any fit (85db67d). `python verify.py`: **ALL PASS**
(4 checks; results/verify.txt).

## Bottom line for the dark-matter question
Re-weighting embedding parameters (1-term law in N_c, c fitted on the training sizes only) gives a small, consistent
gain over the plain 1-term fit at >= 10x extrapolation: +0.012-0.014 nats on Porian rw, Porian owt2 and Chinchilla.
But it is far worse than simply dropping the two smallest sizes when predicting the largest third. So the miscounted-
parameter story does not explain the tiny models' pull on the forecast.

On Porian:
- Drop2 cuts the largest-third RMSE to 0.009 / 0.016 nats.
- e1_strict leaves it at 0.057 / 0.071 nats, close to the 1-term fit's 0.059 / 0.086.

**Pre-registered verdict (arm e1_strict): NOT ESTABLISHED.** Conditions (i) and (ii) hold, (iii) fails:

| condition | needed | observed | holds? |
|---|---|---|---|
| (i) ladders with a positive median gain at >= 10x | 3 | 3 of 3 (Pythia has no e1_strict pair at >= 10x) | yes |
| (ii) median ladder gain | >= 0.005 nats | +0.0123 | yes |
| (iii) median RMSE_e1_strict - RMSE_drop2 on the largest third | <= 0.005 nats | +0.049 | **no** |

Secondary arm e1_common (c fixed at E1's shared c = 3.0, which peeks at all sizes): NOT ESTABLISHED. It gains on 2 of
4 ladders, the median gain is -0.002 nats, and it is worse than drop2 by a median of +0.033.

How thin the >= 10x evidence is:
- e1_strict has 3 pairs from 2 folds on each Porian ladder and 5 pairs on Chinchilla.
- Only Chinchilla's interval excludes 0: [+0.003, +0.018].
- On Porian, the cluster-bootstrap intervals over the 2 folds are [-0.047, +0.015] (owt2) and [-0.099, +0.014] (rw).

## Headline table 1: >= 10x extrapolation (pairs where the arm and arm 1 are both defined)
| ladder | arm | pairs (folds) | median abs err arm / one | mean abs err arm / one | median gain [95% fold-bootstrap] | wins/losses/ties |
|---|---|---|---|---|---|---|
| chinchilla | drop1 | 5 (5) | 0.1667 / 0.2341 | 0.1426 / 0.2269 | +0.0674 [-0.0883, +0.3529] | 3/2/0 |
| chinchilla | drop2 | 4 (4) | 0.1624 / 0.1802 | 0.1590 / 0.1887 | +0.0178 [-0.0815, +0.1645] | 2/2/0 |
| chinchilla | lrt | 7 (6) | 0.3030 / 0.3030 | 0.2812 / 0.2890 | +0.0000 [-0.0268, +0.0613] | 1/2/4 |
| chinchilla | **e1_strict** | 5 (5) | 0.2281 / 0.2341 | 0.2169 / 0.2269 | **+0.0116 [+0.0032, +0.0176]** | 5/0/0 |
| chinchilla | e1_common | 7 (6) | 0.3272 / 0.3030 | 0.3037 / 0.2890 | -0.0168 [-0.0205, -0.0041] | 0/7/0 |
| chinchilla | e1_lrt | 5 (5) | 0.2281 / 0.2341 | 0.2282 / 0.2269 | +0.0060 [-0.0416, +0.0176] | 3/2/0 |
| porian-owt2 | drop1 | 3 (2) | 0.4621 / 0.3664 | 0.4185 / 0.3634 | -0.0957 [-0.0998, +0.0341] | 1/2/0 |
| porian-owt2 | drop2 | 1 (1) | 0.2257 / 0.3047 | 0.2257 / 0.3047 | +0.0790 (one pair) | 1/0/0 |
| porian-owt2 | lrt | 5 (3) | 0.4137 / 0.4137 | 0.3947 / 0.3947 | 0 (never fires at k <= 6) | 0/0/5 |
| porian-owt2 | **e1_strict** | 3 (2) | 0.3529 / 0.3664 | 0.3686 / 0.3634 | **+0.0135 [-0.0465, +0.0153]** | 2/1/0 |
| porian-owt2 | e1_common | 5 (3) | 0.4019 / 0.4137 | 0.3767 / 0.3947 | +0.0135 [+0.0092, +0.0409] | 5/0/0 |
| porian-owt2 | e1_lrt | 3 (2) | same as e1_strict | | +0.0135 | 2/1/0 |
| porian-rw | drop1 | 3 (2) | 0.4160 / 0.3282 | 0.3503 / 0.3098 | -0.0878 [-0.0919, +0.0624] | 1/2/0 |
| porian-rw | drop2 | 1 (1) | 0.1372 / 0.2286 | 0.1372 / 0.2286 | +0.0914 (one pair) | 1/0/0 |
| porian-rw | lrt | 5 (3) | 0.3138 / 0.3138 | 0.3034 / 0.3034 | 0 | 0/0/5 |
| porian-rw | **e1_strict** | 3 (2) | 0.3280 / 0.3282 | 0.3335 / 0.3098 | **+0.0123 [-0.0994, +0.0142]** | 2/1/0 |
| porian-rw | e1_common | 5 (3) | 0.3045 / 0.3138 | 0.2864 / 0.3034 | +0.0123 [+0.0076, +0.0416] | 5/0/0 |
| porian-rw | e1_lrt | 3 (2) | same as e1_strict | | +0.0123 | 2/1/0 |
| pythia | lrt | 1 (1) | 0.0856 / 0.0856 | | 0 | 0/0/1 |
| pythia | e1_common | 1 (1) | 0.1096 / 0.0856 | | -0.0240 | 0/1/0 |

Errors are in nats. Pythia reaches >= 10x only at k = 4 (12B / 1B = 11.7x), where e1_strict, drop1 and drop2 are
undefined.

## Headline table 2: largest third
Single hold-out fold, RMSE in nats: fit the smallest n - ceil(n/3) sizes, predict the ceil(n/3) largest.

| ladder (fit / predicted) | one | drop1 | drop2 | lrt | **e1_strict** | e1_common | e1_lrt |
|---|---|---|---|---|---|---|---|
| chinchilla (18 / 10) | 0.0156 | 0.0120 | 0.0121 | 0.0156 | **0.0136** | 0.0190 | 0.0136 |
| porian-owt2 (9 / 5) | 0.0855 | 0.0556 | 0.0159 | 0.0271 | **0.0714** | 0.0675 | 0.0348 |
| porian-rw (9 / 5) | 0.0587 | 0.0333 | 0.0085 | 0.0316 | **0.0570** | 0.0414 | 0.0325 |
| pythia (5 / 3) | 0.0443 | 0.1215 | undefined (3 sizes left) | 0.0443 | **0.0127** | 0.0598 | 0.0127 |

Paired bootstrap over targets (B = 2,000), RMSE_e1_strict minus a baseline:

| ladder | vs one | vs drop2 |
|---|---|---|
| chinchilla | -0.0020 [-0.0041, +0.0001] | +0.0016 [-0.0010, +0.0048] |
| porian-owt2 | -0.0142 [-0.0200, -0.0053] | +0.0555 [+0.0332, +0.0718] |
| porian-rw | -0.0016 [-0.0023, -0.0006] | +0.0485 [+0.0263, +0.0640] |
| pythia | -0.0315 [-0.0427, -0.0148] | - |

Pythia is the one ladder where the count correction helps a lot on the largest third (0.044 -> 0.013), but that is
3 targets from a 5-size fit, and drop2 cannot be compared there.

## Table 3: median |error| (nats) by extrapolation ratio N_pred / N_max_fit (c_std count), as in RESULTS_dm5.md
| ladder | arm | 1-3x | 3-10x | 10-30x |
|---|---|---|---|---|
| chinchilla | one | .0196 | .0701 | .3030 |
| | drop1 / drop2 | .0167 / .0176 | .0677 / .0723 | .1667 / .1624 |
| | e1_strict / e1_common | .0192 / .0210 | .0534 / .0733 | .2281 / .3272 |
| porian-owt2 | one | .0444 | .1706 | .4137 |
| | drop1 / drop2 | .0338 / .0111 | .1365 / .0269 | .4621 / .2257 |
| | e1_strict / e1_common | .0410 / .0390 | .1715 / .1612 | .3529 / .4019 |
| porian-rw | one | .0318 | .1288 | .3138 |
| | drop1 / drop2 | .0191 / .0040 | .0971 / .0305 | .4160 / .1372 |
| | e1_strict / e1_common | .0324 / .0268 | .1623 / .1115 | .3280 / .3045 |
| pythia | one | .0223 | .0518 | .0856 |
| | e1_strict / e1_common | .0076 / .0300 | .0125 / .0698 | - / .1096 |

No pair reaches >= 30x. The arm-1, drop1 and drop2 numbers on Porian and Chinchilla equal DM-5's RESULTS_dm5.md
(same y, same count, same fitter). That serves as a consistency check.

The full table, including lrt and e1_lrt, is in `results/ratio_bins.csv`. The figure is
`results/e3_error_vs_ratio.png`.

## What the fold-wise c does (descriptive)
- Chinchilla: the training-fold c is 0 in all 23 folds.
- Porian: it ranges over 0.77-3.0 (rw, median 2.43) and 1.32-3.0 (owt2, median 3.0), with many folds at the grid
  edge.
- Pythia: 0.71-0.93.
- The LRT-selected 2-term at c_std fires in 5 of 10 Porian folds per dataset and in 7 of 24 Chinchilla folds, all at
  k > 6. So it never fires at >= 10x, as in DM-5.

## Controls (verify.py, ALL PASS)
1. 48 folds and 2,940 rows as expected. No NaN in defined predictions. e1_strict is defined exactly for k >= 5.
2. Positive control: planted c = 0.3 with arm 1 using the wrong count (c = 2). e1_strict gains +0.057 nats median at
   >= 10x (|err| 0.057 vs 0.099, 60 pairs).
3. Null control: planted c = 2 = c_std. e1_strict gains -0.012 (no false win).
4. No peeking: perturbing held-out y (sd 0.5) changes no prediction and no chosen c.

## Post hoc / labelled
- e1_common peeks: its c comes from all sizes of all ladders, so it is not a strict forecast. It is reported
  because the brief asked for it.
- e1_lrt (2-term at the strict c) is the optional arm 5, not judged. On the Porian largest third it is close to the
  c_std LRT arm (0.035 / 0.033 vs 0.027 / 0.032) and still worse than drop2.

## Caveats
- **Few pairs at >= 10x**:
  - Porian: 2 folds (k = 4 is excluded for e1_strict).
  - Chinchilla: 5 pairs, with a usable range of only ~31x after the D = 20N filter.
  - Pythia: only k = 4.
  - Only Porian rw, Porian owt2 and Chinchilla have e1_strict pairs at >= 10x. Porian rw and owt2 share architecture
    and design, so they are not independent.
- The largest-third comparison is one fold per ladder (5-10 targets). Its bootstrap is over targets, not over
  ladders or fits.
- The ratio uses the c_std count for every arm. ColPret LLaMA is excluded (no embedding counts).
- B = 200 for the LRTs and 2,000 for the intervals. Everything ran in seconds on skampere2.

## Files
- Code: `e3.py` -> `summarize_e3.py`, plus `verify.py`. The ladders and c_sh come from
  `../12_e1_profiled_param_count/results/`.
- `results/`: `folds.csv`, `at10x.csv`, `largest_third.csv`, `largest_third_pooled.csv`, `ratio_bins.csv`,
  `verdict.csv`, `e3_error_vs_ratio.png` (grid on).

**Verdict line:** E1 correction (e1_strict) NOT ESTABLISHED. It gains +0.012 nats median at >= 10x on 3 of 3
ladders but is +0.049 nats RMSE worse than Drop2 on the largest third. e1_common is also NOT ESTABLISHED.
