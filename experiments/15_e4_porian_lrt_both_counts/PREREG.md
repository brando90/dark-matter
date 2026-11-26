# E4 PREREG (11-25-2026): does a second N term show up on Porian under every convention and every parameter count?

As of 11-25-2026. Written by ub9sc (Claude Code, Opus 5.5 (claude-opus-5-5), CPU on skampere1), committed and pushed
to brando90/dark-matter main ALONE and BEFORE any E4 fit. Before this commit only the data's structure was opened
(which runs exist, their D/N per size). Not blind: DM-2, DM-3 and E1 already tested tuned-short-const at D = 20N
(E1's bootstrap LRT detects a second term there at c = 1 and c = 2, not at c = 0).

## Dark-matter question
Why do the tiniest models live off the scaling curve, and what correction brings them back? The two-term law
L(N) = E + A N^-alpha + B N^-beta (beta < alpha; alpha is the FAST exponent, the term that matters at small N) is one
candidate. E4 asks whether its second term is a property of the Porian ladder or of one way of reading it: does the
DM-1b test detect it under every training convention, D/N slice and parameter count in the data?

## Ladders (one val loss y per size, nats; dm2_real_ladders/data.py loaders, 975 runs)
Datasets rw and owt2. A **convention** = (dataset, recipe, slice):
| recipe | runs | slices |
|---|---|---|
| TS: tuned, short warmup, constant LR | one run per size | D/N = k in {5, 10, 20, 40, 80}: val curve read at D = k N (`porian_curve_value`); final (last checkpoint) |
| TL: tuned, long warmup, constant LR | one run per size (owt2 has 13 sizes) | same 6 slices |
| BC: base hparams, short warmup, cosine ("chinchilla") | 6-7 annealed runs per size | k in {5, 10, 20, 40}: annealed finals interpolated in log(D/N) (`interp_log`, DM-2); final (longest run) |
| BSK / BLK: base hparams, short / long warmup, "kaplan" decay | one run per size | final only (intermediate checkpoints are mid-schedule) |
19 conventions per dataset, 38 in all. A size enters a slice only if it has a value there (no extrapolation).
D/N always uses the tied count, so every count sees the same runs and checkpoints.

**Counts** (V = 50,432, w = width; `porian_counts`): untied total N_nonemb + 2 V w (input embedding and output head;
E1's c = 2); N_nonemb (c = 0); tied N_nonemb + V w (c = 1; = DM-2 N_total = Porian's own N). 38 x 3 = **114 cells**.
Note: a count changes N only, so cells of one convention share y.

## Test (DM-1b bootstrap LRT, experiments/dm1_power_analysis/dm1b.py, unchanged)
Per cell: n = N / 1e6; 1-term and 2-term fits by `dm1b.fit_chunked` (E, A, B >= 0; 2-term RSS <= 1-term RSS);
T = npts ln(RSS1 / RSS2). Parametric bootstrap under the fitted 1-term null: y* = 1-term fit + N(0, RSS1/(npts - 3)),
**B = 1000** refits (all B run, no early stop, so p is exact to 1/1001), seeds fixed per cell. p = (1 + #{T* >= T})/(B + 1);
a refit with no feasible solution counts as an exceedance. **Detect iff p <= 0.05.** Testable iff npts > 6 (DM-1); a
cell with T = 0 has p = 1. Also reported: dBIC = BIC1 (k = 3) - BIC2 (k = 5); the 2-term parameters in the repo
convention (alpha fast); the fast term's share of the slow term at the smallest size.

**Reverse-ladder residual** per cell (companion column, not part of the detect verdict): 1-term law fitted on the
sizes with tied N >= 1e8 present in the cell (>= 4 needed, else n/a); residual y - yhat at 5M, 7M, 9M, 15M, mean over
all sizes < 1e8 and the count positive. Signature: POSITIVE if the 5M residual > 0, >= 75% of the sub-1e8 sizes are
positive and their mean > +0.02 nats; NEGATIVE the mirror; else FLAT.

## Reading of the table (fixed now)
- Per count (its 38 conventions, testable only): **YES** if >= 2/3 detect; **NO** if <= 10% detect (the 0.05 level,
  doubled for correlated tests); **MIXED** otherwise.
- Overall answer to "under every convention and every count?": **UNIVERSAL** iff all three counts read YES and >= 90%
  of testable cells detect. **ABSENT** iff all three counts read NO. Otherwise **CONVENTION-DEPENDENT**, reported with
  the detection rate split by count, dataset, recipe (tuned / base), schedule (constant / decayed) and slice (fixed D/N
  vs final), so the reader sees which factor carries it.
- A detection is read as "dark-matter shaped" only when the reverse-ladder signature of that cell is POSITIVE; the
  number of such cells is reported per count.

## Reproduction check (must pass before the grid is read)
TS, k = 20, all three counts: y equals `dm2_real_ladders/results/curves.csv`; T equals E1's
`12_e1_profiled_param_count/results/e1_lrt_by_c.csv` at c = 0, 1, 2 within 1e-6 relative (same fitter, same data), and
the detection agrees with E1 (rw and owt2: c = 0 no; c = 1 yes; c = 2 yes). Also the tied-count reverse-ladder 5M
residual equals DM-2's +0.347 (rw) / +0.371 (owt2) within 0.005. If any fails, E4 stops and the report says why.

## Verify (verify.py, must pass)
1. The reproduction check.
2. Null control: 40 synthetic 1-term ladders on Porian's 14 sizes at 20N (tied N), parameters = the TS rw k = 20
   1-term fit, noise sd 0.005: false-detection rate <= 0.15 (binomial: 0.05 x 40 = 2 expected, <= 6 allowed).
3. Positive control: the same plus a planted fast term worth +0.3 nats at 5M (alpha 1.4): detected in >= 0.8.
4. Counts: tied = DM-2 N_total exactly; untied - tied = V w; nonemb = DM-2 N_nonemb.

## Pre-registered expectations (not rules)
From DM-2: N_nonemb gives B = 0 on Porian (no detection); base-hparam 20N curves showed no term; final slices gave
B = 0. So the expected answer is CONVENTION-DEPENDENT, with detections concentrated in tuned constant-LR slices at
tied / untied counts.
