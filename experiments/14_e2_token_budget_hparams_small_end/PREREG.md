# E2 PREREG (11-25-2026): is Porian's small-end misfit a token-budget or a hyperparameter confound?

As of 11-25-2026. Written by ub9sc (Claude Code, Opus 5.5 (claude-opus-5-5), CPU on skampere1), committed and pushed
to brando90/dark-matter main ALONE and BEFORE any E2 fit. Before this commit only the data's structure was opened:
which (dataset, hparams, warmup, decay, width) runs exist, their tokens per parameter (D/N), learning rates, batch
sizes, beta2, warmup tokens, and which sweep configurations exist. No loss of the sweep was looked at and no E2 fit
was run. Porian is not blind: DM-2, DM-3 and E1 (experiments/dm2_real_ladders/RESULTS.md,
experiments/12_e1_profiled_param_count/results.md) already report the tuned-short-const 20N excess.

## Dark-matter question
Why do the tiniest models live off the scaling curve, and what correction brings them back? E2 asks whether the
off-curve tiny Porian models are an artefact of how the ladder was trained: (a) the token budget (per-size final
losses come at very different D/N; at fixed D/N the data term is itself a power of N) or (b) the hyperparameters
(base hparams use lr 0.003, batch 256, beta2 0.95 at every size, so a 5M model gets ~200 steps at 20 tokens/param)
and the schedule (every base run is decayed, every tuned run has a constant learning rate).

Law convention (experiments/README.md): L(N) = E + A N^-alpha + B N^-beta with beta < alpha, alpha the FAST exponent.
E2 uses only the 1-term law L = E + A N^-alpha in N and the additive Chinchilla joint law
L(N, D) = E + A N^-alpha + B D^-beta (here beta is the D exponent, as in the Chinchilla form). Never multiplicative.

## Data (Porian et al. 2406.19146, data/data_resolving_discrepancies, 975 runs; loader dm2_real_ladders/data.py)
Datasets rw and owt2, separately. Parameter count (primary): **tied** count N = N_nonemb + V w (V = 50,432) =
DM-2's N_total = Porian's own N, so numbers compare with DM-2/DM-3. Secondary (reported, no verdict): untied total
N_nonemb + 2 V w. D/N always uses the tied N (the same runs and checkpoints under every count).
Small sizes s: **5M (w96), 9M (w160), 15M (w224)** (Porian's names); 7M (w128) is reported but not in any verdict.

Conditions (one loss y per size; all val loss in nats):
| id | runs | y per size | D/N across sizes |
|---|---|---|---|
| TC20 | tuned, short warmup, constant LR | val curve read at D = 20 N (DM-2 `porian_curve_value`, interpolated in log step) | matched (20) |
| TCk | same | at D = k N, k in {5, 10} | matched (k) |
| TCfin | same | last checkpoint | 5 (901M) to 400 (rw small) / 200-274 (owt2) |
| BC20 | base hparams, short warmup, cosine decay ("chinchilla") | annealed finals interpolated in log(D/N) at 20 (DM-2 `interp_log`) | matched (20) |
| BCk | same | same at k in {5, 10} | matched (k) |
| BCfin | same | final of the longest run per size | 5 to 97 |
| BKfin | base hparams, short warmup, "kaplan" decay | final (one run per size) | 5 to 1,266 |
| SW22 | rw only, "sweep": constant LR, short warmup, D/N ~22, lr x batch x beta2 grid at 7 sizes (5M, 7M, 15M, 28M, 57M, 108M, 221M) | final of each configuration | matched (~22) |
Seed noise sd per size and dataset: DM-2 `results/seed_sigma.csv` (7 seeds of the tuned recipe; w96-w160 at 20 tok/param,
w224-w480 at 200). sd at 5M / 9M / 15M: rw 0.062 / 0.005 / 0.005, owt2 0.012 / 0.019 / 0.005.

## Fits
- **Small-end excess** X_s(cond) = y_s - yhat(N_s), yhat the 1-term law fitted (DM-1b `fit_batch` 1-term branch, via
  experiments/12_e1_profiled_param_count/e1lib.fit1; E, A >= 0) on the **reference sizes**: every size of the condition
  with tied N >= 1e8 (108M, 149M, 221M, 347M, 455M, and 612M / 901M where they have a value). This is DM-2/DM-3's
  reverse ladder. Needs >= 4 reference sizes, else not computed (so k = 40 is not used).
- **Joint law** (additive Chinchilla, `dm2_real_ladders/fitlib.fitJ1`, n = N/1e6, d = D/1e9, variable projection):
  fitted on every point of the reference sizes with 5 <= D/N <= 80: TC = the tuned-const curve read at
  D/N in {5, 10, 20, 40, 80} where the run reaches it; BC = the actual annealed finals with D/N in [5, 80].
  Joint excess X^J_s = y_s(TC20 or BC20) - yhat_J(N_s, D = 20 N_s). Secondary: the same joint law fitted on all sizes,
  residual at s at 20N.
- **Joint misfit flag**: if the joint fit's residual sd on its own points (sqrt(RSS/(n - 5))) exceeds 0.02 nats
  (about 4x the 15-57M seed sd), the joint-law factor J is reported but its verdict is "undetermined (joint misfit)",
  and the return-to-curve test uses X_s(TC20) instead of X^J_s.
- **Sweep gaps** (rw): per size, y_best = mean of the 3 lowest final losses over all sweep configurations (the plain
  minimum is reported too; it is biased low by selection). gap_base_s = y(base-like config) - y_best, base-like =
  lr 0.003, beta2 0.95, the largest batch <= 256 present at that size (64 at 5M-15M; 256 at 57M-221M).
  gap_tuned_s = y(tuned recipe read on the TC curve at the sweep's D/N of that size) - y_best. "Large" = mean over
  108M and 221M.

## Factor contributions (nats, at each s; positive = the factor accounts for excess)
| factor | definition | baseline for % |
|---|---|---|
| T: token budget (per-size final -> matched D/N) | X_s(TCfin) - X_s(TC20); X_s(BCfin) - X_s(BC20); secondary X_s(BKfin) - X_s(BC20) | X_s(fin) |
| J: data term at matched D/N | X_s(TC20) - X^J_s(TC); same for BC | X_s(20) |
| R: recipe (base + decay -> tuned + constant, D/N = 20) | X_s(BC20) - X_s(TC20) | X_s(BC20) |
| H: pure hparams (rw; constant LR, D/N 22) | gap_base_s - gap_base,large | X_s(BC20) |
| S: schedule and the rest (rw) | R_s - H_s (cosine vs constant, warmup, batch 64 -> 256, interactions; additivity assumed) | X_s(BC20) |
| G: residual mis-tuning of the tuned recipe (rw) | gap_tuned_s - gap_tuned,large | X_s(TC20) |
Secondary (descriptive, no verdict): X_s(TCk), X_s(BCk) for k = 5, 10, 20, to see whether the excess moves with D/N.
H uses batch 64 at the small sizes where base uses 256, so H UNDER-states the base mis-tuning; this is said in results.

## Bootstrap (95% percentile intervals)
Parametric. Reference-size points: y* = fitted + N(0, sigma_ref^2), sigma_ref = max(fit residual sd, 0.002)
(fit sd = sqrt(RSS/(n - 3)) for the 1-term law, sqrt(RSS/(n - 5)) for the joint law). Each small-size value and each
sweep value: y* = y + N(0, sd_seed^2) with the seed sd of that size and dataset (w640 and w832: the w480 value). One
draw per observed number, shared wherever the same number is used (TC20's y_s in X(TC20) and X^J(TC)). Refit per draw.
B = 2000 for N-only and sweep quantities, B = 500 for joint fits (factor intervals use the first 500 joint draws
paired with the first 500 N-only draws). Seeds fixed in code; partial results cached to disk.

## Decision rules (fixed now)
**"Explained" for a factor F at size s** (C = its contribution, base = its baseline excess):
- EXPLAINS: C >= 50% of base AND C >= 2 sd_seed,s AND the 95% interval of C excludes 0 (lower > 0).
- PARTLY: lower > 0 AND C >= 20% of base, not EXPLAINS.
- NO: otherwise. OPPOSITE: upper < 0 AND |C| >= 20% of base (matching makes the excess larger).
- N/A: base <= 0 (no excess to explain).
Why 20% and 50%: DM-3 (post hoc) found the Kaplan data term reproduces ~15% of the 5M excess (+0.05 of +0.37 on rw,
+0.07 of +0.35 on owt2) and read that as "not the answer"; 20% sits just above it, so an effect of DM-3's size reads NO.
50% is "most of it". 2 sd_seed guards against single-run noise (rw 5M sd is 0.062, so 0.12 nats there).

**Return to the curve** (the headline): matched excess M_s = X^J_s(TC) - G_s on rw, X^J_s(TC) on owt2 (no sweep), with
X_s(TC20) in place of X^J_s if the joint misfit flag is set. RETURNS: |M_s| <= 2 sd_seed,s AND the 95% interval of
M_s contains 0. STAYS OFF: M_s > 2 sd_seed,s AND lower > 0. Otherwise UNDETERMINED.
Per dataset: CONFOUND EXPLAINS the small-end misfit if all three sizes RETURN; CONFOUND DOES NOT EXPLAIN IT if >= 2 of
3 STAY OFF; otherwise PARTIAL. The E2 answer is the pair (rw, owt2); if they differ it is reported as differing.

## Reading of the final table
One row per (dataset, factor, size): C, 95% interval, % of baseline, verdict word. A factor "explains the small-end
misfit" on a dataset only if it is EXPLAINS at >= 2 of the 3 sizes. The token-budget answer is T and J together
(T: is the per-size-final ladder the confound; J: is the remaining 20N excess the data term). The hyperparameter
answer is R, H, S (base vs tuned) and G (is the tuned recipe itself mis-tuned at the small end).

## Reproduction check (must pass before any E2 number is read)
X_s(TC20) at 5M with these reference sizes equals DM-2's reverse-ladder residual (+0.347 rw, +0.371 owt2, RESULTS.md)
within 0.005 nats, and TC20 / BC20 y values equal `dm2_real_ladders/results/curves.csv` within 1e-9.

## Verify (verify.py, must pass)
1. The reproduction check above.
2. Positive control (token budget): a synthetic Porian-shaped ladder (same sizes and D/N points as TC) from a joint
   law E + A N^-alpha + B D^-beta with no second N term, noise sd 0.003: J is EXPLAINS at 5M and M_5M RETURNS in
   >= 80% of 20 replicates.
3. Null control: the same ladder plus a planted fast N term (+0.3 nats at 5M, exponent 1.4) and no change in the data
   term: J is not EXPLAINS and M_5M STAYS OFF in >= 80% of 20 replicates.
4. Arithmetic: S = R - H exactly; T, J, R, G recomputed from the per-condition tables.

## Pre-registered expectations (not rules)
From DM-2/DM-3: X(BC20) at 5M is larger than X(TC20) (DM-2 base reverse means +0.47-0.64), so R > 0; J is small
(DM-3 ~15%); TCfin's reverse residuals are erratic (DM-2) so T may be OPPOSITE or N/A.
