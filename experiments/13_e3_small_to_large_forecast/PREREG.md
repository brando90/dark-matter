# E3 PREREG (11-25-2026): small-to-large forecast with the E1 parameter-count correction

Written 11-25-2026 by ub9sc (Claude Code, Opus 5.5, on skampere2), committed and pushed to brando90/dark-matter main
BEFORE any E3 (and any E1) fit. Extends DM-5 (experiments/dm5_small_to_large, PREREG_dm5.md). DM-5's results are known:
Drop2 helps on Porian, is a wash on Chinchilla, hurts LLaMA; the LRT-selected 2-term never beats 1-term at >= 10x.

## Dark-matter question
If the tiny models are off the curve because parameters are miscounted (E1's N_c = N_nonemb + c N_emb), then fitting
the 1-term law in N_c should forecast LARGER models better than the plain 1-term fit. Does it, out of sample?

Convention: L = E + A N^-alpha + B N^-beta, beta < alpha, alpha FAST (experiments/README.md). 1-term: E + A N^-p.

## Ladders
The four E1 ladders (experiments/12_e1_profiled_param_count/PREREG.md): porian-rw, porian-owt2 (14 sizes), chinchilla
(28), pythia (8). Same y, N_nonemb and N_emb (one V x d block). ColPret LLaMA is excluded (no embedding counts, so the
E1 arms are undefined there; DM-5 already reports its 1-term / Drop / LRT numbers, and it has no pair at >= 10x).
Standard count c_std: porian 2 (true untied total, DM-5's primary), chinchilla 1 (DM-2 N_total), pythia 2 (total).

## Folds (DM-5 rolling origin)
Sort by N in the c_std count. For k = 4 .. n-1: fit on the smallest k sizes, predict every larger
size. Extrapolation ratio r = N_target / N_max_fit in the c_std count (the same r for every arm).

## Arms
1. **one**: 1-term at c_std, all k fit sizes.
2. **drop1 / drop2**: 1-term at c_std after dropping the smallest 1 / 2 fit sizes (defined when k - d >= 4).
3. **lrt**: DM-1b bootstrap-LRT-selected 2-term at c_std (B = 200, 2-term iff (1+j)/201 <= 0.05; defined for k > 6,
   for k <= 6 the rule is the 1-term fit, as DM-5).
4a. **e1_strict**: 1-term in N_c with c profiled on the training fold only (grid c in {0, 0.05, ..., 3.0}, bounded
   refinement between neighbours), prediction at the target's N_c with that c. Defined for k >= 5 (with k = 4 the
   4-parameter fit is exact and c is not identified). No information from the held-out sizes.
4b. **e1_common**: 1-term in N_c with c fixed at E1's shared c (c_sh, the pooled 1-term fit over all four ladders and
   ALL sizes). It has seen the held-out sizes through c_sh: labelled "not a strict forecast", secondary.
5. **e1_lrt** (optional, not in the verdict): DM-1b LRT-selected 2-term in N_c at the e1_strict c (k > 6).

Fitter: DM-1b `fit_batch` (imported from experiments/dm5_small_to_large), as DM-5.

## Metrics
- Errors in nats: err = pred - y, |err|.
- **At >= 10x**: all (fold, target) pairs with r >= 10 where both the arm and arm 1 are defined. Per ladder: median and
  mean |err| per arm; gain = |err_one| - |err_arm| (positive = arm better); ladder-level median gain with a 95% paired
  bootstrap interval resampling folds (cluster bootstrap over k, B = 2,000); win / loss counts over pairs.
- **Largest third**: the single fold k* = n - ceil(n/3) (fit the smallest n - ceil(n/3), predict the ceil(n/3)
  largest). RMSE per arm per ladder; RMSE differences vs arm 1 and vs drop2 with 95% paired bootstrap intervals over
  targets (B = 2,000). Also pooled over all folds: RMSE on targets that are in the largest third.
- Table by ratio bin [1, 3), [3, 10), [10, 30), [30, inf): median |err| per ladder and arm, as RESULTS_dm5.md.

## Verdict rule (fixed now)
The E1 correction (arm 4a, e1_strict) **HELPS** if all three hold:
(i) its ladder-level median gain over arm 1 at r >= 10 is > 0 on at least 3 ladders (of the ladders where both are
defined at r >= 10);
(ii) the median of those ladder-level gains is >= 0.005 nats;
(iii) it is not worse than drop2 on the largest third: the median over ladders (where both are defined at k*) of
RMSE_e1_strict - RMSE_drop2 is <= 0.005 nats.
Otherwise **NOT ESTABLISHED** (with the failing condition named). If fewer than 3 ladders have any pair at r >= 10 for
arm 4a, (i) cannot hold and the verdict is NOT ESTABLISHED (insufficient ladders at >= 10x), stated as such.
The same rule is applied to arm 4b and reported as secondary (it peeks). Arms 2, 3, 5 are reported, not judged.
Expected from the sizes: pythia has one fit size at >= 10x only for k = 4 (12B / 1B), where e1_strict is undefined;
so only Porian rw, Porian owt2 and Chinchilla are likely to have e1_strict pairs at >= 10x.

## verify.py (ALL PASS required before results.md)
1. Fold counts: sum over ladders of (n - 4) folds; every arm row present; no NaN where an arm is defined.
2. Positive control: a synthetic ladder with porian-rw's sizes and counts, planted 1-term law in N_c with c = 0.3
   (rw 1-term fit at c = 0.3), sd 0.005, 20 replicates; arm 1 uses c_std = 2 (the wrong count). e1_strict must beat
   arm 1 at r >= 10: median gain > 0.005 nats over the replicates' pairs.
3. Null control: the same with the planted c = c_std = 2 (arm 1 is right): e1_strict must not pass (i)-(ii) on it,
   i.e. median gain at r >= 10 < 0.005 nats.
4. The e1_strict c is chosen from training sizes only: a check that perturbing the held-out y values leaves every
   e1_strict c and training-fold fit unchanged.

## Compute
CPU only, `nice`, <= 16 processes; ~130 folds x (one LRT with B = 200 + a c profile) is minutes.

## Limitations (now)
Few ladders reach 10x (Porian 5M-455M, Chinchilla's D = 20N range is ~31x, Pythia at k = 4 only). Porian rw / owt2
share architecture and runs design, so they are not independent ladders. A forecast win at one ladder is not a
mechanism.
