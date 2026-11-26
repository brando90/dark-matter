# Results: five CPU checks of the two-term scaling conjecture

The conjecture, in the naming used by `py_src/conjecture_fits.py`, is

$$
L(N) = E + \frac{A}{N^{\alpha}} + \frac{B}{N^{\beta}}, \qquad \beta < \alpha
$$

where \(\alpha\) is the fast exponent (it matters at small \(N\)) and \(\beta\) is the slow one. Experiment 06 makes the experiments index use this same naming. The root `README.md` still writes the fast exponent as \(\beta\). The joint baseline is the additive Chinchilla form \(E + A N^{-\alpha} + G D^{-\gamma}\), never the multiplicative form \(A N^{-\alpha} D^{-\beta}\).

All five runs finished on CPU (Python 3.12.3, numpy / scipy / pandas / matplotlib). The repo pin is `>=3.11,<3.12`; this machine has 3.12.3, so the numbers below are from that interpreter. No SNAP, Sherlock, or FarmShare job was required.

## One line each

1. **Back-test, small \(N\) to large \(N\).** The two-term law does not reliably beat a single power plus a floor. On the pre-specified Chinchilla ladder (\(D/N \approx 20\), 7 held-out sizes) test RMSE is 0.052 vs 0.057; at \(D/N \approx 15\) and \(30\) the second term dies and the RMSE matches the single term. On GPT-3 validation loss the two-term model is better by a meaningful margin on 1 of 10 token budgets (RMSE 0.007 vs 0.025 at the largest budget) and by \(2\times 10^{-5}\) nats on one other; it is worse on the mean (0.033 vs 0.025). On Pythia LAMBADA it is saturated (5 training sizes, 5 parameters) and does not help.
2. **Joint and compute.** A second power of \(N\) on top of the joint law does not reduce held-out error (Chinchilla size-holdout macro RMSE 0.127 either way). A two-term law in \(C = 6ND\) collapses to the one-term compute law on every split.
3. **Residuals.** On 33 Chinchilla sizes at \(D/N \approx 20\), the single-term residual has no trend in \(\log N\) (Spearman \(-0.02\), quadratic \(R^2 = 0.03\)) and the two-term fit does not lower residual RMSE (0.0326 vs 0.0324). The joint RMSE of 0.096 on all 245 runs is four very short runs; the median absolute residual is 0.010.
4. **Identifiability.** All 24 joint starts agree (parameter exponent 0.351–0.353). Bootstrap resampling does not: \(\log A\) correlates with \(\alpha\) at 0.999, and \(G\)'s 10th–90th percentile runs from \(3\times 10^{3}\) to \(4\times 10^{4}\). On the matched ladder the slow term's share of the loss at the median \(N\) runs from 0.14 to 0.89.
5. **Synthetic recovery.** With 36 sizes and 2% log-noise, a two-term truth does not produce a reliable held-out win (mean RMSE difference versus the single term is \(+0.005\), win rate 10/20). Both exponents are recovered noiselessly by the self-check, and a single exponent is recovered at \(\sigma=0.02\) (median \(\alpha = 0.50\)). Two exponents of the same \(N\) are not. A joint \((N, D)\) grid does recover \(\alpha\) (median 0.356 vs 0.35).
6. **Naming.** The experiments index now uses the code constraint \(\beta < \alpha\), with \(\alpha\) fast and \(\beta\) slow. The root README's \(\beta > \alpha\) is the same law with the letters swapped, and that file is left as written.
7. **Power.** For the separated truth from experiment 05, no cell with up to 64 sizes, a \(10^{4}\) range in \(N\), and log-noise down to 0.5% meets the pre-set rule (80% win rate and a 0.01 nat median held-out gap). The best cell wins 16/16 but the median gap is 0.0066 nats. With no noise the same 64-size, \(10^{4}\) ladder has a 0.0165 nat gap, so the signal is real and smaller than 0.5% noise.
8. **Public ladders.** On the Epoch Figure 4 extraction at \(D/N \approx 20\) the two-term fit collapses and the held-out RMSE matches the single term (0.0493). On Pythia final Pile validation loss the single term predicts 6.9B and 12B to 0.006 nats and the second term turns off. On Porian rw and owt2 the two-term power does not help, but a log correction cuts RMSE from about 0.08 to about 0.026 on two held-out sizes. The C4 ladder has only 4 training sizes, so the two-term model is saturated.
9. **Other forms.** Huber AIC/BIC pick the single term on every N-only ladder and the five-parameter joint law on both run clouds, because the training objective drops by \(\sim 10^{-4}\) when a parameter is added. A second power of \(D\) collapses. Broken power wins held-out RMSE only on the ColPret Chinchilla coordinates (0.039 vs 0.057) and is the worst model on the Epoch extraction (0.120 vs 0.049).
10. **Tiny \(N\).** A single-term fit on the large end underpredicts every one of the smallest 20% of Chinchilla sizes (mean residual \(+0.087\) nats) and every one of the smallest 20% of Epoch sizes (\(+0.110\)). A two-term fit on that same large end collapses on Epoch and does not remove the gap. Pythia and Porian change sign when the cutoff moves from 20% to 40%, so the tiny-end miss is not stable there.

## What the negative results are not

They are not an optimizer failure. L-BFGS returned `success=True` on every fit cited above, and the 24 random starts of the joint law land in one basin. They are also not a unit error: on the Chinchilla, GPT-3, and Pythia tables, \(C / (6ND) = 1\).

## Data the repo does not contain

- Kaplan et al. 2020 WebText runs. The Kaplan-style ladder used here is the GPT-3 validation curve in `colpret_gpt3.csv.gz` (Brown et al. 2020).
- Pythia pretraining cross-entropy. `colpret_pythia.csv.gz` has LAMBADA perplexity. The target is \(\ln(\mathrm{ppl})\), labeled as such in experiment 01–03.
- Chinchilla runs at \(D/N \approx 20\) above 4.5B parameters. The 6.8B–16B runs in the table are far below 20 tokens per parameter, so they are not in the matched ladder.
- The published Besiroglu point \((E, A, \alpha, G, \gamma) = (1.82, 482, 0.34, 2085, 0.36)\) is not the minimizer on these 245 runs. Its Huber objective is \(1.19\times 10^{-2}\) against \(1.83\times 10^{-3}\) for the best fit, and optimizing from that start leaves it. Those coefficients were not used as targets for GPT-3 or Pythia.

## Checks done before this summary

- Experiment 01: test RMSE recomputed from `predictions.csv` matches `metrics_by_slice.csv` with maximum absolute difference 0. Train and test sizes are disjoint, and every test \(N\) is larger than every train \(N\).
- Experiment 02: macro RMSE recomputed from `predictions.csv` matches `metrics.csv` with maximum absolute difference 0. The size split has no test \(N\) inside the training range. The compute split has no test \(C\) inside the training range. Many test rows on the compute split do reuse an \(N\) seen at lower compute (80% of Chinchilla test rows, 100% of Pythia); the writeup says so, so a compute-split win is not a large-model prediction.
- Experiments 01, 02, and 05: no non-positive held-out prediction.
- Experiment 03: the \(\log D\) quadratic \(R^2\) of 0.32 on joint residuals falls to 0.075 after dropping two residuals above 0.5. The summary quotes both.
- Experiment 04: bootstrap intervals are stability checks. The runs share a recipe, so they are not confidence intervals.
- Experiment 05: held-out sizes are the largest 7 of 36, fixed before scoring. Paired win rates are computed per replicate, not by comparing marginal medians (those medians differ because a few bad two-term fits have a heavy tail).
- Experiment 09: test RMSE recomputed from `predictions.csv` matches `metrics.csv`. AIC/BIC use the training Huber objective only. The second \(D\) term's slow-fraction of 1.0 is why it is marked collapsed.
- Experiment 10: the small end is strictly smaller than the training sizes. Mean residuals recomputed from `small_end_predictions.csv` match `metrics.csv`. Pythia's 40% split trains on 5 sizes, so the two-term model has 0 degrees of freedom and is marked saturated.

## Cluster

None of these five need a GPU or a larger machine. An embedding-width ladder that would need a cluster already lives in `experiments/dm4_embedding_ladder/` and was not run here.

## Preregistered Porian follow-ups (2026-11-26)

As of 2026-11-26. **Why do the tiniest models live off the scaling curve, and what correction brings them back?** Original preregistrations/fits: ub9sc, Claude Code, Opus 5.5. Completion: ub9sc, Codex CLI, GPT-6 Astra ultra, Vals plan; Python CPU fits on skampere1.

- **[15_e4_porian_lrt_both_counts](15_e4_porian_lrt_both_counts/results.md): CONVENTION-DEPENDENT.** Bootstrap likelihood-ratio detections: non-embedding 1/38, tied 13/38, untied 12/38 (26/114 total); 1,000 completed refits per test. Only four detections also meet the positive small-end reverse-ladder signature. At tuned/short/constant 20 tokens per tied parameter, tied-count RMSE falls from 0.01668 to 0.00489 nats (rw) and 0.02328 to 0.00665 (owt2), with p = 0.000999 each and BIC improvements 29.061/29.798. Verification passes: null 0/40, planted term 40/40. Counts/slices share observations; tuning and schedule are confounded. Full table and caveats are in the linked report.
- **[14_e2_token_budget_hparams_small_end](14_e2_token_budget_hparams_small_end/results.md): SCIENTIFICALLY UNRESOLVED.** Conditional table classification is PARTIAL on rw and owt2; no target size meets the return criterion. At 5M/9M/15M, changing base+cosine to tuned+constant reduces reverse-ladder excess by 0.840/0.757/0.534 nats (rw) and 0.872/0.781/0.557 (owt2); full 95% intervals are in the report. At 5M, corrected matched excess is −0.216 [−1.295, +0.130] nats (rw) and −0.275 [−1.495, +0.158] (owt2); rw's 9M sweep correction is unavailable. Software checks pass and all 2,000 N-only/sweep and 500 joint bootstrap draws are saved, but amended positive controls yield EXPLAINS 3/20 and RETURNS 0/20, and planted-term controls yield not-EXPLAINS 17/20 and STAYS OFF 9/20 (each gate requires both ≥16/20). Full scientific verification fails; no validated confound explanation is claimed. Original failed controls and the prospective repair amendment are retained.

## CPU tests of the two-term test, the Chinchilla extra terms and Porian parameter counts (11-26-2026)

- **[16_two_term_lrt_calibration_chinchilla_backtest_porian_n_count_start_25Nov2026_end_26Nov2026](16_two_term_lrt_calibration_chinchilla_backtest_porian_n_count_start_25Nov2026_end_26Nov2026/results.md): NO EVIDENCE FOR OR AGAINST A SECOND N TERM; THE REPO'S `lr_test` CANNOT REJECT.** On simulated 10-size ladders its largest statistic is 6.40e-05 (0.3% noise) and 1.38e-04 (1% noise) against a 5% cutoff of 5.99; a Gaussian version has 5.3% / 2.7% false positives at 10 sizes but only 60% / 18% power at 7 sizes. On the 245 Chinchilla points a second \(D\) term cuts the Huber objective by 37.5%, but its token-axis held-out gain falls from 79% to 66% without the 4 shortest runs and to 19% without the 11 runs under one token per parameter (post hoc cuts). On Porian the \(N\) definition moves the compute-optimal exponent between 0.520 and 0.752, and a second \(N\) term lowers the fit objective by 10.47 to 10.83% with embeddings counted against 0.03 to 2.25% without. CPU only, not preregistered, Claude Code (Claude Sonnet 5.5); folder 15's bootstrap test points the same way in direction (convention dependent).
