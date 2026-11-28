# Experiments index (dated 11-27-2026)

As of 11-27-2026.

Convention (fixed in experiment 06): the law is \(L(N) = E + A N^{-\alpha} + B N^{-\beta}\) with **\(\beta < \alpha\)**.

- \(\alpha\) is the **fast** exponent (the larger one). \(A N^{-\alpha}\) dominates at small \(N\) and dies first as \(N\) grows.
- \(\beta\) is the **slow** exponent (the smaller one). \(B N^{-\beta}\) dominates the approach to \(E\) at large \(N\).

This is the constraint in `fit_two_term_N_only` and `py_src/conjecture_fits.py` (`beta = alpha * sigmoid(v)`). The root `README.md` and the DM-1 / DM-2 / DM-5 preregistrations use the opposite letters: their \(\beta\) is this index's \(\alpha\) (fast), and their \(\alpha\) is this index's \(\beta\) (slow). Those files are unchanged historical records. New writeups use \(\beta < \alpha\) and say which exponent is fast.

| folder | what | by |
|---|---|---|
| `00_prepare_data_for_scaling_law_fitting` | Porian and ColPret data download, joint fit script | Brando |
| `dm1_power_analysis` | DM-1 and DM-1b: when can a second term be detected (43,200 synthetic replicates; Wald vs bootstrap LRT) | ub9sc |
| `dm2_real_ladders` | DM-2 and DM-3: two-term fits on Porian, Chinchilla, LLaMA; profiled embedding weight c on Porian | ub9sc |
| `dm4_embedding_ladder` | DM-4: own ladder that varies embedding size at fixed body size (design, trainer, G1 gate, phase 1 rows) | ub9sc |
| `dm5_small_to_large` | DM-5: rolling-origin small-to-large extrapolation | ub9sc |
| `01_backtest_single_vs_two_term` | hold-out back-test on Chinchilla, GPT-3, Pythia LAMBADA | Grok |
| `02_compute_and_joint_extrapolation` | size and compute holdouts for the joint law vs a second N term | Grok |
| `03_residual_structure` | in-sample residuals of the single-term and joint fits | Grok |
| `04_fit_stability` | multi-start, bootstrap, Huber-delta sweep on Chinchilla | Grok |
| `05_synthetic_recovery` | recover a known two-term law, plus a false-positive check | Grok |
| `06_naming_convention` | index uses \(\beta < \alpha\), \(\alpha\) fast; maps the root README's swapped letters | Grok |
| `07_power_analysis` | sizes, \(N\) range, and noise needed before held-out RMSE sees the slow term | Grok |
| `08_public_ladders` | Epoch Figure 4, Pythia Pile val, Porian, C4 scaling; same holdout as 01 | Grok |
| `09_alternative_forms` | broken power, log correction, and a second \(D\) term; held-out RMSE and Huber AIC/BIC | Grok |
| `10_tiny_n_residual` | fit on large \(N\), score the residual on the smallest sizes | Grok |
| `11_e5_embedding_ladder_runs` | E5: raw results and PREREG analysis of the DM-4 embedding ladder run on Sherlock (phase 1 queued 11-25) | ub9sc |
| `12_e1_profiled_param_count` | E1: profiled embedding weight c (N_c = N_nonemb + c N_emb) on Porian rw/owt2, Chinchilla, Pythia; one-c and tiny-model-return verdicts (PREREG, verify, results.md) | ub9sc |
| `13_e3_small_to_large_forecast` | E3: DM-5 rolling-origin forecast with the E1 count correction (strict and shared c) vs 1-term, Drop1/2, LRT 2-term | ub9sc |
| `14_e2_token_budget_hparams_small_end` | E2: [scientifically unresolved](14_e2_token_budget_hparams_small_end/results.md); conditional PARTIAL/PARTIAL for matched token budgets and training recipe; numerical analysis complete, software checks pass, preregistered statistical controls fail | ub9sc; initial Claude Code, Opus 5.5; completion Codex CLI, GPT-6 Astra ultra (Vals); Python CPU on skampere1 |
| `15_e4_porian_lrt_both_counts` | E4: [CONVENTION-DEPENDENT](15_e4_porian_lrt_both_counts/results.md), 26/114 bootstrap detections across 38 Porian conventions and three counts; 1,000 refits/test; verification passes | ub9sc; initial Claude Code, Opus 5.5; completion Codex CLI, GPT-6 Astra ultra (Vals); Python CPU on skampere1 |
| `16_two_term_lrt_calibration_chinchilla_backtest_porian_n_count_start_25Nov2026_end_26Nov2026` | CPU tests of the two-term conjecture: [no evidence for or against a second N term; the repo's `lr_test` cannot reject](16_two_term_lrt_calibration_chinchilla_backtest_porian_n_count_start_25Nov2026_end_26Nov2026/results.md); calibration and power of the test on simulated ladders, extra-term backtests on the Chinchilla points with and without the runs under one token per parameter, real fixed-data ladders, and Porian parameter-count definitions | Claude Code, Claude Sonnet 5.5; Python CPU, no cluster |
| `17_bootstrap_lrt_port_parity_calibration_start_27Nov2026_end_27Nov2026` | Port of the DM-1b bootstrap likelihood-ratio test into `lr_test` and a loud `fit_two_term_N_only`: [all five criteria pass, and post hoc the test over-rejects under wide-range multiplicative noise](17_bootstrap_lrt_port_parity_calibration_start_27Nov2026_end_27Nov2026/results.md); 220 of 220 committed DM-1b rows reproduced exactly, 33 of 600 additive-noise null ladders rejected, strong second term found in 40 of 40 | Claude Code, Claude Sonnet 5.5; Python CPU, no cluster |

`RESULTS.md` summarizes the numbered CPU experiments. The early numbered CPU experiments record their procedure in each README without a preregistration committed before the fits. E2 and E4 have separate preregistrations committed and pushed before their first fits.
