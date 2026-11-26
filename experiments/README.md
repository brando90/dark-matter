# Experiments index (dated 11-25-2026)

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
| `11_e5_embedding_ladder_runs` | E5: raw results and PREREG analysis of the DM-4 embedding ladder run on Sherlock (phase 1 queued 11-25) | ub9sc |
| `12_e1_profiled_param_count` | E1: profiled embedding weight c (N_c = N_nonemb + c N_emb) on Porian rw/owt2, Chinchilla, Pythia; one-c and tiny-model-return verdicts (PREREG, verify, results.md) | ub9sc |
| `13_e3_small_to_large_forecast` | E3: DM-5 rolling-origin forecast with the E1 count correction (strict and shared c) vs 1-term, Drop1/2, LRT 2-term | ub9sc |

`RESULTS.md` summarizes the numbered CPU experiments. They record the protocol in each README; a PREREG file was not committed before the fits.
