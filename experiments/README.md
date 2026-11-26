# Experiments index (dated 11-25-2026)

Convention (fixed 11-25-2026): the law is L(N) = E + A N^-alpha + B N^-beta where the **slow** term has the smaller exponent and the
**fast ("dark matter") term, the one that matters at small N, has the larger exponent: beta_fast > alpha_slow**. Code that writes beta < alpha
(`fit_two_term_N_only`, Grok's `conjecture_fits.py`) is the same law with the two names swapped. Every result states which term is which.

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

`RESULTS.md` summarizes 01–05. Those five record the protocol in each README; a PREREG file was not committed before the fits. In `conjecture_fits.py`, alpha is the fast exponent and beta < alpha is the slow one (names swapped relative to the sentence above).
