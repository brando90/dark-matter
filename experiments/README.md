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
| `02_e1_profiled_param_count` | E1: N = N_nonemb + c N_emb on every ladder where it is defined | ub9sc |
| `03_e2_token_budget_hparams_small_end` | E2: token budget and hyperparameters at the small end of Porian | ub9sc |
| `04_e3_small_to_large_forecast` | E3: all sizes vs drop-smallest vs two-term vs E1 correction | ub9sc |
| `05_e4_porian_lrt_both_counts` | E4: Porian re-test with the DM-1b bootstrap LRT and every count convention | ub9sc |
| `06_e5_embedding_ladder_runs` | E5: running and summarizing the DM-4 ladder (code stays in `dm4_embedding_ladder`) | ub9sc |

Each numbered folder has PREREG.md (committed before the first fit), the code, a verify script, results.md naming harness and model.
