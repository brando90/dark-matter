# 01 — Back-test: single term vs two terms vs baselines

## Hypothesis

On a ladder with tokens roughly matched, the conjectured two-term law

$$
L(N) = E + \frac{A}{N^{\alpha}} + \frac{B}{N^{\beta}}, \qquad \beta < \alpha
$$

has lower extrapolation error than

- the single-term law \(L(N) = E + A N^{-\alpha}\),
- a log correction \(L(N) = E + A N^{-\alpha}\bigl(1 + \gamma \log(N/N_{\mathrm{ref}})\bigr)\),
- a broken power law \(L(N) = E + A N^{-\alpha} / \bigl(1 + (N/N_b)^{\delta}\bigr)\),

when every model is fit only on smaller \(N\) and scored on the largest held-out sizes.
In this fit, \(\alpha\) is the fast exponent (the one that matters at small \(N\)) and \(\beta < \alpha\) is the slow exponent, matching the experiments index after experiment 06.

## Method

Fits minimize the Hoffmann / Besiroglu Huber objective on log residuals (\(\delta = 10^{-3}\)) with L-BFGS-B, 12 starts, then a polish from the best start. \(E, A, B > 0\) and \(\beta = \alpha \cdot \sigma(v)\) so \(\beta < \alpha\) by construction. \(N_{\mathrm{ref}}\) is the geometric mean of the **training** sizes only.

Holdout rule, fixed before looking at errors: if a ladder has fewer than 15 sizes, hold out the 2 largest; otherwise hold out the largest \(20\%\) (rounded). No test size is used in the fit, the initializer, or \(N_{\mathrm{ref}}\).

Ladders:

1. **Chinchilla** (`colpret_chinchilla.csv.gz`). One run per size whose \(D/N\) is closest to 20 and within a factor 1.5. Primary split: 26 train sizes (\(N\) from \(1.1\times 10^8\) to \(1.73\times 10^9\)), 7 test sizes (\(1.79\times 10^9\) to \(4.52\times 10^9\)). Test \(D/N\) sits between 14.4 and 24.5. The 6.8B–16B Chinchilla runs are **not** in this ladder: their \(D/N\) is far below 20. Sensitivities repeat the rule at target ratios 15 and 30.
2. **GPT-3 validation loss** (`colpret_gpt3.csv.gz`), the Kaplan-style curve that is actually in this repo. It is not a digitization of Kaplan et al. 2020 WebText. At each of 10 log-spaced token targets from \(2\times 10^{10}\) to \(2.5\times 10^{11}\), the loss is the median `val_loss` inside a \(\pm 10\%\) token window (at least 5 rows). Within-window IQR is about 0.01 nats. Hold out 13B and 175B (6 train / 2 test). A separate fit is made at each token budget.
3. **Pythia LAMBADA** (`colpret_pythia.csv.gz`). This export has **no** pretraining cross-entropy. The target is \(\ln(\mathrm{ppl})\), LAMBADA mean NLL in nats, at 16 shared checkpoints with \(D \ge 10^9\) and \(\mathrm{ppl} < 10^4\). Hold out 6.9B and 12B (5 train / 2 test).

Primary metric: RMSE of predicted vs observed loss on held-out sizes, in nats. MAE is reported too. For GPT-3 and Pythia the token budgets are **not** independent experiments (same models), so the mean and median across budgets are stability summaries, not standard errors. Degrees of freedom are \(n_{\mathrm{train}} - k\).

Run from the repo root:

```bash
PYTHONPATH=. python3 experiments/01_backtest_single_vs_two_term/run_backtest.py
```

## Result

The two-term law does **not** reliably extrapolate better than a single power plus a floor.

| Ladder | single RMSE | two-term RMSE | log correction | broken power |
| --- | ---: | ---: | ---: | ---: |
| Chinchilla \(D/N \approx 20\) (7 held-out sizes) | 0.0565 | 0.0523 | 0.0565 | **0.0389** |
| Chinchilla, same rule at \(D/N \approx 15\) | 0.0192 | 0.0192 (collapsed) | **0.0148** | 0.0213 |
| Chinchilla, same rule at \(D/N \approx 30\) | 0.0163 | 0.0163 (collapsed) | **0.0125** | 0.0948 |
| GPT-3, mean over 10 token budgets | 0.0250 | 0.0335 | **0.0245** | 0.1266 |
| GPT-3, median over budgets | **0.0198** | 0.0310 | 0.0202 | 0.1365 |
| GPT-3, largest budget only (\(D \sim 2.5\times 10^{11}\)) | 0.0254 | **0.0073** | 0.0264 | 0.1736 |
| Pythia LAMBADA, largest budget | **0.0377** | 0.0377 (collapsed) | 0.0546 | 0.1510 |
| Pythia, mean / median over 16 budgets | 0.106 / 0.057 | 0.137 / 0.103 | **0.104 / 0.054** | 0.117 / 0.069 |

What that table is saying:

- On the pre-specified Chinchilla ladder the two-term RMSE is only 0.004 nats below the single term (MAE 0.0447 vs 0.0487). The training Huber objective barely moves, and both curves over-predict every held-out run. Those runs are not monotone in \(N\) (losses 2.38, 2.33, 2.38, 2.37, 2.29, 2.30, 2.21), so a smooth law cannot get a tiny RMSE. At the two sensitivity ratios the second term dies (one exponent is unused) and the RMSE matches the single term. The broken power law wins only the ratio-20 split and is the worst model at ratio 30.
- On GPT-3 the two-term model wins **1 of 10** budgets, the largest one, where it has 1 degree of freedom (6 training sizes, 5 parameters) and cuts RMSE from 0.025 to 0.007. On the other nine budgets it is tied or worse, and the mean RMSE is higher than the single term (0.033 vs 0.025). That one-budget win is not a stable extrapolation result.
- On Pythia the two-term model has 0 degrees of freedom at each budget. At the final checkpoint it collapses to the single term (RMSE 0.0377 either way). Across budgets its mean RMSE is worse (0.137 vs 0.106). The earliest budget dominates the mean (single-term RMSE 0.72 there); the median is the fairer one-number summary and still favors the single term over the two-term law. This is downstream NLL, not Pile loss.

Every retained fit returned L-BFGS `success=True`. No held-out prediction was non-positive. Train and test sizes are disjoint, and every test \(N\) is larger than every train \(N\).

Figures: `outputs/heldout_rmse.png`, `outputs/chinchilla_ladder.png`, `outputs/gpt3_ladder.png`, `outputs/pythia_ladder.png`. Numbers: `outputs/metrics_summary.csv`, `outputs/metrics_by_slice.csv`, `outputs/predictions.csv`.
