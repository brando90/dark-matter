# 03 — Residual structure of the single-term fit

## Hypothesis

If a second power of \(N\) is needed, the residual of a single-term fit should still bow when plotted against \(\log N\), and a two-term fit should remove that bow. On the joint table, residuals of

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{G}{D^{\gamma}}
$$

should show no smooth leftover trend in \(\log N\) or \(\log D\).

This experiment is **in-sample**. A lower training error from a model with more parameters is not an extrapolation result. Holdout numbers are in experiments 01 and 02.

Naming in this folder: \(\alpha\) is the fast exponent (larger, matters at small \(N\)) and \(\beta\) is the slow exponent, with \(\beta < \alpha\). The experiments index writes the same law with those names swapped (\(\beta_{\mathrm{fast}} > \alpha_{\mathrm{slow}}\)).

## Method

Huber-on-log-residuals, \(\delta = 10^{-3}\), L-BFGS-B, 12 starts, fit on every point in the curve (no holdout).

- Chinchilla runs with \(D/N\) within a factor 1.5 of 20 (33 sizes). Single-term and two-term.
- All 245 Chinchilla runs. Joint law. Residual vs \(\log N\) and vs \(\log D\), plus an ordinary least-squares regression of the residual on \(\log N\) and \(\log D\).
- GPT-3 validation loss at the largest token window (8 sizes) and Pythia LAMBADA \(\ln(\mathrm{ppl})\) at the last shared checkpoint (7 sizes). Single-term only. Both ladders are too short to support a curvature claim; they are plotted so that is obvious.

Reported summaries: residual RMSE, Spearman rank correlation with the covariate, and the \(R^2\) of a quadratic in that covariate. The Spearman p-value in the CSV assumes independent pairs. These runs share a recipe and a corpus, so that p-value is not a test and is not used below.

```bash
PYTHONPATH=. python3 experiments/03_residual_structure/run_residuals.py
```

## Result

On the matched Chinchilla ladder there is no residual bow for a second power of \(N\) to explain.

| Curve | Fit | n | Residual RMSE | Spearman vs \(\log N\) | Quadratic \(R^2\) in \(\log N\) |
| --- | --- | ---: | ---: | ---: | ---: |
| Chinchilla \(D/N \approx 20\) | single term | 33 | 0.0324 | -0.020 | 0.032 |
| Chinchilla \(D/N \approx 20\) | two-term | 33 | 0.0326 | -0.057 | 0.034 |
| GPT-3, largest \(D\) | single term | 8 | 0.0086 | 0.071 | 0.101 |
| Pythia LAMBADA, largest \(D\) | single term | 7 | 0.0846 | -0.036 | 0.138 |

The two-term fit does find a second exponent (\(\alpha = 0.66\) fast, \(\beta = 0.14\) slow, \(E = 1.14\)) but the residual RMSE does not fall (0.0326 vs 0.0324). The single-term fit on the same 33 points is \(E = 1.70\), \(A = 384\), \(\alpha = 0.30\). Rank correlation of the single-term residual with \(\log N\) is about zero.

The joint fit on all 245 runs (\(E = 1.90\), \(\alpha = 0.35\), \(\gamma = 0.46\), \(A = 529\), \(G = 1.36 \times 10^{4}\)) has median absolute residual **0.010** nats and RMSE **0.096**. The RMSE is not the typical error. Four runs with \(D < 10^{9}\) on models of 2B–7B parameters (tokens per parameter from about 0.04 to 0.3) are under-predicted by 0.28 to 1.08 nats. The two worst residuals are +1.08 (\(N = 2.98 \times 10^{9}\), \(D = 5.16 \times 10^{8}\), loss 4.67) and +0.89 (\(N = 6.8 \times 10^{9}\), \(D = 2.45 \times 10^{8}\), loss 5.01). A quadratic in \(\log D\) explains 32% of residual variance with those points included and 7.5% after dropping the two residuals above 0.5. Spearman correlation of the joint residual with \(\log N\) is 0.029 and with \(\log D\) is 0.010. The leftover structure is those extremely short runs, not a smooth extra power of \(N\) along \(D/N \approx 20\).

GPT-3's endpoint residual RMSE (0.0086) sits inside the ~0.01 nat IQR of the token windows. Pythia's endpoint residual RMSE is 0.085, with one point at +0.21. Neither ladder has enough sizes to separate curvature from noise.

Every fit returned L-BFGS `success=True`. Figures: `outputs/chinchilla_matched_residuals.png`, `outputs/chinchilla_joint_residuals.png`, `outputs/endpoint_residuals.png`.
