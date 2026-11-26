# 04 — Identifiability: multi-start, bootstrap, Huber delta

## Hypothesis

The additive joint law and the two-term \(N\) law are only useful if repeated optimization and resampling return the same exponents. In particular, a second \(N\) term is identified only if its slow exponent and its share of the loss stay in a narrow range.

Naming: \(\alpha\) on the two-term \(N\) law is the fast exponent and \(\beta < \alpha\) is the slow one. On the joint law the token coefficient is \(G\) and the token exponent is \(\gamma\) (Hoffmann's \(B\) and \(\beta\)).

## Method

All fits use L-BFGS-B on the Huber log-residual objective.

- **Multi-start.** 24 starts on all 245 Chinchilla runs. Record every start. "Near best" means objective within 1% of the best.
- **Published point.** Besiroglu-style coefficients \(E=1.82\), \(A=482\), \(\alpha=0.34\), \(G=2085\), \(\gamma=0.36\), scored on this table with the same Huber penalty at \(\delta=10^{-3}\), then used as a single optimizer start. These targets are not applied to any non-Chinchilla set. This table is the ColPret compilation, not a claim that it is the figure set those coefficients were fit on.
- **Bootstrap.** 30 resamples of the 245 runs with replacement, 4 starts each. Same for the 33-point \(D/N \approx 20\) ladder under the two-term law. Runs share a training recipe, so these percentiles are a stability check, not confidence intervals.
- **Huber delta.** Refit the joint law at \(\delta \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 10\}\). The default used everywhere else is \(10^{-3}\). Larger \(\delta\) moves the penalty toward squared error and lets the short, high-loss runs from experiment 03 count more.

```bash
PYTHONPATH=. python3 experiments/04_fit_stability/run_stability.py
```

## Result

The optimizer is not the unstable piece. The parameters along the prefactor–exponent ridge are.

All 24 joint starts finished with L-BFGS `success=True` and an objective within 1% of the best. The parameter exponent only moves from 0.351 to 0.353 across those starts. The best fit is \(E = 1.897\), \(A = 529\), \(\alpha = 0.353\), \(G = 1.358 \times 10^{4}\), \(\gamma = 0.456\), objective \(1.828 \times 10^{-3}\).

The published coefficient vector has objective \(1.190 \times 10^{-2}\) on this table, about 6.5 times the best objective. Starting L-BFGS at that vector moves to the same best fit (objective \(1.828 \times 10^{-3}\), \(\gamma = 0.456\)). This file does not reproduce the published point.

Row-resample percentiles for the joint law (30/30 successes):

| Parameter | Median | 10th | 90th |
| --- | ---: | ---: | ---: |
| \(E\) | 1.88 | 1.83 | 1.94 |
| \(\alpha\) | 0.350 | 0.325 | 0.370 |
| \(A\) | 494 | 330 | 713 |
| \(\gamma\) | 0.467 | 0.389 | 0.512 |
| \(G\) | \(1.75 \times 10^{4}\) | \(3.3 \times 10^{3}\) | \(4.4 \times 10^{4}\) |

\(\log A\) and \(\alpha\) correlate at 0.999, and \(\log G\) and \(\gamma\) at 0.999 (30 pairs). Exponents move less than prefactors; \(G\) still swings by an order of magnitude. The published \((G, \gamma) = (2085, 0.36)\) sits at the low-\(G\) edge of that cloud and is a worse objective on the full table.

Huber delta does not move \(\alpha\) or \(\gamma\) much between \(10^{-4}\) and \(10^{-2}\) (\(\alpha\) stays 0.353–0.356, \(\gamma\) stays 0.453–0.477). At \(\delta = 0.1\) and \(\delta = 10\), where the two very short runs from experiment 03 are no longer downweighted, \(\gamma\) rises to 0.596 and 0.612 and \(G\) rises above \(10^{5}\). The token exponent on this table depends on how those runs are weighted.

On the matched ladder the two-term decomposition is not stable. Across 30 resamples the fast exponent's 10th–90th percentile is 0.21–0.44 and the slow exponent's is 0.09–0.25. The slow term's share of \(L - E\) at the median training \(N\) has median 0.56 and 10th–90th percentile 0.14–0.89. Only 1 of 30 resamples collapsed to a single term. Both terms are usually "on", and their split is not pinned down. That matches experiment 03, where adding the second term did not reduce residual RMSE.

Figures: `outputs/joint_bootstrap_exponents.png`, `outputs/two_term_bootstrap.png`.
