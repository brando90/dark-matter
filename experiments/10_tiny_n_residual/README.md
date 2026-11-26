# 10 — Is the missing term at the small end?

## Hypothesis

A fast term \(A N^{-\alpha}\) is large at small \(N\) and almost gone at large \(N\). A single-term fit that only sees the large sizes should then underpredict loss on the sizes it never saw: residual \(= L - \hat L > 0\). A two-term fit on those same large sizes should not remove the residual if the fast term is invisible there.

## Method

Ladders: Chinchilla and Epoch at \(D/N \approx 20\), Pythia final Pile validation loss, Porian rw with \(N = 12 \cdot \mathrm{depth} \cdot \mathrm{width}^{2}\).

Two cutoffs, fixed before scoring. The small end is the smallest 20% of sizes, and separately the smallest 40%, at least 2 sizes and with at least 4 sizes left to train on. The fit uses only the large side. \(\alpha\) is fast and \(\beta < \alpha\) is slow. Huber-on-log, 12 starts.

```bash
PYTHONPATH=. python3 experiments/10_tiny_n_residual/run_tiny.py
```

## Result

On the Chinchilla and Epoch ladders the smallest 20% are all underpredicted by a single-term fit trained on the rest. The two-term law does not explain that gap, because on the large end it usually collapses to one power.

Mean residual on the small end (nats). Positive means the large-end fit predicted a loss that is too low.

| Ladder | Smallest 20% single / two-term | Smallest 40% single / two-term |
| --- | ---: | ---: |
| Chinchilla \(D/N \approx 20\) | +0.087 / +0.056 | +0.046 / +0.078 (two-term collapsed) |
| Epoch \(D/N \approx 20\) | +0.110 / +0.109 (collapsed) | +0.039 / +0.062 (collapsed) |
| Pythia Pile val | +0.196 / +0.196 (collapsed) | −0.269 / −0.325 (two-term saturated) |
| Porian rw | −0.563 / −1.213 | +0.823 / +0.822 (collapsed) |

Every one of the 7 smallest Chinchilla sizes and all 10 smallest Epoch sizes has a positive residual at the 20% cutoff (Chinchilla +0.032 to +0.148, Epoch +0.079 to +0.172). Widening the small end to 40% drops that to about 70% positive and cuts the mean roughly in half, so the miss is concentrated at the very small end, not spread evenly.

The two-term fit, given only the large sizes, collapses on Epoch at both cutoffs and on Chinchilla at 40%. The one Chinchilla split where it stays active (smallest 20%) lowers the mean residual from 0.087 to 0.056 and does not remove it. A fast term that the large end cannot see is consistent with that. It is not identified by fitting the large end.

Pythia and Porian change sign when the cutoff moves from 20% to 40%. Pythia has 8 sizes, so the 40% split trains on 5 and the two-term model has 0 degrees of freedom. Those two ladders do not support a stable tiny-\(N\) term.

Every fit returned `success=True`. No small-end prediction was non-positive. Mean residuals recomputed from `outputs/small_end_predictions.csv` match `outputs/metrics.csv`. Figure: `outputs/small_end_residual.png`.
