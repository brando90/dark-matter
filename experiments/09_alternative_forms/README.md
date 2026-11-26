# 09 — Other dark-matter forms, held-out RMSE and AIC/BIC

## Hypothesis

If the missing term is not a second power of \(N\), a broken power law, a log correction, or a second power of \(D\) might still lower held-out error. Training-set AIC/BIC should agree with that ranking when the extra parameters earn their keep.

## Method

\(\alpha\) is the fast \(N\) exponent and \(\beta < \alpha\) is the slow one. On the joint law the token coefficient is \(G\) and the token exponent is \(\gamma\). The new form is

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{G}{D^{\gamma}} + \frac{H}{D^{\delta}}, \qquad \delta < \gamma
$$

with \(\gamma\) the fast token exponent and \(\delta\) the slow one. It is still additive.

N-only ladders, one row per size, largest sizes held out by the experiment 01 rule:

- Chinchilla \(D/N \approx 20\) (26 / 7)
- Epoch Figure 4, same ratio rule (40 / 10)
- Pythia final Pile validation loss (6 / 2)
- Porian rw, \(N = 12 \cdot \mathrm{depth} \cdot \mathrm{width}^{2}\) (11 / 2)

Joint clouds, same size holdout, every run: the ColPret Chinchilla table (203 / 42) and the Epoch extraction (194 / 51).

Fits are Huber-on-log, \(\delta = 10^{-3}\), 12 starts, training rows only. Pseudo-AIC and pseudo-BIC use that training objective as \(-\ell\), up to a constant that does not depend on the model:

$$
\mathrm{AIC} = 2k + 2\,\mathrm{Huber}, \qquad \mathrm{BIC} = k \ln n + 2\,\mathrm{Huber}.
$$

Lower is better. They are not Gaussian information criteria. The Huber sum on these ladders is about \(10^{-4}\) to \(10^{-3}\), so the \(2k\) term dominates unless a new parameter cuts the objective by order 1.

```bash
PYTHONPATH=. python3 experiments/09_alternative_forms/run_forms.py
```

## Result

No extra term pays for itself on the training objective. Held-out RMSE moves only in two places, and they disagree with each other.

Held-out RMSE (nats). The AIC winner on every N-only ladder is the single term (AIC 6.00 versus 8 or 10).

| Ladder | Single | Two-term | Log correction | Broken power |
| --- | ---: | ---: | ---: | ---: |
| Chinchilla \(D/N \approx 20\) | 0.0565 | 0.0523 | 0.0565 | **0.0389** |
| Epoch \(D/N \approx 20\) | 0.0493 | 0.0493 (collapsed) | 0.0491 | 0.120 |
| Pythia Pile val | **0.0059** | 0.0059 (collapsed) | 0.0146 | 0.0169 |
| Porian rw | 0.0800 | 0.0798 (collapsed) | **0.0260** | 0.335 |

The broken-power win is the ColPret Chinchilla coordinate set only. On the Epoch extraction of the same figure the broken power law is the worst model (0.120 versus 0.049), and its training Huber objective is not lower by enough to move AIC (10.0008 versus 6.0009). The log-correction win is the two Porian sizes only: the single term predicts both losses about 0.08 too low, and the log correction lands closer, but AIC still prefers the single term because the training objective drops by \(1.2 \times 10^{-4}\).

Joint size-holdout RMSE, micro over runs:

| Cloud | Joint | + second \(N\) term | + second \(D\) term |
| --- | ---: | ---: | ---: |
| Chinchilla runs | **0.2501** | 0.2501 (slow-term fraction 0.06) | 0.2501 (collapsed) |
| Epoch runs | **0.2421** | 0.2422 (fraction 0.06) | 0.2421 (collapsed) |

AIC is 10.00 for the five-parameter joint law and 14.00 for either seven-parameter extension. The second token power collapses (the slow fraction is 1, so one \(D\) exponent is unused). The second parameter power is small and does not change held-out error.

Every fit returned `success=True`. No held-out prediction was non-positive. Test RMSE recomputed from `outputs/predictions.csv` matches `outputs/metrics.csv`. Figure: `outputs/heldout_rmse.png`.
