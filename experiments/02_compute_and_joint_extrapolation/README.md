# 02 — Compute and joint \((N, D)\) extrapolation

## Hypothesis

Adding a second, shallower parameter term,

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{N^{\beta}} + \frac{G}{D^{\gamma}}, \qquad \beta < \alpha,
$$

lowers held-out error relative to the additive Chinchilla law

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{G}{D^{\gamma}}
$$

and relative to one-variable baselines \(L(N)\), \(L(D)\), and \(L(C) = E + A C^{-\alpha}\) with \(C = 6ND\). A two-term law in compute, \(L(C) = E + A C^{-\alpha} + B C^{-\beta}\), is the compute analogue of the conjecture. All of these forms are additive. None is the multiplicative form \(A N^{-\alpha} D^{-\beta}\).
On the extra \(N\) term, \(\alpha\) is the fast exponent and \(\beta < \alpha\) is the slow one, matching the experiments index after experiment 06.

In the CSV, Hoffmann's data coefficient and exponent are `param_G` and `param_gamma`.

## Method

Same Huber-on-log-residuals objective as experiment 01 (\(\delta = 10^{-3}\), L-BFGS-B, 12 starts). Three tables:

- Every Chinchilla run in `colpret_chinchilla.csv.gz` (245 rows). \(C/(6ND) = 1\).
- GPT-3 validation medians in the same 10 token windows as experiment 01 (8 sizes).
- Pythia LAMBADA \(\ln(\mathrm{ppl})\) at the same 16 shared checkpoints. This is not pretraining loss.

Two protocols, both fixed before scoring:

1. **Size.** Hold out the largest \(20\%\) of sizes (2 sizes when the ladder has fewer than 15). Every token budget of a held-out size is test. Every test \(N\) is larger than every train \(N\).
2. **Compute.** Hold out runs with \(C\) above the 80th percentile. Ties at the cutoff stay in train, so every test \(C\) is strictly larger. This split still contains sizes that also appear in train whenever a size was run at both low and high compute.

Primary metric: macro RMSE, the mean over test sizes of each size's RMSE, in nats. Micro RMSE (every row equal) is stored in `metrics.csv` and does not change the conclusion about the extra \(N\) term. Token budgets of one model are dependent, so these are not independent-sample standard errors.

A one-variable baseline is only an extrapolation along the axis that was held out. On the size split, test \(D\) mostly sits inside the training \(D\) range (Chinchilla 81% of test rows, GPT-3 and Pythia 100%), so \(L(D)\) is largely interpolating tokens. On the compute split, no test \(C\) is inside the training \(C\) range, but many test \(N\) were seen at lower compute (Chinchilla 80% of test rows, GPT-3 38%, Pythia 100%).

```bash
PYTHONPATH=. python3 experiments/02_compute_and_joint_extrapolation/run_joint_backtest.py
```

## Result

The second parameter term does not improve extrapolation. Wherever it is not collapsed, held-out macro RMSE matches the five-parameter joint law to about 0.002 nats or gets slightly worse. The two-term compute law collapses to the one-term compute law on every split (the extra exponent is unused).

Macro RMSE (nats):

| Split | Model | Chinchilla | GPT-3 val | Pythia LAMBADA |
| --- | --- | ---: | ---: | ---: |
| Size | \(N\) only | 0.358 | 0.168 | 1.858 |
| Size | \(D\) only | 0.229 | 0.375 | 0.921 |
| Size | \(C\) one-term | 0.216 | 0.172 | 1.330 |
| Size | \(C\) two-term | 0.216 (collapsed) | 0.172 (collapsed) | 1.330 (collapsed) |
| Size | joint | **0.127** | **0.045** | **0.633** |
| Size | joint + second \(N\) term | 0.127 | 0.047 | 0.637 |
| Compute | \(N\) only | 0.162 | 0.231 | 0.340 |
| Compute | \(D\) only | 0.153 | 0.344 | 0.886 |
| Compute | \(C\) one-term | **0.028** | 0.089 | 0.151 |
| Compute | \(C\) two-term | 0.028 (collapsed) | 0.089 (collapsed) | 0.151 (collapsed) |
| Compute | joint | 0.035 | **0.032** | **0.101** |
| Compute | joint + second \(N\) term | 0.035 (collapsed) | 0.032 | 0.101 (collapsed) |

On the Chinchilla **compute** split the plain one-term compute law beats the joint law (0.028 vs 0.035). That split is not "predict a larger model": 80% of its test rows use an \(N\) that was already seen at smaller \(C\). On the **size** split, which does hold out larger models, the joint law is clearly better than any one-variable law, and the extra \(N\) term does not move the error (second-term fraction about 0.06 at the median training \(N\)).

The GPT-3 compute split shows a 0.0003 nat edge for the extra term (0.03217 vs 0.03250). The validation-loss windows have an IQR of about 0.01 nats, so that edge is smaller than the binning noise. On the GPT-3 size split the extra term is worse (0.047 vs 0.045).

Pythia joint errors stay large (0.63 nats on the size split). LAMBADA NLL is a poor match to this functional family; the extra term still does not help.

Every fit returned L-BFGS `success=True`. No prediction was non-positive. Recomputing macro RMSE from `outputs/predictions.csv` matches `outputs/metrics.csv` exactly. The size-holdout joint fit on the smaller Chinchilla runs is \(E \approx 1.79\), \(\alpha \approx 0.30\), \(\gamma \approx 0.43\). That is a held-out fit on this table, not a reproduction of the Besiroglu point estimates, and those targets are not applied to GPT-3 or Pythia.

Figures: `outputs/size_holdout_rmse.png`, `outputs/compute_holdout_rmse.png`, `outputs/chinchilla_size_pred_vs_actual.png`.
