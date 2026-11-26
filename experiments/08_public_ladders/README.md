# 08 — Back-test on ladders the ColPret export did not give us

## Hypothesis

The two-term law might have failed in experiment 01 because the ladders were the wrong ones: Pythia had no pretraining loss, and the Chinchilla table was not the Epoch figure extraction. The same small-to-large holdout on those public ladders, and on Porian and the mlfoundations C4 runs, should be more informative.

## Method

`fetch_public_ladders.py` writes `data/public_ladders/*.csv.gz`.

| Ladder | What it is | Sizes kept |
| --- | --- | --- |
| Epoch Figure 4 | Besiroglu et al. SVG extraction, 245 runs. \(D = C/(6N)\). One run per \(N\) with \(D/N\) within a factor 1.5 of 20. | 50 |
| Pythia Pile validation | Final `validation/lm_loss` from public W&B histories (open-labs mirror), standard Pile, all at \(2.999\times 10^{11}\) tokens. \(N\) is the paper's nominal size, not a recomputed count. | 8 |
| Porian rw / owt2 | `base` / `chinchilla` / `short` runs in the repo pickle. \(N = 12 \cdot \mathrm{depth} \cdot \mathrm{width}^{2}\) (non-embedding proxy). Same \(D/N \approx 20\) rule. | 13 each |
| C4 scaling | `mlfoundations/scaling` `c4_original` models, non-embedding \(N\), C4 validation loss, same ratio rule. | 6 |

Holdout is the experiment 01 rule: 2 largest sizes when there are fewer than 15, otherwise 20%. Fits are Huber-on-log, 12 starts. \(\alpha\) is fast, \(\beta < \alpha\) is slow.

Pythia intermediate validation losses are not in these sampled histories (most files have a single validation point). The 6.9B and 12B training logs start after step 125,000, so a checkpoint curve that kept every size would only cover the last ~12% of training. This back-test uses the final validation loss only.

OLMo's public files in that mirror are single large runs, not a size ladder. The DCLM model database is downstream accuracy, not next-token loss. Neither is fit here.

```bash
PYTHONPATH=. python3 experiments/08_public_ladders/fetch_public_ladders.py
PYTHONPATH=. python3 experiments/08_public_ladders/run_backtest.py
```

## Result

The second \(N\) term still does not improve held-out RMSE. On the Epoch extraction it collapses. On Pythia Pile validation the single term already predicts the two largest models to about 0.006 nats.

| Ladder | Train / test | Single RMSE | Two-term RMSE | Log correction | Broken power |
| --- | ---: | ---: | ---: | ---: | ---: |
| Epoch \(D/N \approx 20\) | 40 / 10 | 0.0493 | 0.0493 (collapsed) | 0.0491 | 0.120 |
| Pythia Pile val, 300B tokens | 6 / 2 | 0.0059 | 0.0059 (collapsed) | 0.0146 | 0.0169 |
| Porian rw | 11 / 2 | 0.0800 | 0.0798 (collapsed) | **0.0260** | 0.335 |
| Porian owt2 | 11 / 2 | 0.0773 | 0.0780 | **0.0255** | 0.318 |
| C4 scaling | 4 / 2 | 0.296 | 0.296 (saturated, collapsed) | 0.299 | 0.533 |

Pythia details: held-out 6.9B loss 1.777 vs single-term prediction 1.784, and 12B loss 1.741 vs 1.745. The two-term fit's second-term fraction is 0.003. With only 6 training sizes the two-term model has 1 degree of freedom; it used that freedom to turn the extra term off.

Porian is the one place a different functional form helps, and it is not the two-term power. The log correction cuts RMSE from about 0.08 to about 0.026 on both rw and owt2. Both held-out sizes move the same way: the single term predicts a loss about 0.08 too low, and the log correction lands within 0.04. That is two datasets but only two test sizes each, and \(N\) is the \(12 d_{\mathrm{model}}^{2} n_{\mathrm{layer}}\) proxy. It is a real, repeated gap on this grid, not a license to claim the log correction is the missing term in general. The broken power law is much worse on every ladder here.

Every fit returned L-BFGS `success=True`. Test RMSE recomputed from `outputs/predictions.csv` matches `outputs/metrics.csv`. Figure: `outputs/heldout_rmse.png`.
