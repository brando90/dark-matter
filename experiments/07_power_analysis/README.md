# 07 — Power: when held-out RMSE can see the slow term

## Hypothesis

Experiment 05 could not show a held-out gain for \(B N^{-\beta}\) at 36 sizes, a \(10^{3}\) range, and 2% log-noise. This grid asks how many sizes, how wide an \(N\) range, and how small a noise level are required before that gain is detectable.

## Method

Truth is the separated law from experiment 05, fixed before the grid:

$$
L(N) = 1.5 + 12649.07\, N^{-0.50} + 25.24\, N^{-0.20}.
$$

\(\alpha = 0.50\) is fast, \(\beta = 0.20\) is slow. At \(N = 10^{9}\) each power is about 0.40. \(N\) runs from \(10^{8}\) to \(10^{8} \times 10^{s}\) for \(s \in \{2, 3, 4\}\) (ranges of \(10^{2}\), \(10^{3}\), \(10^{4}\)) with \(n \in \{8, 16, 24, 36, 48, 64\}\) log-spaced sizes. Log-noise \(\sigma \in \{0.005, 0.01, 0.02, 0.05\}\). Sixteen replicates per cell. The largest 20% of sizes are held out (2 sizes when \(n < 15\)). Fits are Huber-on-log, \(\delta = 10^{-3}\), L-BFGS-B, 6 starts, training rows only.

A cell is **detectable** only if every replicate succeeds with positive predictions, the two-term test RMSE is lower in at least 80% of replicates, and the median paired gap (single RMSE minus two-term RMSE) is at least 0.01 nats. That rule is a constant in `run_power.py`. It was not changed after seeing the grid.

A noiseless pass on the same truth, same split, and \(n \in \{16, 36, 64\}\) is saved in `outputs/noiseless_ceiling.csv`. It is the gap the noise has to preserve. It is not one of the detection cells.

```bash
PYTHONPATH=. python3 experiments/07_power_analysis/run_power.py
```

## Result

No noisy cell is detectable. The slow term's held-out signature, for this truth, is a few hundredths of a nat even with no noise, and 0.5% log-noise already cuts it below the 0.01 nat rule.

Noiseless held-out RMSE (single, two-term):

| \(N\) range | 16 sizes | 36 sizes | 64 sizes |
| --- | ---: | ---: | ---: |
| \(10^{2}\) | 0.0056, 0.0051 | 0.0052, 0.0043 | 0.0051, 0.00004 |
| \(10^{3}\) | 0.0120, 0.0006 | 0.0114, 0.0006 | 0.0112, 0.0004 |
| \(10^{4}\) | 0.0182, 0.0004 | 0.0171, 0.0003 | 0.0168, 0.0003 |

The noiseless gap crosses 0.01 nats once the ladder spans \(10^{3}\) or more (about 0.011 nats at \(10^{3}\), about 0.016 nats at \(10^{4}\)). A \(100\times\) ladder never reaches a 0.01 nat noiseless gap, even at 64 sizes.

With noise, the best cell is 64 sizes, a \(10^{4}\) range, and \(\sigma = 0.005\): the two-term fit wins 16/16 replicates, but the median gap is only 0.0066 nats. Every other cell has a smaller median gap. At the experiment 05 setting (36 sizes, \(10^{3}\), \(\sigma = 0.02\)) the win rate is 0.50 and the median gap is \(-6 \times 10^{-6}\) nats. At \(\sigma = 0.05\) the median gap is negative or negligible across the grid, and the mean gap is often negative because a minority of two-term fits extrapolate badly.

So, inside this grid, there is no count of sizes that makes \(B N^{-\beta}\) detectable in held-out RMSE at 1% log-noise or above. The noiseless \(10^{4}\) ladder shows the signal is real and about 0.017 nats. Detecting it at the pre-set 0.01 nat bar needs noise below the 0.5% level tried here, or a slow term with a larger coefficient than this truth. Figure: `outputs/win_rate_grid.png`.
