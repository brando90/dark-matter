# 05 — Synthetic recovery

## Hypothesis

If the fitter is correct, it should recover a known two-term law

$$
L(N) = E + \frac{A}{N^{\alpha}} + \frac{B}{N^{\beta}}, \qquad \beta < \alpha
$$

and, when that law is the truth, it should predict large \(N\) better than a single term. If the truth is a single term, the extra term should collapse or at least not hurt the held-out error. \(\alpha\) is the fast exponent (matters at small \(N\)); \(\beta\) is the slow one.

A joint control uses `synth_joint` from `py_src/scaling_laws_fit.py`:

$$
L(N, D) = 1.8 + 500\, N^{-0.35} + 2000\, D^{-0.36}
$$

with the same 2% log-normal noise. Here the second exponent multiplies \(D\), not a second power of \(N\).

## Method

\(N\) is 36 log-spaced sizes from \(10^{8}\) to \(10^{11}\). Noise is multiplicative, \(L = \mu \exp(\varepsilon)\), \(\varepsilon \sim \mathcal{N}(0, \sigma^2)\), which matches the Huber-on-log objective. The largest 7 sizes are held out (the same 20% rule as experiment 01). Fits use L-BFGS-B, 8 starts, \(\delta = 10^{-3}\). Twenty replicates at \(\sigma = 0.02\).

Truths:

| Case | Law at the sizes above |
| --- | --- |
| Separated | \(E=1.5\), fast \(A=12649\), \(\alpha=0.50\), slow \(B=25.24\), \(\beta=0.20\). Each power is about 0.40 at \(N=10^{9}\). |
| Close | \(E=1.6\), \(A=1194\), \(\alpha=0.35\), \(B=398\), \(\beta=0.28\). At \(N=10^{9}\) the fast term is about 0.85 and the slow term is about 1.2. |
| Single | \(E=1.5\), \(A=12649\), \(\alpha=0.50\). |
| Log correction | \(E=1.5\), \(A=1990\), \(\alpha=0.40\), \(\gamma=-0.15\), \(N_{\mathrm{ref}}=10^{9}\). |

Parameter recovery fits the matching model on **all** 36 sizes (no holdout), including a quieter separated run at \(\sigma = 0.01\). The joint control is an \(8 \times 8\) grid, 10 replicates, \(\sigma = 0.02\). The noiseless self-check in `py_src/conjecture_fits.py` has to pass before any of this runs.

```bash
PYTHONPATH=. python3 experiments/05_synthetic_recovery/run_recovery.py
```

## Result

The fitter recovers a single power, and it recovers the joint \((N, D)\) exponents on a grid. It does not cleanly recover two powers of the same \(N\), and those two powers do not win the held-out comparison at this noise.

Held-out RMSE, median over 20 replicates (\(\sigma = 0.02\)):

| Truth | Single | Two-term | Log correction | Broken power | Fraction of reps two-term RMSE is lower | Mean (two-term − single) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Separated two-term | 0.0371 | 0.0448 | 0.0385 | 0.0526 | 0.50 | +0.005 |
| Close two-term | **0.0580** | 0.0654 | 0.0648 | 0.119 | 0.40 | +0.016 |
| Single term | **0.0309** | 0.0350 | 0.0316 | 0.0338 | 0.40 | +0.004 |
| Log correction | **0.0315** | 0.0394 | 0.0322 | 0.0388 | 0.35 | +0.008 |

The paired median of (two-term − single) is about 0 in every case, because a collapsed two-term fit copies the single-term error and those replicates sit in the middle of the list. The mean is positive: the active two-term fits have a heavier bad tail. On the separated truth, the 13 replicates that did not collapse have median test RMSE 0.045 against 0.037 for the single term. On the single-term truth, 9 of 20 two-term fits collapse and match the single term; the other 11 have median test RMSE 0.041 against 0.029. Every fit returned `success=True` and every held-out prediction stayed positive.

Parameter recovery on all 36 sizes (median absolute error, and the median fitted exponent):

| Case | \(\sigma\) | Median fitted fast \(\alpha\) | Median fitted slow \(\beta\) | Median \(\lvert\Delta\alpha\rvert\) | Median \(\lvert\Delta\beta\rvert\) | Collapsed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Separated, truth 0.50 / 0.20 | 0.02 | 0.59 | 0.26 | 0.140 | 0.062 | 2/20 |
| Separated, quiet | 0.01 | 0.51 | 0.26 | 0.091 | 0.065 | 2/20 |
| Close, truth 0.35 / 0.28 | 0.02 | 0.61 | 0.27 | 0.257 | 0.042 | 10/20 |
| Single, truth \(\alpha=0.50\) | 0.02 | 0.50 | — | 0.019 | — | — |

\(A\) for the separated law (truth 12649) has a 10th–90th percentile of about \(2.8 \times 10^{3}\) to \(3.4 \times 10^{5}\) at \(\sigma = 0.02\). The slow coefficient is similarly loose (median fitted \(B = 28\) against 25, but the 90th percentile is 147). Quieter noise pulls the fast exponent to 0.51 and still leaves the slow exponent high (median 0.26 against 0.20) with \(B\) median 98 against 25. Close exponents collapse half the time, and when they do not, the fast exponent runs away (median 0.61 against 0.35).

The joint control is the contrast. Median fitted \(\alpha = 0.356\) against 0.35, median \(\gamma = 0.407\) against 0.36, median \(E = 1.83\) against 1.8, all 10 successes. Two exponents are recoverable when they multiply different inputs and the design is a grid. Two exponents of the same \(N\) are not, at a few dozen sizes and 1–2% log noise. The noiseless self-check still recovers both \(N\) exponents, so this is sample size and noise, not a broken optimizer.

Figures: `outputs/synthetic_extrapolation.png`, `outputs/parameter_recovery.png`.
