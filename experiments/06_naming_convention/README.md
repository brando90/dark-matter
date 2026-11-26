# 06 — One exponent naming in the experiments index

## Hypothesis

The experiments index was stating two conventions at once: \(\beta_{\mathrm{fast}} > \alpha_{\mathrm{slow}}\), and a note that the code swaps those letters. That note made every later result easy to misread. The code constraint is the one the fits actually enforce, so the index should use it.

## Method

Read `fit_two_term_N_only` and `py_src/conjecture_fits.py`. Both set the second exponent to a fraction of the first, so the stored \(\beta\) is strictly smaller than the stored \(\alpha\). \(\alpha\) is therefore the fast exponent (dominates at small \(N\)). \(\beta\) is the slow exponent.

The root `README.md` and the DM-1, DM-2, and DM-5 preregistrations use the opposite letters (their \(\beta\) is the fast one). Those files are historical and were not rewritten. The index now says so in one sentence instead of adopting their letters.

`check_naming.py` fits one noisy two-term ladder with both implementations and exits non-zero if either returns \(\beta \ge \alpha\).

```bash
PYTHONPATH=. python3 experiments/06_naming_convention/check_naming.py
```

## Result

The experiments index now states \(L(N) = E + A N^{-\alpha} + B N^{-\beta}\) with \(\beta < \alpha\), \(\alpha\) fast, \(\beta\) slow. On one noisy ladder whose truth is \(\alpha = 0.50\), \(\beta = 0.20\), `conjecture_fits` returned \(\alpha = 0.419\), \(\beta = 0.315\) and `fit_two_term_N_only` returned \(\alpha = 0.422\), \(\beta = 0.304\). Both satisfy \(\beta < \alpha\). The exponents are not the truth; experiment 05 already showed that one noisy ladder does not identify them. This check only guards the inequality. No parameter of experiments 01–05 was renamed, so those CSVs still match their writeups.
