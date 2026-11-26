"""Check that the two-term fit really enforces beta < alpha (beta slow, alpha fast)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from py_src.conjecture_fits import fit_power  # noqa: E402
from py_src.scaling_laws_fit import fit_two_term_N_only  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(0)
    n = np.geomspace(1e8, 1e11, 24)
    # Fast alpha=0.5, slow beta=0.2. Both terms visible near 1e9.
    y = 1.5 + 12649.0 * n ** (-0.5) + 25.24 * n ** (-0.2)
    y = y * np.exp(rng.normal(0.0, 0.01, size=n.shape))
    new = fit_power("two_term", n, y, n_starts=8, seed=0)
    old = fit_two_term_N_only(n, y)
    if not (new.params["beta"] < new.params["alpha"]):
        raise SystemExit(f"conjecture_fits violated beta < alpha: {new.params}")
    if not (old.beta < old.alpha):
        raise SystemExit(f"fit_two_term_N_only violated beta < alpha: {old}")
    print("index convention: alpha fast (larger), beta slow (smaller), beta < alpha")
    print(f"conjecture_fits alpha={new.params['alpha']:.4f} beta={new.params['beta']:.4f}")
    print(f"fit_two_term_N_only alpha={old.alpha:.4f} beta={old.beta:.4f}")
    print("root README letters are swapped: its beta is this alpha, its alpha is this beta")


if __name__ == "__main__":
    main()
