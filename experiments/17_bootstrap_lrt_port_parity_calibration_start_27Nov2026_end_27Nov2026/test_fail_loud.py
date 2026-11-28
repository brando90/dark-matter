"""Criterion 5 of the folder 17 README: the fitter and the test fail loudly and never use other data.

    python -m pytest test_fail_loud.py -q        or        python test_fail_loud.py
"""
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "dm1_power_analysis"))

from py_src import scaling_laws_fit as S  # noqa: E402
from py_src import two_term_lrt as T  # noqa: E402
import dm1  # noqa: E402


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError(f"{fn.__name__}{a!r} did not raise {exc.__name__}")


def _ladder(n=12, seed=0, sigma=0.005):
    N = np.geomspace(1e7, 1e10, n)
    y = 1.7 + 120 * N ** -0.30 + np.random.default_rng(seed).normal(0, sigma, n)
    return N, y


def test_fit_empty_selection_raises_insufficient():
    e = _raises(S.InsufficientDataError, S.fit_two_term_N_only, np.array([]), np.array([]))
    assert "refusing to fall back" in str(e)


def test_fit_too_few_points_raises_insufficient():
    N, y = _ladder(5)
    _raises(S.InsufficientDataError, S.fit_two_term_N_only, N, y)


def test_fit_too_few_distinct_sizes_raises_insufficient():
    N = np.repeat(np.geomspace(1e7, 1e9, 3), 4)
    _raises(S.InsufficientDataError, S.fit_two_term_N_only, N, 1.7 + 120 * N ** -0.3)


def test_fit_nonfinite_raises_value_error():
    N, y = _ladder()
    y[3] = np.nan
    _raises(ValueError, S.fit_two_term_N_only, N, y)


def test_fit_nonpositive_raises_value_error():
    N, y = _ladder()
    y[2] = 0.0
    _raises(ValueError, S.fit_two_term_N_only, N, y)


def test_fit_shape_mismatch_raises_value_error():
    N, y = _ladder()
    _raises(ValueError, S.fit_two_term_N_only, N, y[:-1])


def test_fit_keeps_convention_and_records_convergence():
    N, y = _ladder(24, sigma=0.002)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p = S.fit_two_term_N_only(N, y)
    assert p.beta < p.alpha, "alpha must be the fast (larger) exponent"
    assert isinstance(p.converged, bool) and np.isfinite(p.objective)


def test_lr_test_too_few_points_raises_not_no_detection():
    N, y = _ladder(6)
    _raises(S.InsufficientDataError, S.lr_test, N, y)


def test_lr_test_empty_and_nonfinite_raise():
    _raises(S.InsufficientDataError, S.lr_test, np.array([]), np.array([]))
    N, y = _ladder()
    y[0] = np.inf
    _raises(ValueError, S.lr_test, N, y)


def test_lr_test_rejects_b_too_small_for_level():
    N, y = _ladder()
    _raises(ValueError, S.lr_test, N, y, B=10, level=0.05)


def test_lr_test_old_chi_square_call_raises_type_error():
    e = _raises(TypeError, S.lr_test, -1.0, 3, -2.0, 5)
    assert "lr_test(N, L)" in str(e)


def test_lr_test_increasing_loss_is_a_fit_error():
    N = np.geomspace(1e7, 1e10, 12)
    _raises(S.FitError, S.lr_test, N, 1.0 + 0.1 * np.log(N / 1e7))


def test_old_chi_square_function_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        S.lr_test_chi2_uncalibrated(-1.0, 3, -1.0000001, 5)
    assert any("not calibrated" in str(x.message) for x in w)


def test_lr_test_is_deterministic_and_early_stop_keeps_the_decision():
    N, y = _ladder(13, seed=3, sigma=0.01)
    a = S.lr_test(N, y, B=60, rng=5)
    b = S.lr_test(N, y, B=60, rng=5)
    assert (a.T, a.k_exceed, a.p_value, a.detect) == (b.T, b.k_exceed, b.p_value, b.detect)
    c = S.lr_test(N, y, B=60, rng=5, early_stop=True)
    assert c.detect == a.detect


def test_two_term_fit_object_predicts_like_dm1b_and_uses_repo_letters():
    cell = next(c for c in dm1.cells() if c["s"] == 1.0 and c["ppd"] == 8 and c["nmin"] == 1e5 and c["nmax"] == 1e9
                and c["sigma"] == 0.005 and c["beta"] == 1.0)
    N = dm1.ladder(cell["nmin"], cell["nmax"], cell["ppd"])
    n = N / dm1.N_REF
    y = dm1.law(n, dm1.E0, dm1.A_SCALED, dm1.ALPHA0, cell["s"] * dm1.A_SCALED, cell["beta"]) + \
        np.random.default_rng(11).normal(0, cell["sigma"], len(n))
    one, two = T.fit_one_and_two_term(N, y)
    f1, f2 = (dict((k, float(v[0])) for k, v in f.items()) for f in T.fit_chunked(n, y[None, :]))
    want = dm1.law(n, f2["E"], f2["a"], f2["alpha"], f2["b"], f2["beta"])
    got = S.predict_two_term(two, N)
    assert np.allclose(got, want, rtol=1e-9), "raw-unit conversion changed the prediction"
    assert not two.collapsed and two.beta < two.alpha, "alpha is fast, beta slow"
    assert abs(two.alpha - f2["beta"]) < 1e-12 and abs(two.beta - f2["alpha"]) < 1e-12, "DM-1b letters are swapped"
    assert two.rss <= one.rss


def test_smoke_script_runs_and_fits_a_real_ladder():
    r = subprocess.run([sys.executable, str(REPO / "py_src" / "scaling_laws_fit.py")], capture_output=True, text=True,
                       timeout=300, cwd=str(REPO))
    assert r.returncode == 0, r.stderr[-400:]
    assert "N-ladder: 20 sizes" in r.stdout and "Selected 0 points" not in r.stdout
    assert "Using all data" not in r.stdout


def test_naming_script_still_passes():
    r = subprocess.run([sys.executable, str(REPO / "experiments" / "06_naming_convention" / "check_naming.py")],
                       capture_output=True, text=True, timeout=300, cwd=str(REPO))
    assert r.returncode == 0, r.stderr[-400:]


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
