"""E4 verification, 2026-11-26: reproduction, cache integrity and 40 null/positive controls.

Run with CPU thread limits: nice -n 10 python verify.py --workers 3.
Controls use the preregistered 1,000 bootstrap refits and save each 100 refits.
"""
import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import e4

L = e4.L
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def reproduction():
    curves = pd.read_csv(Path(L.DM2) / "results/curves.csv")
    old_lrt = pd.read_csv(Path(L.E1) / "results/e1_lrt_by_c.csv")
    grid = pd.read_csv(RESULTS / "e4_cells.csv")
    checks = []
    for ds, r5 in (("rw", .347), ("owt2", .371)):
        d = L.ladder(ds, "TS", 20)
        old = curves[(curves.curve == f"porian/{ds}/tuned-short-const") & (curves["slice"] == "D=20N")]
        old = old.dropna(subset=["y"]).sort_values("N_total")
        assert len(d) == len(old) == 14
        np.testing.assert_allclose(d.y, old.y, atol=1e-14, rtol=0)
        np.testing.assert_array_equal(d.N_tied, old.N_total)
        np.testing.assert_array_equal(d.N_nonemb, old.N_nonemb)
        for count, c in L.COUNTS.items():
            N = L.count(c, d.N_nonemb, d.width)
            np.testing.assert_array_equal(L.count(1, d.N_nonemb, d.width), d.N_tied)
            np.testing.assert_array_equal(L.count(2, d.N_nonemb, d.width) - d.N_tied, L.V * d.width)
            f1, f2 = L.e1lib.fit2(N / L.N_REF, d.y.to_numpy())
            T = float(L.dm1b.lrt_stat(len(d), f1["rss"], f2["rss"]))
            prev = old_lrt[(old_lrt.ladder == f"porian-{ds}") & (old_lrt.c == c)
                           & (pd.to_numeric(old_lrt["at"], errors="coerce") == c)]
            assert len(prev) == 1
            prev = prev.iloc[0]
            np.testing.assert_allclose(T, prev["T"], rtol=1e-6, atol=1e-9)
            current = grid[(grid.dataset == ds) & (grid.recipe == "TS") & (grid["slice"].astype(str) == "20") & (grid["count"] == count)].iloc[0]
            np.testing.assert_allclose(current["T"], T, rtol=1e-10, atol=1e-12)
            assert int(current.detect) == int(prev.detect) == int(c > 0)
            if count == "tied":
                residual, _ = L.reverse(N, d.y.to_numpy(), d.N_tied.to_numpy() >= L.REF_MIN)
                actual_r5 = float(residual[d.width.to_numpy() == 96][0])
                assert abs(actual_r5 - r5) < .005
            checks.append(dict(dataset=ds, count=count, T=T, T_E1=float(prev["T"]), p_E4=float(current.p_lrt),
                               p_E1=float(prev.p_lrt), detect=int(current.detect), rev_5M=float(current.rev_5M)))
    pd.DataFrame(checks).to_csv(RESULTS / "reproduction.csv", index=False)
    print("PASS reproduction: 6 TS 20N count/dataset tests; 14 losses equal DM-2; T/detection equal E1; tied 5M residuals equal DM-2 within 0.005 nats", flush=True)


def integrity():
    grid = pd.read_csv(RESULTS / "e4_cells.csv")
    expected = {(ds, rc, str(sl), count) for ds, rc, sl in L.conventions() for count in L.COUNTS}
    actual = set(zip(grid.dataset, grid.recipe, grid["slice"].astype(str), grid["count"]))
    assert actual == expected and len(grid) == len(expected) == 114
    assert grid.testable.eq(1).all() and grid.npts.gt(6).all()
    assert grid.B.eq(1000).all() and grid.B_done.eq(1000).all()
    assert grid.boot_nonfinite.eq(0).all()
    np.testing.assert_allclose(grid.p_lrt, (1 + grid.k_exceed) / (grid.B + 1), atol=1e-15, rtol=0)
    np.testing.assert_array_equal(grid.detect, (grid.p_lrt <= .05).astype(int))
    assert (grid.rss2 <= grid.rss1 + 1e-14).all()
    np.testing.assert_allclose(grid.rmse1, np.sqrt(grid.rss1 / grid.npts), atol=1e-14, rtol=1e-12)
    np.testing.assert_allclose(grid.rmse2, np.sqrt(grid.rss2 / grid.npts), atol=1e-14, rtol=1e-12)
    np.testing.assert_allclose(grid.dbic, grid.bic1 - grid.bic2, atol=1e-12, rtol=1e-12)
    assert (grid.alpha_fast > grid.beta_slow).all()
    for row in grid.itertuples(index=False):
        sl = row.slice if row.slice == "final" else int(row.slice)
        d = L.ladder(row.dataset, row.recipe, sl)
        N = L.count(L.COUNTS[row.count], d.N_nonemb, d.width)
        assert row.cache_key == e4.cache_key(N / L.N_REF, d.y.to_numpy(), 1000, e4.cell_id(row.dataset, row.recipe, sl, row.count))
        f1 = row.E1 + row.A1 * (N / L.N_REF) ** -row.alpha1
        f2 = row.E2 + row.A_fast * (N / L.N_REF) ** -row.alpha_fast + row.B_slow * (N / L.N_REF) ** -row.beta_slow
        np.testing.assert_allclose(np.sum((d.y - f1) ** 2), row.rss1, rtol=1e-7, atol=1e-12)
        np.testing.assert_allclose(np.sum((d.y - f2) ** 2), row.rss2, rtol=1e-7, atol=1e-12)
    print("PASS integrity: exact 114-row grid; all 114,000 refits complete; cache input/config hashes, losses, p-values, RMSE, BIC, nested RSS and exponent ordering checked", flush=True)


def control(job):
    kind, rep = job
    d = L.ladder("rw", "TS", 20)
    n, y = d.N_tied.to_numpy() / L.N_REF, d.y.to_numpy()
    base, _ = L.e1lib.fit2(n, y)
    mu = base["E"] + base["a"] * n ** -base["alpha"]
    # Paired synthetic noise isolates the planted term; independent bootstrap streams.
    y_control = mu + np.random.default_rng([20261126, rep, 42]).normal(0, .005, len(n))
    if kind == "positive":
        y_control = y_control + .3 * (n.min() / n) ** 1.4
    tag = f"{kind}_{rep:02d}"
    key = e4.cache_key(n, y_control, e4.B_BOOT, tag)
    path = RESULTS / "controls" / f"{tag}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("cache_key") == key and saved.get("B_done") == e4.B_BOOT:
            return saved
    rng = np.random.default_rng([20261126, rep, 43, int(kind == "positive")])
    _, _, T, p, k, nbad = e4.lrt_full(n, y_control, rng, e4.B_BOOT, str(path) + ".progress.json", key)
    row = dict(kind=kind, rep=rep, T=T, p_lrt=p, detect=int(p <= .05), k_exceed=k, B=e4.B_BOOT,
               B_done=e4.B_BOOT, boot_nonfinite=nbad, cache_key=key)
    e4.atomic_json(str(path), row)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--reproduction-only", action="store_true")
    args = ap.parse_args()
    reproduction()  # Must pass before the grid is read.
    if args.reproduction_only:
        return
    integrity()
    jobs = [(kind, rep) for kind in ("null", "positive") for rep in range(40)]
    rows = []
    with Pool(max(1, min(args.workers, 3))) as pool:
        for row in pool.imap_unordered(control, jobs):
            rows.append(row)
            if len(rows) % 10 == 0:
                print(f"Controls complete: {len(rows)}/80", flush=True)
    out = pd.DataFrame(rows).sort_values(["kind", "rep"])
    out.to_csv(RESULTS / "controls.csv", index=False)
    null = int(out[out.kind == "null"].detect.sum())
    positive = int(out[out.kind == "positive"].detect.sum())
    assert out.B_done.eq(1000).all() and out.boot_nonfinite.eq(0).all()
    print(f"Null control: {null}/40 detections (required <=6/40); positive control: {positive}/40 detections (required >=32/40)", flush=True)
    assert null <= 6, "preregistered null control failed"
    assert positive >= 32, "preregistered positive control failed"
    print("PASS E4 verification", flush=True)


if __name__ == "__main__":
    main()
