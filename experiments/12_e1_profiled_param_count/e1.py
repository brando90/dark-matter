"""E1 (PREREG.md, 11-25-2026): profile the embedding weight c in N_c = N_nonemb + c N_emb on every ladder, 1-term and
2-term laws, DM-1b bootstrap LRT at each c, smallest-size residual test, pooled shared-c test.

    python e1_data.py && python repro_dm3.py && python e1.py --workers 4     # writes results/e1_*.csv, e1_run.json
"""
import argparse
import json
import math
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import e1lib as L

SEED = 20261125
B_RESID = 500
B_FREE = 200
B_POOL = 200
OUT = os.path.join(L.HERE, "results")
INFORMATIVE_MAX_LEN = 2.7


def load():
    lad = pd.read_csv(os.path.join(OUT, "ladders.csv"))
    return {k: g.sort_values("N_std").reset_index(drop=True) for k, g in lad.groupby("ladder", sort=False)}


def smallest_resid(c, ne, nemb, y, sigma2, eps):
    """r_min = y_1 - yhat_1 (1-term at c, all sizes) and the leave-smallest-out error, each with its parametric
    bootstrap sd (noise sd sigma2, the 2-term fit's residual sd; eps (B, n) standard normals)."""
    n = L.n_of(c, ne, nemb)
    f = L.fit1(n, y)
    mu = L.pred1(f, n)
    r_min = float(y[0] - mu[0])
    Ys = mu[None, :] + sigma2 * eps
    fb = L.fit1_vec(n, Ys)
    rb = Ys[:, 0] - (fb["E"] + fb["a"] * n[0] ** -fb["p"])
    flo = L.fit1(n[1:], y[1:])
    e_lo = float(y[0] - L.pred1(flo, n[0]))
    fl = L.fit1_vec(n[1:], Ys[:, 1:])
    eb = Ys[:, 0] - (fl["E"] + fl["a"] * n[0] ** -fl["p"])
    sd, sd_lo = float(np.std(rb, ddof=1)), float(np.std(eb, ddof=1))
    return dict(c=float(c), r_min=r_min, sd_boot=sd, z=r_min / sd, within=int(abs(r_min) <= 2 * sd),
                e_leaveout=e_lo, sd_leaveout=sd_lo, z_leaveout=e_lo / sd_lo, sigma2=float(sigma2),
                p1=f["p"], E1=f["E"], nboot=len(eps))


def job(args):
    lad, g, idx = args
    t0 = time.time()
    ne, nemb, y = g.N_nonemb.values, g.N_emb.values, g.y.values
    npts, c_std = len(y), float(g.c_std.iloc[0])
    # profiles on the 0.01 grid
    prof = []
    for c in L.GRID_FINE:
        n = L.n_of(c, ne, nemb)
        f1, f2 = L.fit2(n, y)
        t = L.two_term_named(f2)
        prof.append(dict(ladder=lad, c=c, rss1=f1["rss"], E1=f1["E"], A1=f1["a"], p1=f1["alpha"], rss2=f2["rss"],
                         E2=t["E"], A_fast=t["A_fast"], alpha_fast=t["alpha_fast"], B_slow=t["B_slow"],
                         beta_slow=t["beta_slow"]))
    P = pd.DataFrame(prof)
    c1, rss1 = L.refine(lambda c: L.fit1(L.n_of(c, ne, nemb), y)["rss"], L.GRID_FINE, P.rss1.values)
    c2, rss2 = L.refine(lambda c: L.fit2(L.n_of(c, ne, nemb), y)[1]["rss"], L.GRID_FINE, P.rss2.values)
    rss1, rss2 = min(rss1, P.rss1.min()), min(rss2, P.rss2.min())
    s1 = L.profile_set(L.GRID_FINE, P.rss1.values, rss1, npts)
    s2 = L.profile_set(L.GRID_FINE, P.rss2.values, rss2, npts)
    P["in_set1"], P["in_set2"] = s1["mask"].astype(int), s2["mask"].astype(int)
    P["stat1"], P["stat2"] = s1["stat"], s2["stat"]
    f1c1 = L.fit1(L.n_of(c1, ne, nemb), y)
    f2c2 = L.two_term_named(L.fit2(L.n_of(c2, ne, nemb), y)[1])
    summ = dict(ladder=lad, npts=npts, c_std=c_std, N_std_min=float(g.N_std.min()), N_std_max=float(g.N_std.max()),
                c1=c1, rss1_min=rss1, sigma1=math.sqrt(rss1 / (npts - 4)), set1=s1["text"], set1_len=s1["length"],
                set1_excl1=int(s1["excludes1"]), set1_excl0=int(s1["excludes0"]),
                informative=int(s1["length"] < INFORMATIVE_MAX_LEN),
                E1_c1=f1c1["E"], A1_c1=f1c1["a"], p1_c1=f1c1["p"],
                c2=c2, rss2_min=rss2, sigma2=math.sqrt(rss2 / (npts - 6)), set2=s2["text"], set2_len=s2["length"],
                set2_excl1=int(s2["excludes1"]), set2_excl0=int(s2["excludes0"]),
                E2_c2=f2c2["E"], A_fast_c2=f2c2["A_fast"], alpha_fast_c2=f2c2["alpha_fast"],
                B_slow_c2=f2c2["B_slow"], beta_slow_c2=f2c2["beta_slow"],
                dbic_free=L.dm1b.lrt_stat(npts, rss1, rss2) - 2 * math.log(npts),
                rss1_c1_over_c_std=rss1 / L.fit1(L.n_of(c_std, ne, nemb), y)["rss"])
    for c in L.C_EXTRA:
        summ[f"secondary_rss1_c{c:g}"] = L.fit1(L.n_of(c, ne, nemb), y)["rss"]
    # bootstrap LRT at each c (DM-1b rule)
    lr = []
    for i, (tag, c) in enumerate([(f"{c:.1f}", c) for c in L.GRID_01] + [("c1", c1), ("c2", c2), ("c_std", c_std)]):
        r = L.lrt(L.n_of(c, ne, nemb), y, [SEED, 1, idx, i])
        lr.append(dict(ladder=lad, at=tag, c=float(c), T=r["T"], p_lrt=r["p"], detect=r["detect"],
                       tested=r["tested"], boot_nonfinite=r["nbad"]))
    LR = pd.DataFrame(lr)
    # c-free LRT: parametric bootstrap from the 1-term fit at c1, c re-profiled (0.05 grid) for both laws
    rng = np.random.default_rng([SEED, 2, idx])
    mu = L.pred1(f1c1, L.n_of(c1, ne, nemb))
    sig = math.sqrt(rss1 / (npts - 3))
    Ys = mu[None, :] + sig * rng.normal(0, 1, (B_FREE, npts))
    R1, R2 = np.full((len(L.GRID_005), B_FREE), np.inf), np.full((len(L.GRID_005), B_FREE), np.inf)
    for j, c in enumerate(L.GRID_005):
        g1, g2 = L.dm1b.fit_chunked(L.n_of(c, ne, nemb), Ys)
        R1[j], R2[j] = g1["rss"], g2["rss"]
    Tb = L.dm1b.lrt_stat(npts, R1.min(0), R2.min(0))
    T_free = float(L.dm1b.lrt_stat(npts, rss1, rss2))
    bad = ~np.isfinite(Tb)
    kex = int(((Tb >= T_free - 1e-9 * max(T_free, 1.0)) | bad).sum())
    summ.update(T_free=T_free, p_free=(1 + kex) / (B_FREE + 1), detect_free=int((1 + kex) / (B_FREE + 1) <= 0.05),
                free_boot=B_FREE, free_boot_nonfinite=int(bad.sum()))
    # smallest-size residual on the 0.05 grid
    eps = np.random.default_rng([SEED, 3, idx]).normal(0, 1, (B_RESID, npts))
    sm = []
    for c in L.GRID_005:
        k = int(np.argmin(np.abs(L.GRID_FINE - c)))
        sm.append(dict(ladder=lad, **smallest_resid(c, ne, nemb, y, math.sqrt(P.rss2.values[k] / (npts - 5)), eps)))
    SM = pd.DataFrame(sm)
    for tag, c in (("0", 0.0), ("c_std", c_std)):
        row = SM.iloc[int(np.argmin(np.abs(SM.c.values - c)))]
        summ[f"r_min_at_{tag}"], summ[f"sd_min_at_{tag}"] = row.r_min, row.sd_boot
        summ[f"excess_survives_at_{tag}"] = int(row.r_min > 2 * row.sd_boot)
    # residual by size
    rs = []
    fits = [("1term", f"c={c:g}", c) for c in sorted({0.0, 1.0, c_std})] + [("1term", "c1", c1)]
    for law, tag, c in fits:
        n = L.n_of(c, ne, nemb)
        f = L.fit1(n, y)
        res = y - L.pred1(f, n)
        rs += [dict(ladder=lad, law=law, at=tag, c=c, name=g.name.values[j], N_std=g.N_std.values[j],
                    N_c=n[j] * L.N_REF, resid=res[j]) for j in range(npts)]
    n = L.n_of(c2, ne, nemb)
    res = y - L.pred2(L.fit2(n, y)[1], n)
    rs += [dict(ladder=lad, law="2term", at="c2", c=c2, name=g.name.values[j], N_std=g.N_std.values[j],
                N_c=n[j] * L.N_REF, resid=res[j]) for j in range(npts)]
    summ["seconds"] = time.time() - t0
    return dict(lad=lad, prof=P, summ=summ, lrt=LR, small=SM, resid=pd.DataFrame(rs), eps=eps)


def pooled(lads, results):
    """Shared c vs per-ladder c for the 1-term law (and, secondary, the 2-term law and the Porian pair)."""
    out = {}
    keys = list(lads)

    def shared(keys_, col, fitfun, ks):
        Ls = {k: len(lads[k]) * np.log(results[k]["prof"][col].values / len(lads[k])) for k in keys_}
        S = sum(Ls.values())

        def tot(c):
            return sum(len(lads[k]) * math.log(fitfun(k, c) / len(lads[k])) for k in keys_)
        c_sh, s_min = L.refine(tot, L.GRID_FINE, S)
        per = sum(len(lads[k]) * math.log(results[k]["summ"][ks] / len(lads[k])) for k in keys_)
        lam = s_min - per
        ntot = sum(len(lads[k]) for k in keys_)
        return dict(c_sh=c_sh, Lambda=lam, dbic_shared=lam - (len(keys_) - 1) * math.log(ntot), K=len(keys_),
                    ntot=ntot)

    def f1rss(k, c):
        g = lads[k]
        return L.fit1(L.n_of(c, g.N_nonemb.values, g.N_emb.values), g.y.values)["rss"]

    def f2rss(k, c):
        g = lads[k]
        return L.fit2(L.n_of(c, g.N_nonemb.values, g.N_emb.values), g.y.values)[1]["rss"]
    main = shared(keys, "rss1", f1rss, "rss1_min")
    out["primary_1term_all4"] = main
    out["secondary_2term_all4"] = shared(keys, "rss2", f2rss, "rss2_min")
    pk = [k for k in keys if k.startswith("porian")]
    out["secondary_1term_porian_pair"] = shared(pk, "rss1", f1rss, "rss1_min")
    out["secondary_1term_without_pythia"] = shared([k for k in keys if k != "pythia"], "rss1", f1rss, "rss1_min")
    # bootstrap p for the primary test (0.05 grid, no refinement, observed Lambda on the same grid)
    c_sh = main["c_sh"]
    gi = [int(np.argmin(np.abs(L.GRID_FINE - c))) for c in L.GRID_005]
    lk_obs = {k: len(lads[k]) * np.log(results[k]["prof"].rss1.values[gi] / len(lads[k])) for k in keys}
    lam_grid = sum(lk_obs.values()).min() - sum(v.min() for v in lk_obs.values())
    lstar = {}
    for i, k in enumerate(keys):
        g = lads[k]
        ne, nemb, y = g.N_nonemb.values, g.N_emb.values, g.y.values
        n = L.n_of(c_sh, ne, nemb)
        f = L.fit1(n, y)
        sig = math.sqrt(f["rss"] / (len(y) - 3))
        Ys = L.pred1(f, n)[None, :] + sig * np.random.default_rng([SEED, 4, i]).normal(0, 1, (B_POOL, len(y)))
        lstar[k] = np.stack([len(y) * np.log(L.fit1_vec(L.n_of(c, ne, nemb), Ys)["rss"] / len(y))
                             for c in L.GRID_005])  # (grid, B)
    lam_b = sum(lstar.values()).min(0) - sum(v.min(0) for v in lstar.values())
    main.update(Lambda_grid005=float(lam_grid), p_boot=float((1 + (lam_b >= lam_grid - 1e-9).sum()) / (B_POOL + 1)),
                nboot=B_POOL, Lambda_boot_median=float(np.median(lam_b)))
    return out, lam_b


def verdicts(lads, results, pool):
    summ = {k: results[k]["summ"] for k in lads}
    info = [k for k in lads if summ[k]["informative"]]
    mask = np.ones(len(L.GRID_FINE), bool)
    for k in info:
        mask &= results[k]["prof"].in_set1.values.astype(bool)
    common = L.GRID_FINE[mask]
    main = pool["primary_1term_all4"]
    if len(info) < 2:
        v1 = "UNDETERMINED"
    elif len(common) and main["dbic_shared"] <= 6:
        v1 = "SUPPORTED"
    else:
        v1 = "NOT SUPPORTED"
    v = dict(informative=",".join(info), n_informative=len(info), common_set_n=int(len(common)),
             common_set_min=float(common.min()) if len(common) else np.nan,
             common_set_max=float(common.max()) if len(common) else np.nan, V1=v1,
             dbic_shared=main["dbic_shared"], c_sh=main["c_sh"])
    # V2
    if len(common) and v1 != "UNDETERMINED":
        cand = [c for c in L.GRID_005 if np.isclose(common, c).any()]
        if np.isclose(common, round(main["c_sh"], 2)).any():
            cand.append(main["c_sh"])
        ok = []
        for c in cand:
            w = [int(results[k]["small"].set_index("c").within.get(round(c, 3), -1)) if c in L.GRID_005 else
                 results[k]["sh"]["within"] for k in lads]
            if all(x == 1 for x in w):
                ok.append(c)
        v["V2_candidates"] = len(cand)
        v["V2_c_where_all_within"] = ";".join(f"{c:.3f}" for c in ok)
        v["V2"] = "YES" if ok else "NOT BY EMBEDDING ACCOUNTING ALONE"
    else:
        v["V2"] = "UNDETERMINED (no common c)"
        v["V2_candidates"] = 0
        v["V2_c_where_all_within"] = ""
    for k in lads:  # each ladder over its own 95% set (reported either way)
        S = results[k]["small"]
        own = S[[bool(results[k]["prof"].in_set1.values[int(np.argmin(np.abs(L.GRID_FINE - c)))]) for c in S.c]]
        v[f"{k}_own_set_c_within"] = ";".join(f"{c:.2f}" for c in own.c[own.within == 1])
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    lads = load()
    t0 = time.time()
    with Pool(min(a.workers, 16)) as pool:
        res = pool.map(job, [(k, g, i) for i, (k, g) in enumerate(lads.items())], chunksize=1)
    results = {r["lad"]: r for r in res}
    pool_out, lam_b = pooled(lads, results)
    c_sh = pool_out["primary_1term_all4"]["c_sh"]
    sh_rows, rs = [], []
    for k, g in lads.items():
        ne, nemb, y = g.N_nonemb.values, g.N_emb.values, g.y.values
        P = results[k]["prof"]
        rss2 = L.fit2(L.n_of(c_sh, ne, nemb), y)[1]["rss"]
        results[k]["sh"] = smallest_resid(c_sh, ne, nemb, y, math.sqrt(rss2 / (len(y) - 5)), results[k]["eps"])
        sh_rows.append(dict(ladder=k, **results[k]["sh"]))
        n = L.n_of(c_sh, ne, nemb)
        res_ = y - L.pred1(L.fit1(n, y), n)
        rs += [dict(ladder=k, law="1term", at="c_sh", c=c_sh, name=g.name.values[j], N_std=g.N_std.values[j],
                    N_c=n[j] * L.N_REF, resid=res_[j]) for j in range(len(y))]
        results[k]["summ"]["stat1_at_c_sh"] = float(len(y) * math.log(L.fit1(n, y)["rss"] /
                                                                       results[k]["summ"]["rss1_min"]))
    v = verdicts(lads, results, pool_out)
    pd.concat([r["prof"] for r in res]).to_csv(os.path.join(OUT, "e1_profile.csv"), index=False)
    pd.DataFrame([r["summ"] for r in res]).to_csv(os.path.join(OUT, "e1_ladder_summary.csv"), index=False)
    pd.concat([r["lrt"] for r in res]).to_csv(os.path.join(OUT, "e1_lrt_by_c.csv"), index=False)
    pd.concat([r["small"] for r in res]).to_csv(os.path.join(OUT, "e1_smallest_resid_by_c.csv"), index=False)
    pd.DataFrame(sh_rows).to_csv(os.path.join(OUT, "e1_smallest_resid_at_c_sh.csv"), index=False)
    pd.concat([r["resid"] for r in res] + [pd.DataFrame(rs)]).to_csv(os.path.join(OUT, "e1_resid_by_size.csv"),
                                                                     index=False)
    pd.DataFrame([dict(test=k, **d) for k, d in pool_out.items()]).to_csv(os.path.join(OUT, "e1_pooled.csv"),
                                                                          index=False)
    pd.DataFrame(dict(Lambda_boot=lam_b)).to_csv(os.path.join(OUT, "e1_pooled_boot.csv"), index=False)
    pd.DataFrame([v]).to_csv(os.path.join(OUT, "e1_verdict.csv"), index=False)
    json.dump(dict(seed=SEED, B_lrt=L.dm1b.B_BOOT, B_resid=B_RESID, B_free=B_FREE, B_pool=B_POOL,
                   seconds=time.time() - t0, grid_fine=[0, 3, 0.01], host="skampere2"),
              open(os.path.join(OUT, "e1_run.json"), "w"), indent=1)
    print(json.dumps(v, indent=1, default=str))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
