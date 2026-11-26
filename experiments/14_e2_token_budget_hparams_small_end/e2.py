"""E2 (PREREG.md, 11-25-2026): how much of Porian's small-end excess loss is token budget, and how much hyperparameters?

    python e2.py                 # writes results/e2_*.csv and results/bootstrap/ (all draws, resumable)

Every number in nats. Excess X_s = y_s - 1-term law fitted on the reference sizes (tied N >= 1e8) of the same
condition. Factor contributions and decision rules: PREREG.md.
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

import e2lib as L
import cache_io as CACHE

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
SEED = 20261125
B_N, B_J = 2000, 500
SIG_FLOOR = 0.002
JOINT_MISFIT = 0.02
KS = [5, 10, 20]
JOINT_K = [5, 10, 20, 40, 80]
COND = {"TC20": ("TS", 20), "TC5": ("TS", 5), "TC10": ("TS", 10), "TCfin": ("TS", "final"),
        "BC20": ("BC", 20), "BC5": ("BC", 5), "BC10": ("BC", 10), "BCfin": ("BC", "final"), "BKfin": ("BSK", "final")}
SWEEP_LARGE = [640.0, 832.0]


# ---------------- pieces that verify.py also drives ----------------

def excess_boot(lad, cname, sd, rng, B, eps_small=None, eps_ref=None):
    """Observed X at every non-reference size and B bootstrap draws. lad: width, N_nonemb, N_tied, y.
    sd: {width: seed sd}. eps_small: optional {width: (B,) standard normals} to share small-size draws.
    Returns dict(obs={width: X}, boot={width: (B,)}, fit=1-term dict, sigma_ref, nref) or None (< 4 ref sizes)."""
    N = L.count(L.COUNTS[cname], lad.N_nonemb.values, lad.width.values)
    n = N / L.N_REF
    y = lad.y.values
    ref = lad.N_tied.values >= L.REF_MIN
    if ref.sum() < 4:
        return None
    f = L.fit1(n[ref], y[ref])
    nr = int(ref.sum())
    sig = max(np.sqrt(f["rss"] / (nr - 3)), SIG_FLOOR)
    ref_noise = rng.normal(0, 1, (B, nr)) if eps_ref is None else eps_ref
    Ystar = L.pred1(f, n[ref])[None, :] + sig * ref_noise
    fb = L.fit1_many(n[ref], Ystar)
    obs, boot = {}, {}
    for j in np.flatnonzero(~ref):
        w = float(lad.width.values[j])
        if w not in L.SMALL:
            continue
        e = eps_small[w] if eps_small is not None and w in eps_small else rng.normal(0, 1, B)
        ys = y[j] + sd[w] * e
        pb = fb["E"] + fb["a"] * n[j] ** -fb["p"]
        obs[w] = float(y[j] - L.pred1(f, n[j]))
        boot[w] = ys - pb
    return dict(obs=obs, boot=boot, fit=f, sigma_ref=sig, nref=nr, B=B, fitted_draws=fb)


def joint_points_tc(ds):
    pts = []
    for k in JOINT_K:
        d = L.ladder(ds, "TS", k)
        pts.append(d[d.N_tied >= L.REF_MIN].assign(k=k))
    return pd.concat(pts, ignore_index=True)


def joint_points_bc(ds):
    P = L.porian()
    b = P[(P.dataset == ds) & (P.hparams == "base") & (P.warmup == "short") & (P.decay == "chinchilla") &
          (P.N_total >= L.REF_MIN) & (P.tpp >= 5) & (P.tpp <= 80)]
    return pd.DataFrame(dict(width=b.width.values, N_nonemb=b.N_nonemb.values, N_tied=b.N_total.values,
                             y=b.final_loss.values, D=b.final_tokens.values, k=b.tpp.values))


def joint_boot(J, small, cname, sd, rng, B, eps_small, eps_ref=None, partial=None):
    """Additive Chinchilla joint law on points J; predict each small size (rows of `small`, y at D = 20 N_tied).
    Returns dict(obs, boot (first B draws), fit, sd_fit, npts)."""
    N = L.count(L.COUNTS[cname], J.N_nonemb.values, J.width.values) / L.N_REF
    d = J.D.values / 1e9
    y = J.y.values
    f = L.F.fitJ1(N, d, y)
    npts = len(y)
    sdf = float(np.sqrt(f["rss"] / (npts - 5)))
    sig = max(sdf, SIG_FLOOR)
    mu = L.F.predJ(f, N, d)
    ns = L.count(L.COUNTS[cname], small.N_nonemb.values, small.width.values) / L.N_REF
    ds_ = 20 * small.N_tied.values / 1e9
    obs = {float(w): float(yv - L.F.predJ(f, np.array([a]), np.array([b]))[0])
           for w, yv, a, b in zip(small.width, small.y, ns, ds_)}
    preds = np.full((B, len(small)), np.nan)
    ref_noise = rng.normal(0, 1, (B, npts)) if eps_ref is None else eps_ref
    key = CACHE.digest(dict(N=N, d=d, y=y, ns=ns, ds=ds_, B=B, eps_ref=ref_noise))
    start = 0
    if partial is not None and os.path.exists(partial):
        old = CACHE.read(partial)
        if old['key'] == key:
            preds, start = old['preds'], old['completed']
    for i in range(start, B):
        g = L.F.fitJ1(N, d, mu + sig * ref_noise[i], x0=[f["alpha"], f["gamma"]])
        preds[i] = L.F.predJ(g, ns, ds_)
        if partial is not None and ((i + 1) % 25 == 0 or i + 1 == B):
            CACHE.save(partial, dict(key=key, version=CACHE.VERSION, preds=preds, completed=i+1, target_B=B))
            print(f'{os.path.basename(partial)}: {i+1}/{B}', flush=True)
    boot = {float(w): small.y.values[j] + sd[float(w)] * eps_small[float(w)][:B] - preds[:, j]
            for j, w in enumerate(small.width)}
    return dict(obs=obs, boot=boot, fit=f, sd_fit=sdf, npts=npts, B=B, predictions=preds)


def verdict(C, lo, hi, base, sd2):
    if not np.isfinite(C):
        return "n/a"
    if not np.isfinite(base) or base <= 0:
        return "N/A (no excess)"
    if C >= 0.5 * base and C >= sd2 and lo > 0:
        return "EXPLAINS"
    if lo > 0 and C >= 0.2 * base:
        return "PARTLY"
    if hi < 0 and abs(C) >= 0.2 * base:
        return "OPPOSITE"
    return "NO"


def ret_verdict(M, lo, hi, sd2):
    if not np.isfinite(M):
        return "n/a"
    if abs(M) <= sd2 and lo <= 0 <= hi:
        return "RETURNS"
    if M > sd2 and lo > 0:
        return "STAYS OFF"
    return "UNDETERMINED"


# ---------------- sweep (rw) ----------------

def sweep_table(rng, B, sd, bank=None):
    """Per sweep size: y_best3, y_min, base-like, tuned recipe at the sweep's D/N; gaps with B draws."""
    raw = pd.read_pickle(L.D.PORIAN)
    sw = raw[(raw.hparams == "sweep") & (raw.dataset == "rw")].copy()
    sw["y"] = [float(v.dropna().values[-1]) for v in sw["val/loss"]]
    P = L.porian()
    tc = P[(P.dataset == "rw") & (P.hparams == "tuned") & (P.warmup == "short") & (P.decay == "const")]
    rows, draws = [], {}
    for w, g in sw.groupby("width"):
        w = float(w)
        r0 = tc[tc.width == w].iloc[0]
        tok = g.max_step.values * g.bs.values * g.seq_len.values
        tpp = float(np.median(tok / r0.N_total))
        yt = L.D.porian_curve_value(r0, ratio=tpp)
        bl = g[(g.lr == 0.003) & (g.beta2 == 0.95) & (g.bs <= 256)]
        blr = bl.loc[bl.bs.idxmax()]
        ys = np.sort(g.y.values)
        s = sd[("rw", w)]
        Ystar = g.y.values[None, :] + s * rng.normal(0, 1, (B, len(g)))
        best3_b = np.sort(Ystar, 1)[:, :3].mean(1)
        bl_b = Ystar[:, list(g.index).index(blr.name)]
        yt_b = yt + s * (rng.normal(0, 1, B) if bank is None else bank.get("TS", w, tpp*r0.N_total)[:B])
        rows.append(dict(width=w, N_tied=r0.N_total, configs=len(g), tpp=tpp, y_min=ys[0], y_best3=ys[:3].mean(),
                         base_like_bs=blr.bs, y_base_like=blr.y, y_tuned=yt, gap_base=blr.y - ys[:3].mean(),
                         gap_tuned=yt - ys[:3].mean(), gap_tuned_vs_min=yt - ys[0], gap_base_vs_min=blr.y - ys[0],
                         best_cfg=f"lr {g.loc[g.y.idxmin()].lr} bs {g.loc[g.y.idxmin()].bs:g} b2 {g.loc[g.y.idxmin()].beta2}",
                         seed_sd=s))
        draws[w] = dict(gap_base=bl_b - best3_b, gap_tuned=yt_b - best3_b)
    T = pd.DataFrame(rows)
    big = T.width.isin(SWEEP_LARGE)
    T["H"] = T.gap_base - T.gap_base[big].mean()
    T["G"] = T.gap_tuned - T.gap_tuned[big].mean()
    Hb = {w: draws[w]["gap_base"] - np.mean([draws[v]["gap_base"] for v in SWEEP_LARGE], 0) for w in draws}
    Gb = {w: draws[w]["gap_tuned"] - np.mean([draws[v]["gap_tuned"] for v in SWEEP_LARGE], 0) for w in draws}
    return T, Hb, Gb


# ---------------- main analysis ----------------

def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def analyse(cname, verdicts=True):
    sd = L.seed_sd()
    rows_x, rows_f, rows_m, rows_j, sweep_out = [], [], [], [], None
    for di, ds in enumerate(("rw", "owt2")):
        sdd = {w: s for (d_, w), s in sd.items() if d_ == ds}
        rng = np.random.default_rng([SEED, di, len(cname)])
        bank = CACHE.NoiseBank(f"{SEED}|{ds}|real", B_N)
        cache_dir = os.path.join(OUT, "bootstrap", cname, ds)
        lads = {c: L.ladder(ds, *v) for c, v in COND.items()}
        eps_tc = {float(r.width): bank.get("TS", r.width, r.D) for r in lads["TC20"].itertuples() if r.width in L.SMALL}
        eps_bc = {float(r.width): bank.get("BC", r.width, r.D) for r in lads["BC20"].itertuples() if r.width in L.SMALL}
        X = {}
        for c, lad in lads.items():
            recipe = COND[c][0]
            share = {float(r.width): bank.get(recipe, r.width, r.D) for r in lad.itertuples() if r.width in L.SMALL}
            ref_noise = bank.matrix(recipe, lad[lad.N_tied >= L.REF_MIN])
            payload = dict(ladder=lad.to_numpy(), count=cname, sd=sdd, B=B_N, small=share, ref=ref_noise)
            X[c] = CACHE.cached(os.path.join(cache_dir, c + '.npz'), payload, rng,
                            lambda: excess_boot(lad, cname, sdd, rng, B_N, share, ref_noise))
            if X[c] is None:
                continue
            for w in L.SMALL:
                if w in X[c]["obs"]:
                    lo, hi = ci(X[c]["boot"][w])
                    rows_x.append(dict(count=cname, dataset=ds, cond=c, size=L.SMALL[w], width=w, X=X[c]["obs"][w],
                                       lo=lo, hi=hi, nref=X[c]["nref"], sigma_ref=X[c]["sigma_ref"],
                                       E=X[c]["fit"]["E"], A=X[c]["fit"]["a"], alpha=X[c]["fit"]["p"]))
        XJ = {}
        for tag, J, src, eps in (("TC", joint_points_tc(ds), "TC20", eps_tc), ("BC", joint_points_bc(ds), "BC20", eps_bc)):
            small = lads[src][lads[src].width.isin(list(L.SMALL))]
            ref_noise = bank.matrix('TS' if tag == 'TC' else 'BC', J, B_J)
            payload = dict(J=J.to_numpy(), small=small.to_numpy(), count=cname, sd=sdd, B=B_J,
                           small_noise=eps, ref_noise=ref_noise)
            XJ[tag] = CACHE.cached(os.path.join(cache_dir, 'joint_' + tag + '.npz'), payload, rng,
                lambda: joint_boot(J, small, cname, sdd, rng, B_J, eps, ref_noise,
                                   os.path.join(cache_dir, 'partial_joint_' + tag + '.npz')))
            f = XJ[tag]["fit"]
            # secondary: joint law on all sizes (same D/N window), residual at the small sizes at 20N
            Jall = joint_points_tc_all(ds) if tag == "TC" else joint_points_bc_all(ds)
            Nn = L.count(L.COUNTS[cname], Jall.N_nonemb.values, Jall.width.values) / L.N_REF
            fa = L.F.fitJ1(Nn, Jall.D.values / 1e9, Jall.y.values)
            ns = L.count(L.COUNTS[cname], small.N_nonemb.values, small.width.values) / L.N_REF
            ra = small.y.values - L.F.predJ(fa, ns, 20 * small.N_tied.values / 1e9)
            rows_j.append(dict(count=cname, dataset=ds, joint=tag, npts=XJ[tag]["npts"], sd_fit=XJ[tag]["sd_fit"],
                               misfit=int(XJ[tag]["sd_fit"] > JOINT_MISFIT), E=f["E"], A=f["a"], alpha=f["alpha"],
                               B=f["g"], beta_D=f["gamma"], all_npts=len(Jall),
                               all_sd=float(np.sqrt(fa["rss"] / (len(Jall) - 5))),
                               **{f"all_resid_{L.SMALL[w]}": float(r) for w, r in zip(small.width, ra)},
                               **{f"XJ_{L.SMALL[w]}": XJ[tag]["obs"].get(w, np.nan) for w in L.SMALL}))
        if ds == "rw":
            def compute_sweep():
                table, hb, gb = sweep_table(rng, B_N, sd, bank)
                return dict(table=table.to_dict('list'), Hb=hb, Gb=gb)
            saved = CACHE.cached(os.path.join(cache_dir, 'sweep.npz'), dict(B=B_N, sd=sd, count=cname, input_sha256=CACHE.file_sha256(L.D.PORIAN)), rng, compute_sweep)
            ST, Hb, Gb = pd.DataFrame(saved['table']), saved['Hb'], saved['Gb']
            sweep_out = ST.assign(count=cname)
        for w in L.SMALL:
            s2 = 2 * sdd[w]

            def g(c, which="obs"):
                if X.get(c) is None or w not in X[c]["obs"]:
                    return np.nan if which == "obs" else np.full(B_N, np.nan)
                return X[c][which][w]

            def gj(tag, which="obs"):
                v = XJ[tag][which].get(w)
                return (np.nan if which == "obs" else np.full(B_J, np.nan)) if v is None else v
            F_ = {}
            F_["T (TC: final -> 20N)"] = (g("TCfin") - g("TC20"), g("TCfin", "boot") - g("TC20", "boot"), g("TCfin"))
            F_["T (BC: final -> 20N)"] = (g("BCfin") - g("BC20"), g("BCfin", "boot") - g("BC20", "boot"), g("BCfin"))
            F_["T2 (BK final -> BC 20N, secondary)"] = (g("BKfin") - g("BC20"), g("BKfin", "boot") - g("BC20", "boot"),
                                                        g("BKfin"))
            F_["J (TC: data term at 20N)"] = (g("TC20") - gj("TC"), g("TC20", "boot")[:B_J] - gj("TC", "boot"), g("TC20"))
            F_["J (BC: data term at 20N)"] = (g("BC20") - gj("BC"), g("BC20", "boot")[:B_J] - gj("BC", "boot"), g("BC20"))
            R = (g("BC20") - g("TC20"), g("BC20", "boot") - g("TC20", "boot"))
            F_["R (recipe: base+cosine -> tuned+const)"] = (R[0], R[1], g("BC20"))
            if ds == "rw" and w in Hb:
                h = float(ST.loc[ST.width == w, "H"].iloc[0])
                gg = float(ST.loc[ST.width == w, "G"].iloc[0])
                F_["H (pure hparams, sweep)"] = (h, Hb[w], g("BC20"))
                F_["S (schedule and rest = R - H)"] = (R[0] - h, R[1] - Hb[w], g("BC20"))
                F_["G (tuned recipe mis-tuning)"] = (gg, Gb[w], g("TC20"))
            for name, (C, Cb, base) in F_.items():
                lo, hi = ci(Cb)
                rows_f.append(dict(count=cname, dataset=ds, factor=name, size=L.SMALL[w], width=w, C=C, lo=lo, hi=hi,
                                   base=base, pct=100 * C / base if np.isfinite(base) and base > 0 else np.nan,
                                   two_sd=s2, verdict=("UNDETERMINED (joint misfit)" if name.startswith("J (") and XJ["TC" if "(TC:" in name else "BC"]["sd_fit"] > JOINT_MISFIT else verdict(C, lo, hi, base, s2)) if verdicts else ""))
            # matched excess M
            mis = XJ["TC"]["sd_fit"] > JOINT_MISFIT
            Mo = g("TC20") if mis else gj("TC")
            Mb = g("TC20", "boot")[:B_J] if mis else gj("TC", "boot")
            Gtxt = "none (owt2: no sweep)"
            if ds == "rw":
                if w in Gb:
                    Mo, Mb = Mo - float(ST.loc[ST.width == w, "G"].iloc[0]), Mb - Gb[w][:B_J]
                    Gtxt = "G subtracted"
                else:
                    Gtxt = "G not available (size not in sweep)"
                    Mo, Mb = np.nan, np.full(B_J, np.nan)
            lo, hi = ci(Mb)
            rows_m.append(dict(count=cname, dataset=ds, size=L.SMALL[w], width=w, M=Mo, lo=lo, hi=hi, two_sd=s2,
                               base_used="X(TC20) (joint misfit)" if mis else "X^J(TC)", G=Gtxt,
                               verdict=(ret_verdict(Mo, lo, hi, s2) if np.isfinite(Mo) else "UNDETERMINED (G unavailable)") if verdicts else ""))
    return (pd.DataFrame(rows_x), pd.DataFrame(rows_f), pd.DataFrame(rows_m), pd.DataFrame(rows_j), sweep_out)


def joint_points_tc_all(ds):
    return pd.concat([L.ladder(ds, "TS", k).assign(k=k) for k in JOINT_K], ignore_index=True)


def joint_points_bc_all(ds):
    P = L.porian()
    b = P[(P.dataset == ds) & (P.hparams == "base") & (P.warmup == "short") & (P.decay == "chinchilla") &
          (P.tpp >= 5) & (P.tpp <= 80)]
    return pd.DataFrame(dict(width=b.width.values, N_nonemb=b.N_nonemb.values, N_tied=b.N_total.values,
                             y=b.final_loss.values, D=b.final_tokens.values))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--count", choices=["tied", "untied", "both"], default="both")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    outs = {}
    for cname, verd in (("tied", True), ("untied", False)):
        if args.count not in ("both", cname):
            continue
        cache = os.path.join(OUT, f"e2_factors_{cname}.csv")
        X, Fa, M, J, S = analyse(cname, verd)
        X.to_csv(os.path.join(OUT, f"e2_excess_{cname}.csv"), index=False)
        M.to_csv(os.path.join(OUT, f"e2_matched_{cname}.csv"), index=False)
        J.to_csv(os.path.join(OUT, f"e2_joint_{cname}.csv"), index=False)
        if S is not None:
            S.to_csv(os.path.join(OUT, f"e2_sweep_{cname}.csv"), index=False)
        Fa.to_csv(cache, index=False)  # written last: marks the count as done
        outs[cname] = Fa
        print(cname, f"{time.time() - t0:.0f}s", flush=True)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
