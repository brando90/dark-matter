"""E1 figures and tables from results/e1_*.csv (PREREG.md). Every axis has a grid.

    python summarize_e1.py     # writes results/e1_profiles.png, e1_resid_by_size.png, e1_smallest_z.png,
                               # results/e1_tables.md (tables pasted into results.md)
"""
import math
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
LADS = ["porian-rw", "porian-owt2", "chinchilla", "pythia"]


def ranges(txt):
    """'0.00;0.05;0.10;0.50' -> '0.00-0.10, 0.50' (runs on the 0.05 grid)."""
    if txt is None or (isinstance(txt, float) and math.isnan(txt)) or txt == "":
        return "none"
    if not isinstance(txt, str):
        txt = f"{txt:.2f}"
    if not txt:
        return "none"
    cs = [float(x) for x in txt.split(";")]
    out, a = [], cs[0]
    for x, y in zip(cs, cs[1:] + [None]):
        if y is None or y - x > 0.051:
            out.append(f"{a:.2f}" if a == x else f"{a:.2f}-{x:.2f}")
            a = y
    return ", ".join(out)


def main():
    P = pd.read_csv(os.path.join(OUT, "e1_profile.csv"))
    S = pd.read_csv(os.path.join(OUT, "e1_ladder_summary.csv")).set_index("ladder").loc[LADS]
    R = pd.read_csv(os.path.join(OUT, "e1_resid_by_size.csv"))
    SM = pd.read_csv(os.path.join(OUT, "e1_smallest_resid_by_c.csv"))
    SH = pd.read_csv(os.path.join(OUT, "e1_smallest_resid_at_c_sh.csv")).set_index("ladder").loc[LADS]
    LR = pd.read_csv(os.path.join(OUT, "e1_lrt_by_c.csv"))
    PO = pd.read_csv(os.path.join(OUT, "e1_pooled.csv")).set_index("test")
    V = pd.read_csv(os.path.join(OUT, "e1_verdict.csv")).iloc[0]
    c_sh = PO.loc["primary_1term_all4", "c_sh"]

    fig, axs = plt.subplots(1, 5, figsize=(22, 4))
    pooled = 0
    for ax, lad in zip(axs, LADS):
        p = P[P.ladder == lad]
        ax.plot(p.c, p.stat1, label="1-term")
        ax.plot(p.c, p.stat2, label="2-term")
        ax.axhline(3.841, color="k", ls="--", lw=0.8, label="95% cut (3.84)")
        ax.set_ylim(0, 40)
        ax.set_title(lad)
        ax.set_xlabel("c (weight per V x d embedding block)")
        ax.grid(True)
        n = S.loc[lad, "npts"]
        pooled = pooled + n * np.log(p.rss1.values / n)
    axs[0].set_ylabel("n ln(RSS(c) / RSS_min)")
    axs[0].legend(fontsize=7)
    axs[4].plot(P.c.unique(), pooled - pooled.min())
    axs[4].axhline(3.841, color="k", ls="--", lw=0.8)
    axs[4].set_ylim(0, 40)
    axs[4].set_title("pooled 1-term, shared c")
    axs[4].set_xlabel("c")
    axs[4].grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e1_profiles.png"), dpi=120)

    fig, axs = plt.subplots(1, 4, figsize=(20, 4))
    for ax, lad in zip(axs, LADS):
        r = R[R.ladder == lad]
        for (law, at), g in r.groupby(["law", "at"], sort=False):
            ax.plot(g.N_std, g.resid, marker="o", ms=3, label=f"{law} {at} (c={g.c.iloc[0]:.2f})")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(lad)
        ax.set_xlabel("N in the standard count")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(True)
        ax.legend(fontsize=6)
    axs[0].set_ylabel("residual y - fit (nats)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e1_resid_by_size.png"), dpi=120)

    fig, ax = plt.subplots(figsize=(7, 4))
    for lad in LADS:
        s = SM[SM.ladder == lad]
        ax.plot(s.c, s.z, marker=".", label=lad)
    ax.axhspan(-2, 2, color="0.85", label="within 2 bootstrap sd")
    ax.set_xlabel("c")
    ax.set_ylabel("smallest-size residual / bootstrap sd")
    ax.grid(True)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "e1_smallest_z.png"), dpi=120)

    L = []
    L.append("| ladder | sizes | c1 (1-term) | 1-term 95% set | excl. 1 / 0 | c2 (2-term) | 2-term 95% set | "
             "dBIC free (1+c vs 2+c) | c-free LRT p | sd 1-term+c / 2-term+c |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for lad in LADS:
        s = S.loc[lad]
        L.append(f"| {lad} | {s.npts} | {s.c1:.2f} | {s.set1} | {'yes' if s.set1_excl1 else 'no'} / "
                 f"{'yes' if s.set1_excl0 else 'no'} | {s.c2:.2f} | {s.set2} | {s.dbic_free:+.1f} | {s.p_free:.3f} | "
                 f"{s.sigma1:.4f} / {s.sigma2:.4f} |")
    L.append("")
    L.append("| ladder | r_min at c = 0 (z) | at c_std (z) | at c_sh = %.2f (z) | leave-out at c_sh (z) | "
             "c in own 95%% set where within 2 sd |" % c_sh)
    L.append("|---|---|---|---|---|---|")
    for lad in LADS:
        s, h = S.loc[lad], SH.loc[lad]
        sm = SM[SM.ladder == lad].set_index("c")
        z0 = sm.loc[0.0]
        zs = sm.iloc[int(np.argmin(np.abs(sm.index.values - s.c_std)))]
        L.append(f"| {lad} | {z0.r_min:+.3f} ({z0.z:+.1f}) | {zs.r_min:+.3f} ({zs.z:+.1f}) | {h.r_min:+.3f} "
                 f"({h.z:+.1f}) | {h.e_leaveout:+.3f} ({h.z_leaveout:+.1f}) | "
                 f"{ranges(V[lad + '_own_set_c_within'])} |")
    L.append("")
    L.append("| test | c_sh | Lambda | dBIC_shared (BIC_shared - BIC_per) | bootstrap p |")
    L.append("|---|---|---|---|---|")
    for t, r in PO.iterrows():
        L.append(f"| {t} | {r.c_sh:.2f} | {r.Lambda:.2f} | {r.dbic_shared:+.2f} | "
                 f"{'' if pd.isna(r.p_boot) else f'{r.p_boot:.3f} (B {int(r.nboot)})'} |")
    L.append("")
    lr = LR.pivot(index="at", columns="ladder", values="detect")
    grid = lr.loc[[f"{c:.1f}" for c in np.round(np.arange(0, 3.01, 0.1), 1)]]
    L.append("LRT (B = 200) detects a second term at c in {0, 0.1, ..., 3}:")
    for lad in LADS:
        cs = [float(c) for c in grid.index[grid[lad] == 1]]
        L.append(f"- {lad}: {len(cs)} of 31 grid values" + (f" ({min(cs):.1f}-{max(cs):.1f})" if cs else ""))
    L.append("")
    # post hoc, labelled: pooled 1-term profile past the grid, from the preregistered secondary RSS1 at c = 4..10
    ph = []
    for c in (3.0, 4.0, 5.0, 7.0, 10.0):
        tot = 0.0
        for lad in LADS:
            n = S.loc[lad, "npts"]
            rss = P[(P.ladder == lad) & np.isclose(P.c, 3.0)].rss1.iloc[0] if c == 3.0 else S.loc[lad,
                                                                                               f"secondary_rss1_c{c:g}"]
            tot += n * math.log(rss / n)
        ph.append((c, tot))
    base = ph[0][1]
    L.append("Post hoc (labelled): pooled 1-term -2 log-lik relative to c = 3: " +
             ", ".join(f"c = {c:g}: {t - base:+.2f}" for c, t in ph))
    L.append("Per-ladder RSS1 at c = 3 / 4 / 5 / 7 / 10: " + "; ".join(
        f"{lad} " + " / ".join(f"{v:.5f}" for v in [P[(P.ladder == lad) & np.isclose(P.c, 3.0)].rss1.iloc[0]] +
                              [S.loc[lad, f'secondary_rss1_c{c:g}'] for c in (4, 5, 7, 10)]) for lad in LADS))
    txt = "\n".join(L)
    open(os.path.join(OUT, "e1_tables.md"), "w").write(txt + "\n")
    print(txt)
    print(V.to_string())


if __name__ == "__main__":
    main()
