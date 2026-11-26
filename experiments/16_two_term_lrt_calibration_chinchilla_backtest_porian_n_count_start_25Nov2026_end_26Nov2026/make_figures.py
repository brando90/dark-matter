"""
Figures for experiments/16_two_term_lrt_calibration_chinchilla_backtest_porian_n_count_start_25Nov2026_end_26Nov2026. Reads only the saved files in results/ and writes PNGs to
results/figures/. A colour means one thing across the whole report: blue, orange and aqua (categorical slots 1 to 3) are
the joint model, the second N term and the second D term in Figures 2 and 4; magenta and green (slots 5 and 6) are the two
datasets in Figure 3; the simulated truths in Figure 1 are neutral grays. Each figure's own series were checked with the
dataviz palette validator (pairs: blue/orange/aqua, magenta/green: all checks pass, magenta is below 3:1 contrast on white
so every bar carries a value label). Text uses ink tokens, never series colors.

    python make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

INK, INK2, GRID, AXIS, SURF = "#191919", "#666666", "#e7e7e7", "#c2c2c2", "#ffffff"
BLUE, ORANGE, AQUA, GRAY = "#257de0", "#f47230", "#00b894", "#9c9c9c"
MAGENTA, GREEN, DARK = "#ee7fac", "#008a17", "#333333"
DPI = 150
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS, "axes.linewidth": 1.0,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})


def chrome(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(length=0, labelsize=9.5, pad=5)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8, linestyle="-")


def header(fig, title, subtitle):
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.012, 0.912, subtitle, ha="left", va="top", fontsize=9.8, color=INK2, linespacing=1.35)


def legend_row(ax, handles, labels, loc="upper left", ncol=None, anchor=None):
    leg = ax.legend(handles, labels, loc=loc, frameon=False, ncol=ncol or len(labels), fontsize=9.5,
                    handlelength=1.2, handletextpad=0.5, columnspacing=1.4, borderaxespad=0.0, bbox_to_anchor=anchor)
    for t in leg.get_texts():
        t.set_color(INK2)
    return leg


# --------------------------------------------------------------------------------------------------
def fig1():
    rows = [json.loads(l) for l in (RES / "e1_reps.jsonl").read_text().splitlines()]
    K = 10
    sigma = 0.003
    r0 = [r for r in rows if r["truth"] == "H0" and r["sigma"] == sigma]
    r1 = [r for r in rows if r["truth"] == "H1" and r["sigma"] == sigma]
    huber = lambda rs: np.maximum([2.0 * (r["h_obj1"] - r["h_obj2"]) for r in rs], 1e-10)
    gauss = lambda rs: np.maximum([K * np.log(r["g_rss1"] / r["g_rss2"]) for r in rs], 0.0)
    summ = json.loads((RES / "e1_summary.json").read_text())["settings"][f"sigma={sigma}"]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.5), dpi=DPI)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.71, bottom=0.13, wspace=0.18)
    header(fig, "The repo's likelihood-ratio test cannot reject; a plain Gaussian test can",
           f"Simulated 10-model size ladders with 0.3% noise. The truth is one power-law term (light grey, {len(r0)} runs) or two (dark grey, {len(r1)} runs).\n"
           "Each curve is the share of runs whose test statistic is at or below the value on the x-axis; the dashed line is the nominal 5% cutoff.")
    cut = 5.991
    for ax, f, title in ((a, huber, "Repo statistic: 2 × (Huber loss gain), delta = 0.001"),
                         (b, gauss, "Gaussian LRT: K × log(RSS one term / RSS two terms)")):
        chrome(ax, grid="y")
        for rs, col, lab in ((r0, GRAY, "one term true"), (r1, DARK, "two terms true")):
            v = np.sort(f(rs))
            y = np.arange(1, len(v) + 1) / len(v)
            ax.step(np.r_[v[0], v], np.r_[0, y], where="post", color=col, linewidth=2.0, solid_capstyle="round",
                    label=lab)
        ax.axvline(cut, color=INK2, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("share of runs at or below", fontsize=9.5, color=INK2)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=10)
    a.set_xscale("log")
    a.set_xlim(1e-10, 30)
    a.set_xticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1, 100][:-1])
    a.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"1e{int(round(np.log10(x)))}" if x < 1 else f"{x:g}"))
    a.text(cut * 0.80, 0.50, "cutoff 5.99\n(chi-square, 2 df)", ha="right", va="center", fontsize=9, color=INK)
    a.text(2e-10, 0.10, "exact zeros are\ndrawn at 1e-10", fontsize=8.5, color=INK2, ha="left", va="bottom")
    a.text(2e-10, 0.97, f"Every run sits far left\nof the cutoff: false\npositives {summ['repo_nominal_false_positive_rate'] * 100:.0f}%, "
                        f"power {summ['repo_nominal_power'] * 100:.0f}%.", fontsize=9, color=INK, ha="left", va="top")
    b.set_xlim(0, 25)
    b.set_xticks([0, 5, 10, 15, 20, 25])
    fpk, fpn = summ["gauss_nominal_false_positive_count"]
    b.text(cut + 0.5, 0.18, f"cutoff 5.99: false positives {summ['gauss_nominal_false_positive_rate'] * 100:.0f}% ({fpk} of {fpn}),\n"
                            f"power {summ['gauss_nominal_power'] * 100:.0f}%", ha="left", va="center", fontsize=9, color=INK)
    h, l = a.get_legend_handles_labels()
    legend_row(a, h, l, loc="lower right", ncol=1, anchor=(0.915, 0.03))      # right edge left of the cutoff line (at 94% of the axis)
    legend_row(b, h, l, loc="center right", ncol=1, anchor=(1.0, 0.55))
    fig.savefig(FIG / "fig1_lrt_calibration.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------
def fig2():
    s = json.loads((RES / "e2_summary.json").read_text())
    by_axis = s["cap=1.5"]["median_test_rmse_by_axis"]
    prof = pd.read_csv(RES / "e2_profile.csv")
    pb = prof[prof.profiled.str.startswith("b")].sort_values("fixed_value")

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=DPI, gridspec_kw=dict(width_ratios=[1.05, 1]))
    fig.subplots_adjust(left=0.165, right=0.985, top=0.76, bottom=0.14, wspace=0.30)
    header(fig, "A second data term helps on all 245 Chinchilla runs; a second size term does not",
           "245 fitted points (MassiveText). Left: median error on held-out points, over four train/test cuts per axis (smooth power laws only).\n"
           "Right: how much higher the fit objective gets when the data exponent b is forced away from its best value. Figure 4 drops the shortest runs.")
    axes_lab = [("N", "to larger models"), ("D", "to more tokens"), ("C", "to more compute")]
    models = [("joint", BLUE, "joint model"), ("+N dark", ORANGE, "+ second N term"), ("+D dark", AQUA, "+ second D term")]
    chrome(a, grid="x")
    h = 0.17
    gap = 0.03
    ypos = np.arange(len(axes_lab))[::-1].astype(float)
    for j, (m, col, lab) in enumerate(models):
        offs = (1 - j) * (h + gap)
        vals = [by_axis[ax][m] for ax, _ in axes_lab]
        a.barh(ypos + offs, vals, height=h, color=col, label=lab)
        for yy, v in zip(ypos + offs, vals):
            a.text(v + 0.0008, yy, f"{v:.3f}", va="center", ha="left", fontsize=8.5, color=INK)
    a.set_yticks(ypos)
    a.set_yticklabels([t for _, t in axes_lab], fontsize=9.5, color=INK)
    a.set_xlim(0, 0.056)
    a.set_xlabel("median error on held-out points (log-loss units)", fontsize=9.5, color=INK2)
    a.set_title("Extrapolating...", loc="left", fontsize=10.5, color=INK, pad=8)
    a.spines["left"].set_visible(False)
    hh, ll = a.get_legend_handles_labels()
    leg = a.legend(hh, ll, loc="center right", frameon=False, fontsize=9, handlelength=1.1, handletextpad=0.5,
                   labelspacing=0.5, bbox_to_anchor=(1.0, 0.50))
    for t in leg.get_texts():
        t.set_color(INK2)

    chrome(b, grid="y")
    x, y = pb.fixed_value.values, pb.excess_pct.values
    b.plot(x, y, color=BLUE, linewidth=2.0, solid_capstyle="round", solid_joinstyle="round")
    b.fill_between(x, 0, np.where(y <= 5, y, np.nan), color=BLUE, alpha=0.10, linewidth=0)
    i = int(np.argmin(y))
    b.scatter([x[i]], [y[i]], s=64, color=BLUE, edgecolor=SURF, linewidth=2, zorder=5)
    b.axhline(5, color=GRID, linewidth=0.8)
    b.axvline(0.36, color=AXIS, linewidth=1.0)
    b.text(0.375, 62, "0.36:\nBesiroglu et al.\n(reference)", fontsize=8.5, color=INK2, ha="left", va="top")
    band = pb[pb.excess_pct <= 5]
    b.text(0.152, 121, f"Within 5% of the best objective:\nb from {band.fixed_value.min():.3f} to {band.fixed_value.max():.3f},\n"
                       f"compute-optimal\nexponent {band.a_opt.min():.3f} to {band.a_opt.max():.3f}", fontsize=9, color=INK, ha="left", va="top")
    b.set_xscale("log")
    b.set_xticks([0.15, 0.2, 0.3, 0.5, 0.8, 1.2])
    b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    b.minorticks_off()
    b.set_ylim(0, 125)
    b.set_title("Profile of the fit over b", loc="left", fontsize=10.5, color=INK, pad=8)
    b.set_xlabel("data exponent b held fixed (loss ~ D^-b)", fontsize=9.5, color=INK2)
    b.set_ylabel("objective above the best fit (%)", fontsize=9.5, color=INK2)
    fig.savefig(FIG / "fig2_chinchilla_extrapolation.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------
def fig3():
    a4 = pd.read_csv(RES / "e4_n_proxy.csv", keep_default_na=False)
    a4 = a4[(a4.min_steps == 1000) & (a4.D_def == "D_repo")]
    bend = pd.read_csv(RES / "e4_small_n_bend_by_proxy.csv", keep_default_na=False)
    order = ["wxd", "ckpt", "12ld2", "12ld2+emb"]
    xl = ["width × depth\n(current repo proxy)", "exact label\n(all parameters)", "12·depth·width²\n(no embeddings)", "that plus\nembeddings"]
    xl_short = ["width ×\ndepth", "exact\nlabel", "no\nembeddings", "plus\nembeddings"]
    sets = [("rw", MAGENTA, "RefinedWeb"), ("owt2", GREEN, "OpenWebText2")]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.9), dpi=DPI, gridspec_kw=dict(width_ratios=[1.25, 1]))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.77, bottom=0.25, wspace=0.18)
    header(fig, "The parameter count moves the answer; a second size term shows up only with total counts",
           "Porian et al. runs with the Chinchilla learning-rate schedule and at least 1,000 steps (54 per dataset, 15 model sizes from 7M to 901M).\n"
           "Left: compute-optimal exponent of the joint fit, four parameter counts. Right: fit-objective reduction from a second N term, all 15 sizes.")
    chrome(a, grid="y")
    w, gap = 0.30, 0.04
    xs = np.arange(len(order), dtype=float)
    for j, (ds, col, lab) in enumerate(sets):
        vals = [float(a4[(a4.dataset == ds) & (a4.N_proxy == p)].a_opt.iloc[0]) for p in order]
        off = (j - 0.5) * (w + gap)
        a.bar(xs + off, vals, width=w, color=col, label=lab)
        for xx, v in zip(xs + off, vals):
            a.text(xx, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK)
    a.set_xticks(xs)
    a.set_xticklabels(xl, fontsize=8.8, color=INK)
    a.set_ylim(0, 0.9)
    a.set_ylabel("compute-optimal exponent  b / (a + b)", fontsize=9.5, color=INK2)
    hh, ll = a.get_legend_handles_labels()
    legend_row(a, hh, ll, loc="upper left")

    chrome(b, grid="y")
    full = bend[bend.n_min_ckpt == 0.0]
    for j, (ds, col, lab) in enumerate(sets):
        sub = full[full.dataset == ds].set_index("N_proxy")
        vals = [max(float(sub.loc[p, "improvement_pct"]), 0.0) for p in order]
        flag = ["e_hi" in str(sub.loc[p, "flags_Ndark"]) for p in order]
        off = (j - 0.5) * (w + gap)
        b.bar(xs + off, vals, width=w, color=col)
        for xx, v, fl in zip(xs + off, vals, flag):
            b.text(xx + (j - 0.5) * 0.10, v + 0.25, ("†" if fl else "") + f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5, color=INK)
    b.set_xticks(xs)
    b.set_xticklabels(xl_short, fontsize=8.8, color=INK)
    b.set_ylim(0, 17)
    b.set_ylabel("objective reduction from a second N term (%)", fontsize=9.5, color=INK2)
    legend_row(b, hh, ll, loc="upper left")
    fig.text(0.545, 0.035, "† the fit pins a term at the exponent cap (1.5)\nWith only the models of 30M and up (10 sizes),\nthe gain is 0.0% under every definition",
             ha="left", va="bottom", fontsize=8.8, color=INK, linespacing=1.35)
    a.set_title("Compute-optimal exponent", loc="left", fontsize=10.5, color=INK, pad=8)
    b.set_title("Does a second N term help?", loc="left", fontsize=10.5, color=INK, pad=8)
    fig.savefig(FIG / "fig3_porian_n_proxy.png", dpi=DPI)
    plt.close(fig)


# --------------------------------------------------------------------------------------------------
def _a_opt(desc: str) -> float:
    """Compute-optimal exponent b/(a+b) from a printed joint fit 'E=.. + A/N^a + B/D^b'."""
    import re
    a = float(re.search(r"/N\^([0-9.e+-]+)", desc).group(1))
    b = float(re.search(r"/D\^([0-9.e+-]+)", desc).group(1))
    return b / (a + b)


def fig4():
    s_all = json.loads((RES / "e2_summary.json").read_text())
    s1 = json.loads((RES / "e2b" / "f1_drop_DN_lt_0.35" / "e2b_summary.json").read_text())
    s2 = json.loads((RES / "e2b" / "f2_drop_DN_lt_1" / "e2b_summary.json").read_text())
    ao = [_a_opt(s_all["free_joint"]["desc"]), s1["free_joint"]["a_opt"], s2["free_joint"]["a_opt"]]
    subsets = [("all 245 runs", s_all), ("without the 4 shortest\n(D/N below 0.35)", s1), ("without the 11 shortest\n(D/N below 1)", s2)]
    models = [("joint", BLUE, "joint model"), ("+D dark", AQUA, "+ second D term")]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.6), dpi=DPI, sharey=True)
    fig.subplots_adjust(left=0.205, right=0.985, top=0.69, bottom=0.17, wspace=0.07)
    header(fig, "The second data term's gain shrinks, and mostly vanishes, when short runs are dropped",
           "Same quantile cuts as Figure 2, recomputed on each subset (smooth power laws), after dropping the runs with the fewest tokens per parameter.\n"
           "Left: train on the smaller-D runs, test on the larger-D runs. Right: median over all twelve cuts (size, token and compute).\n"
           f"Compute-optimal exponent of the joint fit: {ao[0]:.2f} on all runs, {ao[1]:.2f} without the 4 shortest, {ao[2]:.2f} without the 11 shortest.")
    ypos = np.arange(len(subsets))[::-1].astype(float)
    h, off = 0.30, 0.17
    for ax, getter, title in ((a, lambda s, m: s["cap=1.5"]["median_test_rmse_by_axis"]["D"][m], "Extrapolating to more tokens"),
                              (b, lambda s, m: s["cap=1.5"]["median_test_rmse"][m], "Over all twelve cuts")):
        chrome(ax, grid="x")
        for j, (m, col, lab) in enumerate(models):
            vals = [getter(s, m) for _, s in subsets]
            yy = ypos + (off if j == 0 else -off)
            ax.barh(yy, vals, height=h, color=col, label=lab)
            for y_, v in zip(yy, vals):
                ax.text(v + 0.0006, y_, f"{v:.4f}", va="center", ha="left", fontsize=8.5, color=INK)
        ax.set_xlim(0, 0.034)
        ax.set_xticks([0, 0.01, 0.02, 0.03])
        ax.set_xlabel("median error on held-out points (log-loss units)", fontsize=9.5, color=INK2)
        ax.set_title(title, loc="left", fontsize=10.5, color=INK, pad=8)
        ax.spines["left"].set_visible(False)
    a.set_yticks(ypos)
    a.set_yticklabels([t for t, _ in subsets], fontsize=9.5, color=INK)
    hh, ll = a.get_legend_handles_labels()
    leg = a.legend(hh, ll, loc="lower right", frameon=False, fontsize=9, handlelength=1.1, handletextpad=0.5, labelspacing=0.5)
    for t in leg.get_texts():
        t.set_color(INK2)
    fig.text(0.012, 0.035, "The 4-run and 11-run cut-offs were chosen after seeing the residuals (post hoc). Lower is better.",
             ha="left", va="bottom", fontsize=8.8, color=INK2)
    fig.savefig(FIG / "fig4_chinchilla_short_runs.png", dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("wrote", sorted(p.name for p in FIG.glob("*.png")))
