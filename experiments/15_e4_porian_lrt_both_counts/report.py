"""Render E4 tables and scientific figures from completed, verified outputs (2026-11-26)."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import e4

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
FIG = HERE / "figures"
DATE = "2026-11-26"


def fmt(x, digits=4):
    return "n/a" if pd.isna(x) or x == "" else f"{float(x):.{digits}f}"


def main():
    df = pd.read_csv(OUT / "e4_cells.csv")
    df["rev_signature"] = df.rev_signature.fillna("n/a")
    controls = pd.read_csv(OUT / "controls.csv")
    assert len(df) == 114 and df.B_done.eq(1000).all()
    null = int(controls[controls.kind == "null"].detect.sum())
    positive = int(controls[controls.kind == "positive"].detect.sum())
    assert null <= 6 and positive >= 32 and len(controls) == 80
    FIG.mkdir(exist_ok=True)
    counts = ["nonemb", "tied", "untied"]
    conv = [(rc, str(sl)) for rc in e4.L.RECIPES for sl in e4.L.SLICES[rc]]
    summary = []
    for count in counts:
        q = df[df["count"] == count]
        detected = int(q.detect.sum())
        verdict = "YES" if detected / len(q) >= 2/3 else "NO" if detected / len(q) <= .1 else "MIXED"
        shaped = int(((q.detect == 1) & (q.rev_signature == "POSITIVE")).sum())
        summary.append(dict(count=count, detections=detected, tests=len(q), verdict=verdict, dark_matter_shaped=shaped,
                            reverse_unavailable=int((q.rev_ok == 0).sum())))
    s = pd.DataFrame(summary)
    s.to_csv(OUT / "summary.csv", index=False)
    overall = "UNIVERSAL" if (s.verdict == "YES").all() and df.detect.mean() >= .9 else "ABSENT" if (s.verdict == "NO").all() else "CONVENTION-DEPENDENT"

    # Figure 1: every convention/count, exact p values and the threshold.
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 12), sharey=True, constrained_layout=True)
    for ax, ds in zip(axes, ["rw", "owt2"]):
        arr = np.empty((len(conv), 3))
        for i, (rc, sl) in enumerate(conv):
            for j, count in enumerate(counts):
                r = df[(df.dataset == ds) & (df.recipe == rc) & (df["slice"] == sl) & (df["count"] == count)].iloc[0]
                arr[i, j] = r.p_lrt
        ax.imshow(-np.log10(arr), vmin=0, vmax=np.log10(1001), cmap="Blues", aspect="auto")
        for i in range(len(conv)):
            for j in range(3):
                ax.text(j, i, f"{arr[i,j]:.3f}" + (" *" if arr[i,j] <= .05 else ""), ha="center", va="center", fontsize=9,
                        color="white" if arr[i,j] < .02 else "black")
        ax.set_xticks(range(3), counts)
        ax.set_yticks(range(len(conv)), [f"{rc} / {sl if sl == 'final' else sl+'N'}" for rc, sl in conv])
        ax.set_title(ds)
        ax.set_xlabel("Parameter count")
        ax.grid(True)
    axes[0].set_ylabel("Training recipe / token-budget slice")
    fig.suptitle("E4: bootstrap p-values for a second N term\n* p ≤ 0.05; 1,000 refits for every entry")
    fig.savefig(FIG / "lrt_by_convention.png", dpi=170)
    fig.savefig(FIG / "lrt_by_convention.pdf")
    plt.close(fig)

    # Figure 2: the matched-20N reverse ladder, highlighting its count dependence.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), constrained_layout=True)
    for ax, ds in zip(axes, ["rw", "owt2"]):
        d = e4.L.ladder(ds, "TS", 20)
        for count in counts:
            N = e4.L.count(e4.L.COUNTS[count], d.N_nonemb, d.width)
            residual, _ = e4.L.reverse(N, d.y.to_numpy(), d.N_tied.to_numpy() >= 1e8)
            ax.plot(d.N_tied / 1e6, residual, marker="o", label=count)
        ax.axhline(0, color="black", linewidth=.8)
        ax.axvline(100, color="gray", linestyle="--", linewidth=.8)
        ax.set_xscale("log")
        ax.set_xlabel("Tied model size (millions of parameters)")
        ax.set_ylabel("Reverse-ladder residual (nats)")
        ax.set_title(ds + ": tuned, short warmup, constant LR, D/N = 20")
        ax.legend(title="Fit uses count")
        ax.grid(True)
    fig.savefig(FIG / "reverse_20N_by_count.png", dpi=170)
    fig.savefig(FIG / "reverse_20N_by_count.pdf")
    plt.close(fig)

    # Complete numeric table: every convention/count and all requested small-size residuals.
    columns = ["dataset", "recipe", "slice", "count", "npts", "rmse1", "rmse2", "bic1", "bic2", "dbic", "T", "p_lrt", "B_done", "detect",
               "rev_5M", "rev_7M", "rev_9M", "rev_15M", "rev_mean_small", "rev_frac_pos", "rev_signature"]
    order = {(ds, rc, sl, count): i for i, (ds, rc, sl, count) in enumerate((ds, rc, sl, count) for ds in ["rw", "owt2"] for rc, sl in conv for count in counts)}
    df["order"] = [order[(r.dataset, r.recipe, r.slice, r.count)] for r in df.itertuples()]
    df = df.sort_values("order")
    df[columns].to_csv(OUT / "full_table.csv", index=False)
    header = "| Dataset | Recipe | D/N slice | Count | n | RMSE1 | RMSE2 | ΔBIC | p | Detect | 5M | 7M | 9M | 15M | Small mean | Positive fraction | Reverse signature |"
    table = [header, "|" + "---|" * 17]
    for r in df.itertuples():
        table.append("| " + " | ".join([r.dataset, r.recipe, r.slice, r.count, str(r.npts), fmt(r.rmse1), fmt(r.rmse2), fmt(r.dbic, 3), fmt(r.p_lrt, 6),
                                        "yes" if r.detect else "no", fmt(r.rev_5M), fmt(r.rev_7M), fmt(r.rev_9M), fmt(r.rev_15M), fmt(r.rev_mean_small),
                                        fmt(r.rev_frac_pos, 3), r.rev_signature]) + " |")
    fulltable = "\n".join(table)
    (OUT / "full_table.md").write_text(f"# E4 complete convention/count table ({DATE})\n\nDate: {DATE}. Losses and residuals are in nats. Every row has B = 1,000 completed refits.\n\n" + fulltable + "\n")
    reprod = df[(df.recipe == "TS") & (df["slice"] == "20")]
    headline = []
    for ds in ["rw", "owt2"]:
        r = reprod[(reprod.dataset == ds) & (reprod["count"] == "tied")].iloc[0]
        headline.append(f"{ds}: T = {r['T']:.6f}, p = {r.p_lrt:.6f}, ΔBIC = {r.dbic:.3f}, RMSE {r.rmse1:.5f} → {r.rmse2:.5f} nats, 5M reverse residual +{r.rev_5M:.6f} nats")
    lines = [f"# E4 results ({DATE}): {overall}", f"\nDate: {DATE}. Owner: ub9sc (ultimate brando9 scalable coworker).",
        "\n**Dark-matter question: why do the tiniest models live off the scaling curve, and what correction brings them back?** The two-term law is one candidate. E4 asks whether a second N term is detected under every Porian training convention and parameter count.",
        f"\n**Preregistered verdict: {overall}.** A second term is detected in {int(df.detect.sum())}/114 tests ({df.detect.mean():.1%}). " + "; ".join(f"{r.count}: {r.detections}/{r.tests} ({r.detections/r.tests:.1%}), {r.verdict}" for r in s.itertuples()) + ". None reaches the preregistered YES threshold of two thirds. The overall rule therefore supports convention dependence, not a universal correction or absence under all counts.",
        "\nAll detections are in tuned constant-learning-rate, fixed-D/N slices: 26/72 tuned constant tests versus 0/42 base decayed tests; 26/84 fixed-D/N tests versus 0/30 final-checkpoint tests. Both datasets have 13/57 detections. By recipe: TS 20/36, TL 6/36, BC 0/30, BSK 0/6, BLK 0/6. Tuning and schedule are not crossed in the available runs, so these counts do not separate their causal effects.",
        "\nOnly 4/114 tests both detect and have the preregistered POSITIVE reverse-ladder signature: TS at D/N = 20, rw and owt2, tied and untied counts (2 per count; 0 nonemb). Among the other 22 detections, 8 have FLAT signatures and 14 lack four reference sizes. A reverse ladder is unavailable in 30/114 tests (10 per count). Detection alone is not evidence of the preregistered small-end excess-loss signature.",
        "\n" + "; ".join(headline) + ". Under nonemb at the same slices, both T = 0, p = 1 and ΔBIC = −5.278; the 5M residuals are −0.360215 nats (rw) and −0.374477 nats (owt2). Changing the count changes the apparent correction.",
        "\n## What ran and what was checked",
        "\nOriginal preregistration and fits: Claude Code, Opus 5.5 (claude-opus-5-5), CPU on skampere1. Resume, implementation verification and reports: Codex CLI, GPT-6 Astra ultra, Vals plan; fits are Python computations on skampere1 CPU, not an LLM numerical judgment. No paid API or GPU run. Lean checking and differential testing (DT) are not applicable to these statistical fits. Package versions and input/import SHA-256 hashes are in [run_environment.json](results/run_environment.json). The run used nice priority 10, three worker processes and one BLAS/OpenMP thread per worker.",
        "\nThe [PREREG](PREREG.md) was committed alone before fitting (d690c90). It is unchanged. [verify.py](verify.py) passes the reproduction, counts, full-grid integrity, positive control and null control checks. The 14 finite TS 20N losses equal DM-2 exactly; all six T statistics match E1 within the preregistered relative tolerance, and all six detection decisions agree. Tied 5M residuals agree with DM-2 within 0.005 nats. E1 used B = 200 and E4 uses B = 1,000, so their numerical p-values need not be equal. [Reproduction numbers](results/reproduction.csv), [verification log](results/verify.log).",
        f"\nControls use 40 synthetic ladders each, the rw TS 20N tied-count 1-term fit, independent 0.005-nat Gaussian noise per size, and 1,000 null-bootstrap refits per ladder. Null: {null}/40 false detections (rule ≤6/40). Positive: {positive}/40 detections (rule ≥32/40) after planting a fast term with exponent 1.4 and +0.3 nats at the actual smallest tied size (5,173,248 parameters). [All control results](results/controls.csv). These controls check this design and noise model; they do not establish universal calibration for misspecified real-data curves.",
        "\nResume corrections: the initial snapshot had all 114 result rows but had analytically returned p = 1 for 74 T = 0 rows without executing their claimed 1,000 refits. The resumed run completed all 74,000 missing refits, retaining 40,000 completed positive-T refits only after rechecking observed RSS. All 114 tests now have 1,000 actual refits, with zero nonfinite refits; p-values and detections are unchanged. Each bootstrap saves progress every 100 refits. Cache records now bind the data, seed, bootstrap size and fitter code to a SHA-256 key and record completed refits. The original [run log](results/e4_run.log) is retained beside the [resume log](results/e4_resume.log).",
        "\nTwo draft verifier assertions initially failed because the DM-2 table includes two missing-loss rows and the E1 table repeats c = 2 as c_std. The verifier now selects finite DM-2 losses and the canonical E1 at = c row; no data or statistical rule changed. [Initial assertion logs](results/verify_initial_failure.log).",
        "\n## Data, method and limits",
        "\nThe loader returns 975 Porian runs (the job brief said 977). A convention is dataset × recipe × slice: 19 per dataset, 38 total, evaluated under three counts. Every test has more than six sizes. TS = tuned/short-warmup/constant; TL = tuned/long-warmup/constant; BC = base/short-warmup/cosine decay; BSK and BLK = base/short- or long-warmup/Kaplan decay. For TS/TL, D/N is 5, 10, 20, 40 or 80, plus final; BC has 5, 10, 20 or 40, plus final; BSK/BLK use final only. Constant-rate checkpoints and cosine final losses are interpolated as preregistered, without extrapolation. D/N always uses the tied count, so counts share losses and checkpoints.",
        "\nCounts are nonemb = N_nonemb; tied = N_nonemb + Vw (DM-2's N_total); untied = N_nonemb + 2Vw, with V = 50,432. The 5M/7M/9M/15M labels denote the same architectures under every count, indexed by widths 96/128/160/224; they are approximate tied sizes, not the numerical count used for every fit.",
        "\n" + r"The N-only laws are $L=E+A n^{-\alpha}$ and $L=E+A n^{-\alpha}+B n^{-\beta}$ with $n=N/10^6$, $E,A,B\ge0$, and $\alpha>\beta$: alpha is the fast term. Fitting imports DM-1b unchanged through E1/E2; the imported DM-5 copies are byte-identical to dm1_power_analysis. The null bootstrap uses $T=m\log(\mathrm{RSS}_1/\mathrm{RSS}_2)$ and Gaussian noise variance $\mathrm{RSS}_1/(m-3)$, where m is the number of sizes. All B = 1,000 refits run; $p=(1+k)/1001$, and detection requires $p\le0.05$. The inherited numerical comparison tolerance is $10^{-9}\max(T,1)$. ΔBIC = BIC1 − BIC2 uses 3 versus 5 coefficients; positive favors two terms. RMSE = sqrt(RSS/m), in nats. These are N-only comparisons; any joint law in this project remains additive Chinchilla $E+A/N^\alpha+B/D^\beta$.",
        "\nReverse-ladder residual = observed loss minus the 1-term prediction fitted on present sizes with tied N ≥ 100M (at least four required). POSITIVE requires the 5M residual >0, at least 75% of sizes below 100M positive, and their mean >+0.02 nats; NEGATIVE is the mirror; otherwise FLAT. The reverse-ladder signature is a companion result and does not alter detection. Tests overlap across counts and slices, are correlated, and have no familywise multiplicity adjustment; the preregistered aggregation rule determines the verdict. Non-detection does not prove absence of a second term. Interpolated checkpoints and seed-independent Gaussian bootstrap noise are modeling assumptions.",
        "\n## Figures",
        "\n![Every convention and count, annotated bootstrap p-values](figures/lrt_by_convention.png)",
        "\nFigure 1. Bootstrap p-values for all 114 tests; an asterisk marks detection at p ≤ 0.05. Color is −log10(p), with the finite-bootstrap floor 1/1001. Every plotted axis has a grid.",
        "\n![Reverse-ladder residuals under each parameter count](figures/reverse_20N_by_count.png)",
        "\nFigure 2. Reverse-ladder residuals at TS 20N, using the same held-out small architectures under three counts. The horizontal axis is tied size for alignment; fitting uses the count in the legend. Dashed vertical line marks the 100M tied reference threshold. The smallest-size residual changes sign when embeddings are excluded.",
        "\n## Complete table",
        "\nAll losses and residuals are nats. Every row has B = 1,000 completed refits and zero nonfinite refits. ΔBIC is BIC1 − BIC2. Full precision, separate BIC1/BIC2, T and all reverse residuals: [full_table.csv](results/full_table.csv). Fitted coefficients and cache metadata: [e4_cells.csv](results/e4_cells.csv). n/a means fewer than four reference sizes.\n",
        fulltable,
        "\nTLDR-end: [dark-matter: E4] The second N term is convention-dependent (26/114 detections); only four detections have the preregistered positive small-end reverse-ladder signature. All 114,000 grid refits and 80,000 control refits are complete, and verification passes.\n"]
    (HERE / "results.md").write_text("\n".join(lines))
    print(f"Wrote report/figures/full table: {overall}; detections {int(df.detect.sum())}/114; controls {null}/40 null, {positive}/40 positive")


if __name__ == "__main__":
    main()
