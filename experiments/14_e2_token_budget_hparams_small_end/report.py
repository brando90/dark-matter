"""Build E2 tables, scientific figures, and honest gate-limited results (2026-11-26)."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import e2
import e2lib as L
import cache_io as CACHE

HERE = Path(__file__).resolve().parent
RES, FIG = HERE/'results', HERE/'figures'
SIZES = ['5M', '9M', '15M']


def number(x, signed=False):
    if not np.isfinite(x):
        return 'unavailable'
    return f'{x:+.3f}' if signed else f'{x:.3f}'


def interval(row, key='C'):
    if not np.isfinite(row[key]):
        return 'unavailable'
    return f"{number(row[key], True)} [{number(row['lo'], True)}, {number(row['hi'], True)}]"


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIG/f'{name}.png', dpi=180, bbox_inches='tight')
    fig.savefig(FIG/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    x = pd.read_csv(RES/'e2_excess_tied.csv')
    f = pd.read_csv(RES/'e2_factors_tied.csv')
    m = pd.read_csv(RES/'e2_matched_tied.csv')
    j = pd.read_csv(RES/'e2_joint_tied.csv')
    controls = pd.read_csv(RES/'verify_controls.csv')
    status = json.loads((RES/'verify.json').read_text())
    ft = f[f['size'].isin(SIZES)].copy()
    mt = m[m['size'].isin(SIZES)].copy()
    table = []
    for ds in ['rw','owt2']:
        rows = mt[mt.dataset == ds]
        returns = int((rows.verdict == 'RETURNS').sum())
        stays = int((rows.verdict == 'STAYS OFF').sum())
        v = 'CONFOUND EXPLAINS' if returns == 3 else 'CONFOUND DOES NOT EXPLAIN IT' if stays >= 2 else 'PARTIAL'
        table.append(dict(dataset=ds, returns=returns, stays_off=stays, targets=3,
                          conditional_verdict=v, scientific_gate_pass=status['scientific_gate_pass']))
    pd.DataFrame(table).to_csv(RES/'e2_dataset_verdicts.csv', index=False)

    metrics = []
    for count in ['tied','untied']:
        for ds in ['rw','owt2']:
            for name in list(e2.COND) + ['joint_TC','joint_BC']:
                q = CACHE.read(RES/'bootstrap'/count/ds/f'{name}.npz')['result']
                if q is None:
                    continue
                joint = name.startswith('joint')
                n, k = q['npts'] if joint else q['nref'], 5 if joint else 3
                rss = q['fit']['rss']
                metrics.append(dict(count=count,dataset=ds,condition=name,n=n,k=k,RSS=rss,
                    RMSE_nats=np.sqrt(rss/n),BIC=L.F.bic(rss,n,k),B=q['B']))
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(RES/'e2_fit_metrics.csv',index=False)

    fig, axes = plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,ds in zip(axes,['rw','owt2']):
        for offset,cond,label,color in [(-.18,'BC20','Base + cosine at 20N','tab:orange'),(0,'TC20','Tuned + constant at 20N','tab:blue')]:
            a = x[(x.dataset==ds)&(x.cond==cond)].set_index('size').loc[SIZES]
            xx=np.arange(3)+offset
            ax.errorbar(xx,a.X,yerr=np.maximum(np.array([a.X-a.lo,a.hi-a.X]),0),fmt='o',capsize=3,label=label,color=color)
        a=mt[mt.dataset==ds].set_index('size').loc[SIZES]
        valid=np.isfinite(a.M)
        ax.errorbar(np.arange(3)[valid]+.18,a.M[valid],yerr=np.maximum(np.array([a.M[valid]-a.lo[valid],a.hi[valid]-a.M[valid]]),0),fmt='s',capsize=3,label='Corrected matched excess M',color='tab:green')
        ax.axhline(0,color='black',lw=.8)
        ax.set_xticks(range(3),SIZES);ax.set_title(ds);ax.set_xlabel('Porian model-size label');ax.grid(True)
    axes[0].set_ylabel('Small-end excess (nats; 95% bootstrap interval)')
    axes[1].legend(fontsize=8,loc='lower right')
    fig.suptitle('E2: corrected excess is uncertain; statistical controls fail')
    savefig(fig,'small_end_excess')

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,ds in zip(axes,['rw','owt2']):
        for offset,prefix,label,color in [(-.2,'J (TC','J: joint data term','tab:purple'),(0,'R (','R: base + cosine to tuned + constant','tab:blue'),(.2,'H (','H: sweep hyperparameters','tab:orange')]:
            a=ft[(ft.dataset==ds)&ft.factor.str.startswith(prefix)].set_index('size').reindex(SIZES)
            valid=a.C.notna()
            ax.errorbar(np.arange(3)[valid]+offset,a.C[valid],yerr=np.maximum(np.array([a.C[valid]-a.lo[valid],a.hi[valid]-a.C[valid]]),0),fmt='o',capsize=3,label=label,color=color)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(3),SIZES)
        ax.set_title(ds);ax.set_xlabel('Porian model-size label');ax.grid(True)
    axes[0].set_ylabel('Factor contribution (nats; 95% bootstrap interval)')
    axes[1].legend(fontsize=8)
    fig.suptitle('E2: factor classifications are conditional on failed control gates')
    savefig(fig,'factor_contributions')

    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,ds in zip(axes,['rw','owt2']):
        lad=L.ladder(ds,'TS',20)
        f1=CACHE.read(RES/'bootstrap'/'tied'/ds/'TC20.npz')['result']['fit']
        fj=CACHE.read(RES/'bootstrap'/'tied'/ds/'joint_TC.npz')['result']['fit']
        n=np.geomspace(lad.N_tied.min()/1e6,lad.N_tied.max()/1e6,200)
        ax.scatter(lad.N_tied/1e6,lad.y,label='Tuned + constant observations',color='black',s=24)
        ax.plot(n,L.pred1(f1,n),label='N-only fit on reference sizes')
        ax.plot(n,L.F.predJ(fj,n,.02*n),label='Joint fit on reference sizes')
        ax.axvline(100,ls=':',color='gray',label='Reference-size cutoff')
        ax.set_xscale('log');ax.set_xlabel('Tied parameters (millions)');ax.set_title(ds);ax.grid(True)
    axes[0].set_ylabel('Validation loss at 20N (nats)');axes[1].legend(fontsize=8)
    savefig(fig,'reference_extrapolation')

    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,fast,title in zip(axes,[0,1],['Data-term-only control','Planted +0.3 nat fast term at 5M']):
        a=controls[controls.fast==fast]
        ax.errorbar(a.rep,a.XJ,yerr=np.maximum(np.array([a.XJ-a.M_lo,a.M_hi-a.XJ]),0),fmt='o',capsize=2)
        ax.axhline(0,color='black',lw=.8);ax.axhspan(-.006,.006,color='tab:green',alpha=.2,label='Return magnitude threshold ±0.006')
        ax.set_xlabel('Fixed replicate');ax.set_title(title);ax.grid(True)
    axes[0].set_ylabel('Joint excess at 5M (nats; 95% interval)');axes[1].legend(fontsize=8)
    savefig(fig,'fixed_controls')

    p,n=controls[controls.fast==0],controls[controls.fast==1]
    pj,pm=int((p.vJ=='EXPLAINS').sum()),int((p.vM=='RETURNS').sum())
    nj,nm=int((n.vJ!='EXPLAINS').sum()),int((n.vM=='STAYS OFF').sum())
    text = [
'# E2 Results (2026-11-26): token budget, training recipe, and the small end',
'', 'As of 2026-11-26. By ub9sc (ultimate brando9 scalable coworker).', '',
'**Why do the tiniest models live off the scaling curve, and what correction brings them back?** E2 asks whether Porian\'s small-end misfit is a token-budget or hyperparameter confound.', '',
'**Result: scientifically unresolved.** The unchanged preregistered statistical control gates fail after the implementation corrections. The real-data table mechanically reads **(rw: PARTIAL, owt2: PARTIAL)**; these are conditional table classifications, not validated scientific verdicts. No target size meets the return-to-curve criterion. The base-plus-cosine to tuned-plus-constant recipe change reduces the reverse-ladder excess at all three target sizes on both datasets, but the additive joint-law extrapolation overshoots the small-model losses and has broad uncertainty.', '',
'Initial code/fits: Claude Code, Opus 5.5 (claude-opus-5-5). Resume, repairs, numerical reruns and reporting: Codex CLI, GPT-6 Astra ultra via Vals; Python CPU on skampere1, nice 15, three processes maximum with BLAS threads fixed to one. These are executed numerical fits and bootstrap calculations, not a model-only reading. Lean and differential testing are not applicable. Package versions and data hashes: [run_environment.json](results/run_environment.json).', '',
'PREREG was pushed before the first fit; [the dated amendment](AMENDMENT_2026-11-26.md) was pushed before the revised fits. The original PREREG is unchanged. Original failed controls and summaries remain in [original_2026-11-26](results/original_2026-11-26/).', '',
'## Return to the curve', '',
'Tied count is primary: N = N_nonemb + 50,432 × width (the legacy DM-2 loader calls this N_total). The input has **975 runs and 16 model sizes**, rather than the 977 stated in the brief. D/N always uses tied N, including the descriptive untied-count analysis. TC means tuned plus constant learning rate; BC means base plus cosine decay. The 20N reverse ladder fits five reference sizes, 108M–455M. Alpha is the fast exponent in the two-term N-only convention; E2 uses a one-term N law and the additive Chinchilla law only.', '',
'X is observed loss minus the N-only prediction from reference sizes. M is the joint-law excess, minus the sweep-estimated tuned-recipe correction G on rw. All intervals below are 95% parametric bootstrap intervals in nats. The rw 9M sweep correction is unavailable, so M is unavailable there; no G=0 substitution is made.', '',
'| Dataset | Size | X(TC20), nats [95% interval] | M, nats [95% interval] | Two seed SD, nats | Conditional return classification |',
'|---|---|---|---|---|---|']
    for r in mt.to_dict('records'):
        xx=x[(x.dataset==r['dataset'])&(x.cond=='TC20')&(x['size']==r['size'])].iloc[0]
        text.append(f"| {r['dataset']} | {r['size']} | {interval(xx, 'X')} | {interval(r, 'M')} | {r['two_sd']:.4f} | {r['verdict']} |")
    text += ['', 'RETURNS requires both |M| ≤ two seed SD and an interval containing zero; STAYS OFF requires M above two seed SD and a strictly positive lower bound. A negative overshoot outside the return threshold is UNDETERMINED under the fixed rule. The dataset criterion requires all three RETURNS for CONFOUND EXPLAINS, or at least two STAY OFF for CONFOUND DOES NOT EXPLAIN IT; all other outcomes are PARTIAL.', '',
'![Small-end excess](figures/small_end_excess.png)', '',
'## How much each factor explains', '',
'Positive contribution means a factor reduces the baseline excess. T compares per-size finals with matched 20N. J adds the joint data term. R changes base plus cosine to tuned plus constant. H estimates pure hyperparameter changes from the rw constant-learning-rate sweep. S = R − H includes schedule and remaining recipe differences. G estimates residual mis-tuning of the tuned recipe. These contributions overlap and must not be added as independent causal effects. H uses batch 64 at small sizes, while the actual base recipe uses batch 256; its attribution understates that batch mismatch. The available data do not independently cross tuned/base with decay/constant, so the schedule alone is not identified.', '',
'EXPLAINS requires contribution ≥50% of baseline, ≥two seed SD, and lower interval bound >0. PARTLY requires ≥20% of baseline and lower >0. OPPOSITE requires upper <0 and magnitude ≥20% of baseline. A nonpositive baseline gives N/A. A factor explains a dataset only when at least two of its three target-size rows are EXPLAINS. Every classification remains conditional because the controls fail.', '',
'| Dataset | Factor | Size | Contribution, nats [95% interval] | % of baseline | Conditional classification |',
'|---|---|---|---|---|---|']
    abbreviations={'T (TC: final -> 20N)':'T, tuned final → 20N','T (BC: final -> 20N)':'T, base final → 20N','T2 (BK final -> BC 20N, secondary)':'T2, Kaplan → base 20N (secondary)','J (TC: data term at 20N)':'J, tuned','J (BC: data term at 20N)':'J, base','R (recipe: base+cosine -> tuned+const)':'R, recipe','H (pure hparams, sweep)':'H, sweep hyperparameters','S (schedule and rest = R - H)':'S, schedule and rest','G (tuned recipe mis-tuning)':'G, tuned mis-tuning'}
    for r in ft.to_dict('records'):
        pct='N/A' if not np.isfinite(r['pct']) else f"{r['pct']:.1f}%"
        text.append(f"| {r['dataset']} | {abbreviations[r['factor']]} | {r['size']} | {interval(r)} | {pct} | {r['verdict']} |")
    for factor in ['H, sweep hyperparameters','S, schedule and rest','G, tuned mis-tuning']:
        text.append(f'| rw | {factor} | 9M | unavailable: no 9M sweep | unavailable | unavailable |')
    text += ['', 'The very large negative T estimates are unstable reverse extrapolations of heterogeneous per-size-final ladders. Their baseline excess is negative, so the preregistered classification is N/A. They are **not physical loss improvements of tens of nats**. The primary matched 20N recipe comparison is more interpretable. The tuned J estimate can exceed 100% of X while M still fails RETURNS because the joint prediction overshoots the observed loss.', '',
'![Factor contributions](figures/factor_contributions.png)', '',
'## Fit diagnostics and limitations', '',
'The additive joint form is L(N,D) = E + A/N^alpha + B/D^beta, with nonnegative coefficients. No multiplicative joint law is used. Reference-size TC joint residual SD is 0.019393 nats on rw and 0.019216 nats on owt2: both fall just below the preregistered 0.02-nat misfit threshold. Therefore the primary rule uses the joint prediction. The all-size secondary joint residual SD is 0.038513 and 0.041756 nats, respectively. This broadens the evidence against treating the additive joint fit as a sufficient explanation.', '',
'| Dataset | Reference fit | n | RMSE (nats) | BIC | Bootstrap draws |', '|---|---|---:|---:|---:|---:|']
    for r in metrics[(metrics['count']=='tied')&metrics.condition.isin(['TC20','BC20','joint_TC','joint_BC'])].itertuples():
        text.append(f'| {r.dataset} | {r.condition} | {r.n} | {r.RMSE_nats:.6f} | {r.BIC:.3f} | {r.B} |')
    text += ['', 'BIC is reported descriptively; fits with different observations cannot be compared by BIC as if they used the same response vector. E2 has no likelihood-ratio p-value: uncertainty is expressed by the preregistered percentile intervals. Complete diagnostics for all conditions and both counts are in [e2_fit_metrics.csv](results/e2_fit_metrics.csv). Secondary matched D/N=5 and 10, 7M values, and untied-count results are in the accompanying CSVs; secondary results carry no verdict.', '',
'Bootstrap assumptions: reference observations use fitted-model Gaussian noise with residual SD floored at 0.002 nats. Small observations and sweep losses use empirical tuned-recipe seed SD. Shared measurements reuse standardized innovations; fit-specific reference means/SD can differ. Distinct checkpoints of one training run are treated as separate observations, so checkpoint correlation is unmodeled. The sweep best-three statistic is reselected in each bootstrap draw and can have selection bias. The seed uncertainty model does not establish causal separation of schedule and hyperparameters.', '',
'![Reference-size extrapolation](figures/reference_extrapolation.png)', '',
'## Verification and completion state', '',
'| Suite | J condition | Matched-excess condition | Fixed requirement | Status |', '|---|---|---|---|---|',
'| Original positive | EXPLAINS 4/20 | RETURNS 17/20 | Both ≥16/20 | FAILED |',
'| Original planted fast term | not EXPLAINS 17/20 | STAYS OFF 10/20 | Both ≥16/20 | FAILED |',
f'| Amended positive | EXPLAINS {pj}/20 | RETURNS {pm}/20 | Both ≥16/20 | {"PASS" if pj>=16 and pm>=16 else "FAILED"} |',
f'| Amended planted fast term | not EXPLAINS {nj}/20 | STAYS OFF {nm}/20 | Both ≥16/20 | {"PASS" if nj>=16 and nm>=16 else "FAILED"} |', '',
'The fixed control truth is unchanged. Without observation noise, its data-only reverse excess is +0.199058 nats and joint excess is approximately zero. Adding the planted +0.3-nat fast term gives reverse excess +0.436106 and joint excess +0.203142 nats; the joint correction is +0.232964 nats, already 53.4% of the reverse excess. Thus the null requirement that J must not EXPLAIN is not guaranteed even by its noiseless planted truth. Noise and extrapolation uncertainty further impair the return classifications. The executed noiseless diagnostics are saved in [deterministic_diagnostic.csv](results/deterministic_diagnostic.csv). This diagnosis does not relax either gate.', '',
'**Software checks pass**: exact TC20/BC20 reproduction against the legacy curves, reverse residual reproduction (+0.347285 rw, +0.370665 owt2), every factor identity, persisted draw counts, shared observation innovations, and unavailable rw 9M correction. **The scientific control gates fail**; `verify.py` deliberately exits nonzero and says so. `verify.py --software-only` passes and explicitly does not validate the scientific verdict. Logs: [verify.txt](results/verify.txt), [verify_software.txt](results/verify_software.txt), [controls_run.log](results/controls_run.log).', '',
'All real N-only/sweep draws use B=2,000 and joint draws B=500, saved in [bootstrap](results/bootstrap/), with [bootstrap_manifest.csv](results/bootstrap_manifest.csv). Controls use 20 fixed replicates per truth, 300 N-only and 100 joint draws per replicate as in the original script. Caches and every-25-fit joint checkpoints allow resuming; output files identify incomplete work.', '',
'![Fixed synthetic controls](figures/fixed_controls.png)', '',
'Numerical execution, artifacts, and software integrity checks are complete. The requested passing preregistered scientific verification could not be achieved with the fixed control design, and no validated E2 explanation is claimed. Any new control design or extrapolation estimator requires a new prospective experiment.', '',
'Term usage is fixed throughout: small-end excess X, factor contribution, corrected matched excess M, reference sizes, and scientific control gate. No terminology was varied for style.', '',
'TLDR-end: [dark-matter: E2] Recipe changes reduce the observed reverse-ladder excess; matched joint corrections do not establish return to the curve. The conditional table is PARTIAL/PARTIAL, and failed fixed controls leave the scientific answer unresolved.' ]
    (HERE/'results.md').write_text('\n'.join(text)+'\n')
    (HERE/'README.md').write_text('''# E2 (2026-11-26): token budget and hyperparameters at the small end

As of 2026-11-26. By ub9sc; initial Claude Code Opus 5.5, resume Codex CLI GPT-6 Astra ultra via Vals; Python CPU on skampere1.

Why do the tiniest models live off the scaling curve, and what correction brings them back? E2 tests whether Porian's small-end excess is explained by token budget, training recipe, or the additive joint data term.

**Scientific status: unresolved.** The conditional PREREG table is PARTIAL on rw and owt2. Original and corrected fixed statistical controls fail; software integrity checks pass. See [results.md](results.md), [PREREG.md](PREREG.md), and the [dated amendment](AMENDMENT_2026-11-26.md).

Run from this directory with the project Python environment (NumPy, SciPy, pandas, matplotlib):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
nice -n 15 python e2.py --count both
nice -n 15 python run_controls.py
nice -n 15 python verify.py --software-only
nice -n 15 python verify.py
nice -n 15 python report.py
```

`verify.py` intentionally exits 1 if the fixed scientific gates fail; software-only success does not override that failure. Every joint calculation saves a checkpoint after 25 fits, and completed draw arrays persist under `results/bootstrap/`. Original failed artifacts remain under `results/original_2026-11-26/`.

TLDR-end: [dark-matter: E2] The numerical analysis is complete; the scientific control gate fails, so no validated explanation is claimed.
''')
    print('Wrote results.md, README.md, four figures (PNG/PDF), metrics and dataset classification tables.')


if __name__ == '__main__':
    main()
