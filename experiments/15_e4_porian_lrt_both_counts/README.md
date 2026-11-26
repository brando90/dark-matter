# E4: Porian bootstrap LRT under every convention and count (2026-11-26)

Date: 2026-11-26. By ub9sc (ultimate brando9 scalable coworker).

**Why do the tiniest models live off the scaling curve, and what correction brings them back?** E4 tests whether a second N term is detected under every Porian training convention and parameter count. The hypothesis, data selection, exact detection rule and table interpretation were fixed in [PREREG.md](PREREG.md) before fitting (commit d690c90).

The grid contains rw/owt2 × 19 recipe/slice combinations × three parameter counts, with 1,000 parametric-bootstrap refits per test. Alpha is the fast exponent, beta the slow exponent. The N-only two-term model is additive; any joint model remains the additive Chinchilla form. The reverse-ladder signature is reported separately from the detection rule.

**Result: CONVENTION-DEPENDENT, 26/114 detections.** Nonemb: 1/38 (NO); tied: 13/38 (MIXED); untied: 12/38 (MIXED). Only four detections also have the preregistered POSITIVE reverse-ladder signature. All 114,000 grid refits and 80,000 control refits are complete; verification passes (0/40 null detections, 40/40 positive detections).

Read [results.md](results.md) for the verdict, numbers, controls, caveats and figures. [results/full_table.csv](results/full_table.csv) is the complete convention/count table with RMSE, separate BIC values, bootstrap p-values and residuals at 5M, 7M, 9M and 15M. [results/e4_cells.csv](results/e4_cells.csv) also includes fitted coefficients and cache metadata.

Original fits: Claude Code, Opus 5.5. Resume and verification: Codex CLI, GPT-6 Astra ultra, Vals plan; Python CPU fits on skampere1. [Environment and input hashes](results/run_environment.json).

From the repository root, with the project environment active:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python experiments/15_e4_porian_lrt_both_counts/e4.py --workers 3 > experiments/15_e4_porian_lrt_both_counts/results/e4_resume.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python experiments/15_e4_porian_lrt_both_counts/verify.py --workers 3 > experiments/15_e4_porian_lrt_both_counts/results/verify.log 2>&1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 python experiments/15_e4_porian_lrt_both_counts/report.py > experiments/15_e4_porian_lrt_both_counts/results/report.log 2>&1
```

`e4.py` imports existing E2/E1/DM-1b/DM-2 helpers without editing them. Bootstrap progress is saved after each 100 refits; completed outputs are checked against data/configuration hashes before reuse. The initial run skipped refitting 74 zero-T cases; the resumed run completes those refits while preserving their p = 1 decisions and the original log. Verification controls are resumable per synthetic ladder and per 100 bootstrap refits.
