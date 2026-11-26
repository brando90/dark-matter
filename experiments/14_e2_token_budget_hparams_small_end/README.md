# E2 (2026-11-26): token budget and hyperparameters at the small end

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
