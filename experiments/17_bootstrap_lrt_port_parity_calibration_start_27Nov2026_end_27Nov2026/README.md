# Porting the bootstrap likelihood-ratio test into lr_test, and making fit_two_term_N_only fail loudly (11-27-2026)

**Doc link:** <https://github.com/brando90/dark-matter/blob/main/experiments/17_bootstrap_lrt_port_parity_calibration_start_27Nov2026_end_27Nov2026/README.md>

**TLDR:** The two-term test in `py_src/scaling_laws_fit.py` could never reject, so I replaced it with the calibrated bootstrap test the repo already had in `experiments/dm1_power_analysis/`, and made the two-term fitter raise instead of quietly fitting other data. The port reproduces 220 of 220 committed DM-1b rows exactly, rejects 33 of 600 additive-noise null ladders (0.055) and finds a strong second term in 40 of 40. Post hoc, with multiplicative noise and a wide loss range it rejects 0.115 of null ladders, about twice its nominal level.

Date: 11-27-2026. Status: COMPLETE. Owner: Claude Code, Claude Sonnet 5.5, CPU only.

## Hypothesis

1. The new `lr_test(N, L)` in `py_src/scaling_laws_fit.py` is the DM-1b test, bit for bit, because its fitter block is copied from `experiments/dm1_power_analysis/dm1b.py` and only three names change.
2. Called through the public function with fresh seeds, it keeps DM-1b's calibration, about five percent false positives at level 0.05, and it detects a strong second term.
3. `fit_two_term_N_only` and `lr_test` now raise on an empty, too small, non-finite or non-positive selection. Before this change the repo's own smoke test selected zero points and silently fitted all four hundred points of a grid whose data size also varied, and then reported a statistic of zero.

## Method

1. Source identity: re-derive the fitter block from `dm1b.py`, apply the three renames, and compare it with the block between the BEGIN and END markers in `py_src/two_term_lrt.py`.
2. Row parity: regenerate DM-1b replicates from their seeds, run the ported test, and compare with the rows committed in `experiments/dm1_power_analysis/results_dm1b/raw_dm1b_shard000_of001.csv`. The sample is drawn with seed 20261127: forty rows for each planted second-term size s in 0, 0.03, 0.1, 0.3 and 1.0 from ladders of at most seventeen points, plus twenty rows from ladders of more than seventeen points.
3. Fresh-seed calibration through the public `lr_test`: two hundred one-term ladders in each of three DM-1 cells (1e5 to 1e9 and 1e6 to 1e9 at four points per decade with noise sd 0.01, and 1e7 to 1e9 at four per decade with noise sd 0.02), with B = 200 and DM-1b's exact early stop. Seeds are `[cell, rep, 20261127]` for the data and the same list with a trailing 1 for the bootstrap.
4. Fresh-seed power: forty ladders of DM-1b's positive control (1e5 to 1e9, eight per decade, sd 0.005, second term of relative size s = 1 with DM-1's beta = 1) and forty of a weak cell (1e7 to 1e9, four per decade, sd 0.01, s = 0.1) that DM-1b says the data cannot resolve.
5. Loud failure: `test_fail_loud.py`, the repo smoke test, and `experiments/06_naming_convention/check_naming.py`.

Every extra term in this folder is additive. The fitter block uses the additive form `E + a n^-p + b n^-(p+d)` on the loss itself, never a product of powers.

## Pre-registered acceptance criteria

Written on 11-27-2026 before `parity_dm1b.py` or `calibration_power.py` was run. A failed criterion is reported as failed. Criterion 5 is the exception: `test_fail_loud.py` was written and run against the code during development, before this commit, and passed (17 of 17), so that criterion is not prospective.

1. Source identity: the two blocks are identical after the three renames.
2. Row parity: on all sampled rows `k_exceed`, the number of refits used and the detect decision match exactly, T matches to 1e-9 relative, and p to 1e-12.
3. Null calibration: the pooled rejection rate over the six hundred null ladders is at most 0.07, and every cell is at most 0.10, which is DM-1b's own worst-cell gate. DM-1b measured 0.049 pooled and 0.070 in its worst cell.
4. Power: at least thirty-six of the forty positive-control ladders are detected. The weak cell has no threshold and is reported as it comes out.
5. Loud failure: every test in `test_fail_loud.py` passes, `python py_src/scaling_laws_fit.py` exits zero and fits a twenty-point ladder, and `check_naming.py` exits zero.

## Result

All five criteria pass. Numbers are in [results.md](results.md).

1. The fitter block is identical to `dm1b.py`'s after three renames, and 220 of 220 sampled committed rows reproduce every decision field, with the largest relative difference in T at 1.8e-16.
2. On six hundred fresh one-term ladders with additive noise the public `lr_test` rejects 33 times, a rate of 0.055; the worst of the three cells is 0.085, inside the 0.1 cell gate. It detects DM-1b's positive control in 40 of 40 ladders and the weak cell in 2 of 40, as DM-1b predicts.
3. Post hoc, not preregistered: with log-normal multiplicative noise at relative sd 0.005 the pooled null rejection is 0.115, and the rate rises with the loss range, 0.05 for losses of about 2 to 3.4, 0.12 for 2 to 5.5 and 0.175 for 2 to 10. The test assumes additive noise of constant size, so on real ladders a detection is weaker evidence than its five percent level suggests.
4. All seventeen fail-loud tests pass. The repo smoke test now fits a genuine twenty-point N-ladder; it used to select zero points and silently fit all four hundred points of a grid whose data size also varied.

## What would move this next

1. Add a multiplicative-noise version of the bootstrap, resampling log residuals, and check its null calibration across loss ranges before using the test on Porian or Chinchilla ladders again.
2. Re-run the folder 15 detections with that version and compare the 26 of 114 count.

## Files

1. `parity_dm1b.py`: criteria 1 and 2. Writes `results/parity.csv` and `results/parity_summary.json`.
2. `calibration_power.py`: criteria 3 and 4. Writes `results/calibration.csv` and `results/calibration_summary.json`.
3. `test_fail_loud.py`: criterion 5, runnable with pytest or directly.
4. `py_src/two_term_lrt.py` and `py_src/scaling_laws_fit.py`: the library change under test.
5. `posthoc_multiplicative_noise.py`: the post hoc multiplicative-noise check, not preregistered. Writes `results/posthoc_multiplicative.csv` and `results/posthoc_multiplicative_summary.json`.
6. `report_numbers.py`: writes `results/report_numbers.txt`, the numbered items that every number in `results.md` is traced to.
7. `check_results_md.py`: exits with an error if a number in a table row of `results.md` is missing from the items that row cites.

## Rerun

```bash
source "$HOME/uv_envs/dark-matter2/bin/activate"
python experiments/17_bootstrap_lrt_port_parity_calibration_start_27Nov2026_end_27Nov2026/parity_dm1b.py
python experiments/17_bootstrap_lrt_port_parity_calibration_start_27Nov2026_end_27Nov2026/calibration_power.py --workers 2
python -m pytest experiments/17_bootstrap_lrt_port_parity_calibration_start_27Nov2026_end_27Nov2026/test_fail_loud.py -q
python experiments/17_bootstrap_lrt_port_parity_calibration_start_27Nov2026_end_27Nov2026/posthoc_multiplicative_noise.py --workers 2
python3 -I experiments/17_bootstrap_lrt_port_parity_calibration_start_27Nov2026_end_27Nov2026/report_numbers.py > experiments/17_bootstrap_lrt_port_parity_calibration_start_27Nov2026_end_27Nov2026/results/report_numbers.txt
python3 -I experiments/17_bootstrap_lrt_port_parity_calibration_start_27Nov2026_end_27Nov2026/check_results_md.py
```
