# Do experiments 02, 03, 04 and 09 survive without the eleven Chinchilla runs under one token per parameter, and where do those runs come from? (11-27-2026)

**Doc link:** <https://github.com/brando90/dark-matter/blob/main/experiments/18_short_runs_with_without_exp02_03_04_09_start_27Nov2026/README.md>

**TLDR:** Folder 16 found that a few very short Chinchilla runs drive the held-out error and the token-exponent results of the Chinchilla fits. This folder re-runs experiments 02, 03, 04 and 09 unchanged, once on all 245 runs and once without the 11 runs under one token per parameter, and traces where those runs come from. Status: PREREGISTERED on 11-27-2026, before any of the re-runs. Results: pending.

Date: 11-27-2026. Owner: Claude Code, Claude Sonnet 5.5, CPU only.

## Hypothesis

The eleven Chinchilla runs with fewer than one token per parameter, and above all the four with fewer than a third of a token per parameter, carry most of the held-out error of the joint fits in experiments 02 and 09 and shape the in-sample and stability results of experiments 03 and 04. The extra-term models in those experiments, a second N term, a second D term and a second compute term, do not beat their base models in the committed outputs. The question is whether removing the eleven runs changes that.

## Method

1. A wrapper, `run_arm.py`, loads each original experiment script unchanged, replaces its `load_chinchilla_runs` by a version that keeps either all 245 runs (arm `all245`) or only runs with D/N of at least 1 (arm `drop_DN_lt_1`, 234 runs), points its output folder at `outputs/<experiment>/<arm>/`, and calls its `main()`. Nothing in experiments 02, 03, 04 or 09 is edited. The matched ladder of D/N near 20 (`chinchilla_matched`) is untouched in both arms, so every N-only result on that ladder must be identical between arms.
2. Harness check: arm `all245` must reproduce the committed outputs of each experiment. I report the largest relative difference; if it is above 1e-6 I say so and compare the arms against each other, in one environment, instead.
3. `summarize.py` builds the with-and-without tables for the Chinchilla rows only.
4. `trace_short_runs.py` groups the 245 runs by the nearest of Hoffmann et al.'s nine IsoFLOP budgets, lists where the eleven short runs sit in their slices, and checks the dataset's stated source. It does not and cannot check whether any single run was really trained; the original run logs are not public.

Every extra term in the re-run experiments is additive.

## Pre-registered predictions

Written on 11-27-2026 after reading the committed outputs of the all-runs arm and before running the without arm. A prediction that fails is reported as failed.

1. Experiments 02 and 09, size holdout, Chinchilla runs: removing the eleven runs lowers the joint model's held-out RMSE, 0.250 in the committed runs of both, by at least half.
2. Experiments 02 and 09: no extra-term model beats its base model by more than 2% in relative held-out RMSE, in either arm. The models are `joint_extra_n` against `joint` and `compute_two` against `compute_one` in experiment 02, and `joint_extra_n` and `joint_extra_d` against `joint` in experiment 09.
3. Experiment 03: the quadratic R-squared of the joint residuals against log D, 0.320 with all 245 runs, falls below 0.10 without the eleven runs, and the Spearman correlations of the residuals with log N and with log D stay below 0.2 in absolute value in that arm.
4. Experiment 04: the best joint fit's token exponent gamma, 0.4557 on all 245 runs, falls by at least 0.03 without the eleven runs, toward the published 0.36.
5. Harness: arm `all245` reproduces every committed key number to 1e-6 relative.
6. Provenance: ten of the eleven runs under one token per parameter lie at the large-N end of the slice nearest the 1e19 FLOP budget, and the eleventh is the 12.57 billion parameter run of the 6e20 slice. At a fixed budget, D/N falls as 1/N squared, so the shortest runs are where an IsoFLOP design puts its largest models.

## Result

Pending.

## Files

1. `run_arm.py` and `run_all.py`: the wrapper and the driver that runs the eight arms on two workers.
2. `summarize.py`, `report_numbers.py`, `check_results_md.py`: tables, the numbers file and its provenance check.
3. `trace_short_runs.py`: the provenance trace.
4. `outputs/<experiment>/<arm>/`: the unchanged scripts' own CSV outputs for each arm.

## Rerun

```bash
source "$HOME/uv_envs/dark-matter2/bin/activate"
python experiments/18_short_runs_with_without_exp02_03_04_09_start_27Nov2026/run_all.py --workers 2
python experiments/18_short_runs_with_without_exp02_03_04_09_start_27Nov2026/summarize.py
python experiments/18_short_runs_with_without_exp02_03_04_09_start_27Nov2026/trace_short_runs.py
```
