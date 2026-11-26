# Migration check: stair-lab/dark-matter2 and brando90/dark-matter2 into brando90/dark-matter (11-25-2026)

Checked 11-25-2026 18:50 PST by ub9sc (the assistant ultimate_brando9), AT-3604. Harness: git and the GitHub API, Claude Code agent, Sonnet 5.5.

## What each repository held
| source | what it held | now in dark-matter main |
|---|---|---|
| stair-lab/dark-matter2 (PUBLIC, not private) main a277c3e | 23 commits, all by brando90; code, data, README | yes: the `Copy contents of stair-lab/dark-matter2` commit (02af148); `diff` of py_src, data, pyproject, uv.lock against it is empty apart from the two files Grok added in 1916055 |
| stair-lab/dark-matter2 issues #1, #2 | 2 open tasks (scaling ladders list; why alpha < beta) | copied to `docs/legacy_issues_stair_lab_dark_matter2.md` and opened as issues here |
| stair-lab/dark-matter2 discussions | one auto-generated welcome post (#3) | nothing to keep |
| stair-lab/dark-matter2 contributors, forks | only brando90; one fork (brando90/dark-matter2) | no other person's commits exist there |
| brando90/dark-matter2 main b785286 | PRs #1 to #5 merged: README from grant, DM-1 power analysis, DM-1b bootstrap LRT, DM-2/DM-3 real ladders, DM-5 small-to-large | yes: merged with full history (merge fe03d0c) |
| brando90/dark-matter2 branch dm2-real-ladders | 4 commits not in its own main (DM-4 phase 1 result rows w128, w160, w224); PR #6 still open there | yes: merged (d8f16dd) |
| brando90/dark-matter2 branches dm1-power-analysis, dm1b-lrt, dm5-small-to-large, readme-from-grant | all already in its main | yes |

Check run: for every remote branch of brando90/dark-matter2, `git rev-list HEAD..<branch>` is 0 in dark-matter main after the merge.

## Folder map (names kept as in dark-matter2, so old links and scripts work)
- `experiments/dm1_power_analysis/` DM-1 and DM-1b
- `experiments/dm2_real_ladders/` DM-2 and DM-3 (profiled c on Porian)
- `experiments/dm4_embedding_ladder/` DM-4 design, phase 0, G1 gate, phase 1 rows so far
- `experiments/dm5_small_to_large/` DM-5
- `experiments/01_backtest_single_vs_two_term/` Grok, 11-25-2026
- New numbered folders from 02 on (see `experiments/README.md`).

## Left for Brando
Deleting stair-lab/dark-matter2 is Brando's own step (the GitHub token used here has no admin right there). The brando90/dark-matter2 fork can be deleted or archived by Brando at the same time; PR #6 there has its content in dark-matter already.
