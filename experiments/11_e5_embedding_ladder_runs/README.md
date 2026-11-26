# 11_e5_embedding_ladder_runs: E5, DM-4 embedding ladder runs on Stanford clusters (2026-11-25)

Question: does the embedding size at fixed body size move the small-model loss, and what weight c on embedding parameters puts the
tiniest models back on the curve (N = N_nonemb + c N_emb)? Design, PREREG (with Amendments 1 and 2) and the trainer are in
`../dm4_embedding_ladder/` and `../dm2_real_ladders/PREREG.md`; this folder holds the runs' raw results and the PREREG analysis.

State, 2026-11-25 19:15 PST: **no run has finished yet.**

| phase | rows | host | job | state |
|---|---|---|---|---|
| 1 | 92 const rows of `runs.csv` (widths 128 to 320; each with its cooldown branches), about 68 L40S GPU-hours | Sherlock `gpu`, `-C GPU_SKU:L40S` | array 53727060 (20 pool tasks, 6 h each, `--requeue`; script `../dm4_embedding_ladder/sherlock/phase1_pool.sbatch`; rows claimed by `mkdir` in `$SCRATCH/dm4/phase1_claims`) | queued (Sherlock projects a start about 2026-11-26 13:00) |
| 2 | 20 rows (widths 384 to 704), about 95 L40S GPU-hours | Sherlock | not submitted: after phase 1 shows the gates hold (PREREG) | waiting |

Why a pool of 20 tasks: the Sherlock `gpu` QOS allows 100 submitted jobs per user and 79 were already queued. Training code on
Sherlock: `train.py` from this repo's main (md5 43ca0167094cfb0946e738c908c03744, Amendment 2 shuffle), data and tokenizers from the
phase-0 prep (`$SCRATCH/dm4/data`, sha256-checked shards, `bpt.json`).

Collected results go to `raw/phase<k>/run<idx>/result.json` (small files only, no checkpoints); the analysis (c per width, c vs width,
DM-3 two-term detection by the PREREG rule) goes to `results.md` with the harness and model named.
