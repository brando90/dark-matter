#!/bin/bash
#SBATCH -J dm4-p1
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_SKU:L40S
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH --array=0-19
#SBATCH --requeue
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=brandojazz@gmail.com,brando.science@gmail.com
#SBATCH -o logs/p1pool-%A_%a.log
# DM-4 phase 1 on Sherlock (E5, 2026-11-25): runs.csv phase 1, 92 const rows (each with its WSD cooldown branches), ~68 L40S GPU-h,
# G1 passed after Amendment 2. The gpu QOS allows 100 submitted jobs per user, so this is a POOL of 20 array tasks, each
# claiming rows (largest first, mkdir claim in $DM4/phase1_claims) until none are left or its 6 h end. A requeued task first
# resumes the row it claimed itself (train.py resumes from $OUT/ckpt.pt; a finished row exits at once). To fill gaps after the pool ends: resubmit; unfinished claims of dead tasks are
# listed by `ls $DM4/phase1_claims` against the finished rows.
# WIDTHS="288 320" (env, default all five) limits the pool to those widths (E5 split: the mercury2 pool takes 128, 160, 224).
# Submit from $SCRATCH/dm4:  mkdir -p logs phase1 phase1_claims && CODE=$PWD/code/experiments/dm4_embedding_ladder WIDTHS="288 320" sbatch --export=ALL --chdir=$PWD $CODE/sherlock/phase1_pool.sbatch
set -uo pipefail
source "$CODE/sherlock/env.sh"   # sbatch runs a spool copy: $CODE comes from the submit env
cd "$CODE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CL="$DM4/phase1_claims"; mkdir -p "$CL"
ORDER=$(python - <<'PY'
import csv, os
rows = [r for r in csv.DictReader(open("runs.csv")) if r["phase"] == "1" and r["run"] == "const"]
assert len(rows) == 92, len(rows)
ws = os.environ.get("WIDTHS", "128 160 224 288 320").split()
rows = [r for r in rows if r["width"] in ws]
rows.sort(key=lambda r: -float(r["gpuh_measured"]))
print(" ".join(r["array_index"] for r in rows))
PY
)
me="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"; mytag="task${SLURM_ARRAY_TASK_ID}"
runit() { python train.py --root "$DM4" --index "$1" --out "$DM4/phase1/run$1" --log-every 200 && echo P1_RUN_DONE "$1"; }
# 1) resume my own unfinished claim (after a requeue)
for idx in $ORDER; do
  if [ "$(cat "$CL/run$idx/owner" 2>/dev/null)" = "$mytag" ]; then runit "$idx" || exit 1; fi   # a finished row is a no-op
done
# 2) claim new rows
for idx in $ORDER; do
  if mkdir "$CL/run$idx" 2>/dev/null; then echo "$mytag" > "$CL/run$idx/owner"; echo "$me $(date -Is)" >> "$CL/run$idx/log"; runit "$idx" || { echo FAIL "$idx"; exit 1; }; fi
done
echo POOL_DONE "$me"
