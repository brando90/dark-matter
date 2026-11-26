# What still needs a GPU

Experiments 01–10 are fits to public ladders. They ran on CPU. None of them needs SNAP, Sherlock, or FarmShare.

The question they cannot answer is whether a fast term appears when embedding size changes at fixed body size. That needs new training runs. The trainer is already `experiments/dm4_embedding_ladder/train.py`. It reads one row of `runs.csv` and trains one model on one GPU. The tokenized data root is `$DM4`, the same root the Sherlock scripts use. If that root is not already on skampere2, this command will not find the data.

The smallest const cell is width 128, depth 4, 8192 tied, three seeds. `runs.csv` indices are 0 (seed 0), 6 (seed 1), and 7 (seed 2). Each is one process. GPUs 4, 5, and 6 take the three seeds. GPU 7 stays free.

```bash
ssh skampere2
cd ~/dark-matter
export DM4=${DM4:-$HOME/dm4}
mkdir -p "$DM4/snap_tiny" "$DM4/logs"
# indices: seed 0 -> GPU 4, seed 1 -> GPU 5, seed 2 -> GPU 6
for pair in "0:4" "6:5" "7:6"; do
  idx=${pair%%:*}
  gpu=${pair##*:}
  CUDA_VISIBLE_DEVICES=$gpu python experiments/dm4_embedding_ladder/train.py \
    --root "$DM4" \
    --runs-csv experiments/dm4_embedding_ladder/runs.csv \
    --index "$idx" \
    --out "$DM4/snap_tiny/run${idx}" \
    --log-every 200 \
    > "$DM4/logs/snap_run${idx}.log" 2>&1 &
done
wait
```

Sherlock phase 1 for the wider rows is already queued as job 53727060 (`experiments/dm4_embedding_ladder/sherlock/phase1_pool.sbatch`). This SNAP command does not replace that job. It only runs the three smallest const seeds on the free skampere2 GPUs.
