# AGENTS.md

Shared engineering and research guidelines for AI coding agents working in `dark-matter`.

## 1. Project Overview

The **Dark Matter** project investigates higher-order / "missing" terms in neural scaling laws beyond single-slope power laws, aiming to extrapolate large-model performance more accurately from smaller runs.

- **Canonical joint scaling law (Chinchilla / Hoffmann et al., Besiroglu et al. arXiv:2404.10102)**:
  `L(N, D) = E + A / N^α + B / D^β`
  Fitted with the Huber-in-log-space `logsumexp` objective (`fit_joint_powerlaw` in `py_src/scaling_laws_fit.py`, `delta=1e-3`, `L-BFGS-B`), assuming compute `C ≈ 6 * N * D` and optimal allocation exponent `a = β / (α + β)`.
- **Two-term N-only conjecture**:
  `L(N) = E + A / N^α + B / N^β` with `β < α` in the code's naming (`fit_two_term_N_only`; the same law as the README's β > α with the two names swapped: always say which term is the fast one, the one that matters at small N), tested against a 1-term baseline via the Huber likelihood-ratio test (`lr_test`).
- **Non-negotiable rule**: **Never** replace the additive Chinchilla form `E + A/N^α + B/D^β` with a multiplicative `A * N^(-α) * D^(-β)` form.

## 2. Environment & Common Commands

- **Python**: `>=3.11,<3.12` managed with `uv` (see `pyproject.toml` and `uv.lock`).
- **Activate standard venv**:
  ```bash
  source "$HOME/uv_envs/dark-matter2/bin/activate"
  uv sync --active
  ```
- **Synthetic smoke test** (joint fit, two-term N-only fit, and LR test):
  ```bash
  python py_src/scaling_laws_fit.py
  ```
- **Run scaling-law fit on a CSV**:
  ```bash
  python experiments/00_prepare_data_for_scaling_law_fitting/main_fit_scalings_laws.py \
    --data_csv <path/to/data.csv> \
    --plots_suffix <name>
  ```
- **Batch runner** (checks `$SCALING_DATA_CSV` / `$1` / default candidates):
  ```bash
  ./experiments/00_prepare_data_for_scaling_law_fitting/main_fit_scalings_laws.sh
  ```
- **Prepare Porian et al. (Resolving Discrepancies) dataset**:
  ```bash
  python experiments/00_prepare_data_for_scaling_law_fitting/00_get_resolving_discrepencies_in_scaling_laws_data.py
  ```
- **Platform dependencies**: Keep CUDA/Linux-only packages (`vllm`, `bitsandbytes`, `flash-attn`, `nvidia-htop`, `unsloth`, `unsloth_zoo`) gated with `platform_system != 'Darwin'` in `pyproject.toml` so macOS environments stay installable.

## 3. Repository Structure

- `py_src/` — Reusable Python modules:
  - `scaling_laws_fit.py`: `JointParams`, `TwoTermParams`, `fit_joint_powerlaw`, `fit_two_term_N_only`, `lr_test`, `compute_optimal_from_joint`.
  - `chinchilla_verification.py`: `run_chinchilla_verification`, synthetic Chinchilla generator, residual/compute-loss/IsoFLOP plotting utilities.
- `experiments/00_prepare_data_for_scaling_law_fitting/` — Experiment scripts, shell launcher, reference papers (`research_papers/`), and generated outputs (`plots_out_*/`).
- `data/data_resolving_discrepancies/` — Porian et al. (arXiv:2406.19146) dataset (`experiment_results.pickle.xz`) and `EXPERIMENTS_RESULTS_SUMMARY.md`.
- `data/data_hitchhikers_guide/` — 66 compressed `.csv.gz` ColPret / Hitchhiker's Guide (arXiv:2410.11840) datasets.

Put reusable fitting, data, and plotting utilities in `py_src/`; keep experiment-specific runners and plot outputs in numbered `experiments/XX_.../` folders.

## 4. Data Schemas & Git Hygiene

- `.gitignore` ignores uncompressed `data/**/*.csv` while tracking `!data/**/*.csv.gz` (and `.pickle.xz`). **Never** commit large uncompressed CSVs or JSON exports (e.g., `experiment_results.json` ~72 MB).
- Supported input CSV schemas in `run_chinchilla_verification`:
  1. Canonical: `N_params`, `tokens_D`, `loss` (optional `flops_C`, auto-computed as `6.0 * N_params * tokens_D`).
  2. Resolving-discrepancies format: `width`, `depth`, `max_step`, `seq_len`, `bs`, `final_val_loss`.

## 5. Known Legacy Quirks & Pitfalls

- This repository was migrated from `stair-lab/dark-matter2` into `dark-matter`. Legacy files (`main_fit_scalings_laws.sh`, `00_get_resolving_discrepencies_in_scaling_laws_data.py`, `dark-matter2.code-workspace`, `README.md`) still reference `$HOME/dark-matter2` and `$HOME/uv_envs/dark-matter2`. In **new** scripts, resolve paths relative to `Path(__file__).resolve()` / repo root instead of hardcoding `$HOME/dark-matter2`.
- `pyproject.toml` lists `[tool.hatch.build.targets.wheel] packages = ["py_src/dark_matter2"]`, while modules currently live directly in `py_src/`. Experiment scripts add `REPO_ROOT` to `sys.path` and import `from py_src...`.
- `py_src/chinchilla_verification.py` derives `N_params = width * depth` for resolving-discrepancies CSVs—this is a rough proxy, not an exact transformer parameter count. Verify/update before interpreting absolute `A` or `D/N` values from that schema.
- `experiments/00_prepare_data_for_scaling_law_fitting/README.md` still references the older script name `main_chinchilla_verification.py` (current script is `main_fit_scalings_laws.py`).

## 6. Verification & Scientific Rigor

- **Never fabricate or extrapolate numbers, metrics, or file paths.** Run the fitting script and inspect actual outputs.
- Check `verification.log`, fitted parameters (`E, A, α, B, β`), `R²`, `RMSE`, `README_plots.md`, and generated plots (`residuals.png`, `compute_loss_offset_vs_nooffset.png`, `isoflops_model_size_vs_compute.png`, `isoflops_dataset_size_vs_compute.png`).
- **Do not** enforce Besiroglu/Chinchilla target coefficients (`E≈1.82, A≈482, α≈0.34, B≈2085, β≈0.36, D/N≈20`) on non-Chinchilla empirical datasets; those target checks apply only to synthetic Chinchilla verification.
