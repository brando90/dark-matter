# E1 PREREG (11-25-2026): profiled parameter count N_c = N_nonemb + c N_emb on every ladder with embedding counts

Written 11-25-2026 by ub9sc (Claude Code, Opus 5.5, on skampere2), committed and pushed to brando90/dark-matter main
BEFORE any E1 fit. Before this commit only the data's structure was opened (which sizes and columns exist, the
architecture tables) and the fitters were timed on one Porian ladder (no profile, no c, no residual looked at). The
DM-3 result on Porian (experiments/dm2_real_ladders/RESULTS.md) is known, so Porian is not blind.

## Dark-matter question
Why do the tiniest language models sit off the scaling-law curve? Candidate correction (a): count parameters properly,
N_c = N_nonemb + c * N_emb. Does ONE c bring the tiny models back onto the single power law, and is c the same on
every ladder?

Law convention (experiments/README.md): L(N) = E + A N^-alpha + B N^-beta with beta < alpha; alpha is the FAST exponent
(dominates at small N). The 1-term law is L = E + A N^-p. The fitter code (DM-1b `dm1b.fit_batch`) names the fast
exponent `beta` and the slow one `alpha`; every table here is translated to alpha = fast, beta = slow.

## Ladders (one loss y per size; y in nats; n = N_c / 1e6)
Unit of c: one V x d block (vocab x width), the DM-3 convention ("c per 50,432 w", DM-4 Amendment 1). N_emb := V d.
| id | y | sizes | N_nonemb | N_emb = V d | what c = 1 and c = 2 mean |
|---|---|---|---|---|---|
| porian-rw, porian-owt2 | DM-2 `results/curves.csv`, tuned-short-const, val loss at D = 20 N (N = DM-2 N_total) | 14 each (5M-455M) | DM-2 `porian_counts` | 50,432 w | c = 1: Porian's own N (head counted, input embedding not) = DM-2 N_total; c = 2: true untied total (DM-5 primary) |
| chinchilla | DM-2 `results/curves.csv`, chinchilla/hoffmann-epoch, loss at D = 20N (interpolated) | 28 | DM-2 arch table (layers x per-layer attn + FFN) | table "Embedding Parameters" (V = 32,168) | c = 1: DM-2 N_total (about Hoffmann's reported N) |
| pythia | Grok's `data/public_ladders/pythia_pile_val_final.csv.gz`: final Pile validation loss, every size at 2.999e11 tokens (FIXED D, not D = 20N) | 8 (70M-12B) | from the official HF configs (`data/pythia_hf_configs.csv`, fetched 11-25-2026): layers x (12 d^2 + 13 d) + 2 d (GPT-NeoX) | d x vocab_size of the config (50,304 / 50,432 / 50,688; untied) | c = 2: total parameter count |

- Pythia's counts are not in the repo's data; they are computed from the public architecture configs, and checked
  against the nominal sizes in the csv (|computed total / nominal - 1| reported; the 70M-12B names are rounded).
- **ColPret LLaMA (MAD) is dropped from E1**: its csv has num_params but no width, depth or vocab, so N_emb is not
  available. It is not invented.
- **Marin Delphi**: no Delphi points are on disk in this repo; not used.
- Standard count c_std (used by E3's baseline): porian 2, chinchilla 1, pythia 2.

## Fits
- Fitter: DM-1b's `fit_batch` (variable projection + compass search + projected LM; E, A, B >= 0), imported from
  experiments/dm5_small_to_large. A vectorized 1-term-only copy of its 1-term branch is used for the large bootstraps
  and is checked against `fit_batch`'s 1-term output (verify.py).
- Profile grid: c in {0, 0.01, ..., 3.00} (301 values) for RSS1(c) (1-term) and RSS2(c) (2-term); the minimum is
  refined with a bounded scalar search between the neighbouring grid values. c1 = argmin RSS1, c2 = argmin RSS2.
- 95% profile set: {c : n ln(RSS(c) / RSS_min) <= 3.841} on the grid, edges interpolated, reported as a union of
  intervals; "3.0+" marks a set that runs into the grid's upper end. Reported: excludes c = 1? excludes c = 0?
  Secondary (not in any verdict): RSS1 at c = 4, 5, 7, 10.
- Second-term test at each c: the DM-1b bootstrap LRT, T = n ln(RSS1/RSS2), B = 200 parametric refits under the fitted
  1-term law at that c (sigma^2 = RSS1/(n-3)), detect iff (1 + #{T* >= T}) / 201 <= 0.05; testable only when n > 6
  (all four ladders). Run on c in {0, 0.1, ..., 3.0} (31 values) and at c1, c2, c_std.
- Second-term test with c free: T_free = n ln(RSS1(c1) / RSS2(c2)), calibrated by B = 200 parametric refits from the
  1-term fit at c1, with c re-profiled in every refit (grid 0, 0.05, ..., 3.0 for both laws). Also
  dBIC_free = BIC(1-term, c free; k = 4) - BIC(2-term, c free; k = 6), positive favours 2-term.
- Residual by size (observed - fitted) for the 1-term law at c = 0, 1, c_std, c1 and at the shared c (below), and for
  the 2-term law at c2.

## Smallest-size residual ("does the tiny model return to the curve?")
At each c: r_min(c) = y_1 - yhat_1 for the 1-term law fitted on all sizes at that c (size 1 = smallest). Its noise
scale sd_boot(c): parametric bootstrap, B = 500, y* = 1-term fit at c + Gaussian noise with sd sigma2(c) =
sqrt(RSS2(c) / (n - 5)) (the 2-term fit's residual sd, so the 1-term misfit does not inflate the noise), 1-term refit
at the same c, sd of r*_min. "Within noise": |r_min(c)| <= 2 sd_boot(c). Evaluated on c in {0, 0.05, ..., 3.0}.
Secondary: the leave-smallest-out residual (1-term fitted on sizes 2..n at c, error at size 1) with its bootstrap sd
from the same simulation.

**5M signal**: the smallest model's excess survives non-embedding counting iff r_min(0) > 2 sd_boot(0) (positive
excess at c = 0). Reported per ladder, also at c_std.

## Pooled test: one shared c vs one c per ladder (1-term law)
Per-ladder profile log-likelihood (Gaussian, sigma per ladder profiled out): l_k(c) = n_k ln(RSS1_k(c) / n_k).
Shared: c_sh = argmin_c sum_k l_k(c); per-ladder: each ladder its own c1. LR statistic
Lambda = sum_k l_k(c_sh) - sum_k l_k(c1_k) >= 0 (df = K - 1). dBIC_shared = BIC_shared - BIC_per = Lambda - (K - 1)
ln(sum n_k); the shared model loses dBIC_shared BIC (positive = per-ladder c preferred). Bootstrap p-value: B = 200
parametric sets from each ladder's 1-term fit at c_sh (sigma_k^2 = RSS1_k(c_sh)/(n_k - 3)), Lambda* recomputed on
the 0.05 grid with refinement. Also reported (secondary, no bootstrap): the same dBIC for the 2-term law and pairwise
for the two Porian ladders alone. K = 4 ladders (porian-rw, porian-owt2, chinchilla, pythia).

## Reproduction check (DM-3, must pass before any E1 result is read)
On DM-3's grid (c in 0..2, step 0.005) and DM-3's fitter (`fitlib.fit1/fit2`): RSS1(c) equals
dm2_real_ladders/results/dm3_profileA.csv within 1e-6 relative at every grid c; c1 within 0.01 of DM-3 (rw 2.0,
owt2 0.045); the 95% set's interval edges within 0.02 of DM-3 (rw [0.05, 0.11] U [1.55, 2.0+], owt2 [0.03, 0.08] U
[1.83, 2.0+]); dBIC_free within 0.5 of +19.5 / +20.8. With the E1 fitter (fit_batch): RSS1 within 1e-4 relative of
DM-3 at every c. If any of these fails, E1 stops there and the report explains why.

## Verdict rules (fixed now)
- A ladder is **informative about c** if its 1-term 95% set covers < 90% of [0, 3] (< 2.7 in total length).
- **V1, one c.** Common set C = intersection of the 1-term 95% sets of the informative ladders (0.01 grid).
  SUPPORTED: >= 2 informative ladders, C non-empty AND dBIC_shared <= 6. NOT SUPPORTED: >= 2 informative ladders and
  (C empty OR dBIC_shared > 6). UNDETERMINED: fewer than 2 informative ladders.
- **V2, tiny models return to the curve.** If C is non-empty: YES if some c in C (0.05 grid, plus c_sh) has
  |r_min(c)| <= 2 sd_boot(c) on EVERY E1 ladder at that same c; otherwise "NOT BY EMBEDDING ACCOUNTING ALONE".
  If C is empty or V1 is undetermined: V2 = UNDETERMINED (no common c), and each ladder's own result over its own
  95% set is reported.
- Reported with the verdicts (not part of them): where the bootstrap LRT detects a second term (which c), and the
  c-free test.

## verify.py (positive and null controls; ALL PASS required before results.md)
1. Reproduction check above.
2. Fitter: the vectorized 1-term fitter equals fit_batch's 1-term RSS within 1e-6 relative on the four real ladders
   at 31 values of c.
3. Positive control, c recovery: 40 synthetic ladders on porian-rw's sizes, planted 1-term law in N_c with c = 0.5
   (E, A, p from the rw 1-term fit at c = 0.5), sd 0.005: median |c1 - 0.5| <= 0.1 and the 95% set covers 0.5 in
   >= 80%; the smallest-size test is "within noise" in >= 80% at the true c.
4. Null control, second term: on those ladders the bootstrap LRT at the true c fires in <= 10%.
5. Positive control, second term: planted 2-term law at c = 0.5 (rw 1-term fit at c = 0.5 plus a fast term
   B n^-alpha with alpha = 1.2 and B n_1^-alpha = 0.3 nats at the smallest size), sd 0.005, 40 ladders: LRT at the true
   c fires in >= 80% and the smallest-size 1-term residual is flagged (outside 2 sd) in >= 80%.
6. Shared c, null: four synthetic ladders with the real ladders' sizes and counts, each a 1-term law in N_c with the
   same c = 0.5 (each ladder's own 1-term fit at c = 0.5), sd 0.005, 40 sets: dBIC_shared <= 6 and C non-empty in
   >= 80%.
7. Shared c, positive: the same with c = 0.1 on the Porian pair and c = 1.5 on Chinchilla and Pythia, sd 0.005:
   dBIC_shared > 6 in >= 80%.
8. Nesting RSS2 <= RSS1 at every grid c on every ladder; every bootstrap has its planned B; no NaN in the outputs.

## Compute
CPU only on skampere2 (shared, load ~65): `nice`, at most 16 processes. Bootstrap sizes are the above (B = 200 / 500);
estimated < 1 CPU-hour in total. Nothing goes to Sherlock.

## Known limitations (now)
Four ladders, two of them Porian (same architecture family, different data). Chinchilla's y is interpolated between
IsoFLOP points; Pythia's is a fixed-D slice (D/N from 25 to 4,284), so its small models are far more over-trained.
y is held fixed as c changes (Porian's D was 20 x the c = 1 count). A c that fits is a fitted weight, not a
mechanism. Seed noise is known only for Porian (DM-2 `seed_sigma.csv`).
