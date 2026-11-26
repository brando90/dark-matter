# E1 results (11-25-2026): does one embedding weight c bring the tiny models back onto one power law?

As of 11-25-2026. Run by ub9sc with **Claude Code, Opus 5.5 (claude-opus-5-5), fits in Python (numpy/scipy) on
skampere2, CPU only**. PREREG.md committed and pushed before any fit (85db67d); Amendment 1 (c14025b) after the first
verify run and before this file. `python verify.py`: **ALL PASS** (8 checks; results/verify.txt).

## Bottom line for the dark-matter question
Counting parameters as N_c = N_nonemb + c N_emb does **not** bring the tiny Porian models back onto a single power law.
A shared c passes the pre-registered "one c" rule only weakly, and at that c the smallest Porian models still sit
above the 1-term curve, by more than 2 bootstrap sd. On Porian, the second (fast) power-law term stays detectable with
c free: bootstrap LRT p = 0.005 on rw and on owt2, the smallest p that B = 200 can give. Chinchilla and Pythia have no
tiny-model excess to explain. Their 1-term fits are already within noise at the smallest size, so they barely
constrain c.

| rule | verdict | why |
|---|---|---|
| V1, one c | **SUPPORTED (weakly)** | Informative ladders: Porian rw, Porian owt2, Chinchilla (Pythia's set covers all of [0, 3]). Their common 95% set is c in [2.21, 2.66]. Shared c vs per-ladder c: dBIC_shared = -4.4 (<= 6), bootstrap p = 0.065. |
| V2, tiny models return to the curve | **NOT BY EMBEDDING ACCOUNTING ALONE** | At no c in the common set are the smallest sizes of all four ladders within 2 bootstrap sd. Porian rw and owt2 fail at every c in [2.21, 2.66]. |

Why V1 is weak (read the numbers, not the label):
- The common set is bounded below by Porian and above by Chinchilla. Chinchilla's set [0, 2.66] has length 2.66,
  just inside the "informative" cut of 2.7.
- The pooled best c, c_sh = 3.0, sits at the grid's upper edge and lies OUTSIDE Chinchilla's own 95% set
  (n ln(RSS/RSS_min) = 4.34 > 3.84).
- Post hoc (labelled): the pooled profile rises by only +0.47 at c = 4 and by +4.0 at c = 5. So the shared optimum
  is around c = 3-4: an embedding block worth 3-4 body parameters (1.5-2 per untied embedding parameter on Porian).
  DM-3 already called that unphysical.
- On Porian alone, c is pushed toward and past the grid edge: rw's 95% set is [2.09, 3.0+] and owt2's is
  [0.04, 0.06] U [2.21, 3.0+]. Both exclude c = 1 and c = 0. They also just exclude c = 2, the true untied
  parameter count.

## Headline table (1-term and 2-term laws, c profiled on [0, 3], unit = one V x d block)
| ladder | sizes | c1 (1-term) | 1-term 95% set | excl. 1 / 0 | c2 (2-term) | 2-term 95% set | dBIC free (1+c vs 2+c) | c-free LRT p | sd 1-term+c / 2-term+c |
|---|---|---|---|---|---|---|---|---|---|
| porian-rw | 14 | 3.00 (edge) | [2.09, 3.00+] | yes / yes | 1.07 | [0.52, 2.10] | +15.0 | 0.005 | 0.0119 / 0.0065 |
| porian-owt2 | 14 | 3.00 (edge) | [0.04, 0.06] U [2.21, 3.00+] | yes / yes | 1.67 | [0.75, 3.00+] | +18.2 | 0.005 | 0.0171 / 0.0083 |
| chinchilla | 28 | 0.00 (edge) | [0.00, 2.66] | no / no | 0.00 | [0.00, 3.00+] | -4.3 | 0.154 | 0.0112 / 0.0113 |
| pythia | 8 | 0.79 | [0.00, 3.00+] | no / no | 0.79 | [0.00, 3.00+] | -4.2 | 1.000 | 0.0255 / 0.0361 |

dBIC free: BIC(1-term + c, k = 4) minus BIC(2-term + c, k = 6). Positive favours the 2-term law. The c-free LRT is
the DM-1b rule, B = 200 parametric refits from the 1-term fit at c1, with c re-profiled in each refit.

**Bootstrap LRT (B = 200) for 2-term vs 1-term, at fixed c** (31 values, c = 0, 0.1, ..., 3):
- Porian rw and owt2: the second term is detected at 30 of 31 values (every c >= 0.1; at c = 0 the 2-term fit
  collapses).
- Chinchilla: detected at 7 of 31 values, scattered over 1.3-3.0, with p 0.040-0.050, at the threshold.
- Pythia: detected at none.

At the 2-term optimum on Porian (c2), in the repo convention (alpha fast):
- rw: A_fast 3.22, alpha 1.46, B_slow 4.56, beta 0.15, E 1.29.
- owt2: A_fast 10.8, alpha 1.67, B_slow 5.16, beta 0.21, E 1.34.

With c on [0, 3] these dBICs (+15.0 / +18.2) are lower than DM-3's (+19.5 / +20.8 on [0, 2]), because the 1-term fit
keeps improving up to c = 3.

## Smallest-size residual (the "5M signal")
r_min is observed minus the 1-term fit at the smallest size (nats). z is r_min divided by its bootstrap sd. The
bootstrap uses B = 500 parametric refits, with noise sd taken from the 2-term fit at the same c.

| ladder | r_min at c = 0 (z) | at c_std (z) | at c_sh = 3.00 (z) | leave-out at c_sh (z) | c in own 95% set where within 2 sd |
|---|---|---|---|---|---|
| porian-rw (5M = w96) | -0.080 (-3.3) | +0.022 (+4.9) | +0.017 (+3.2) | +0.042 (+3.2) | none |
| porian-owt2 (w96) | -0.068 (-2.9) | +0.031 (+6.3) | +0.026 (+4.8) | +0.066 (+4.8) | 0.05 only (the E = 0 branch) |
| chinchilla (smallest 202M) | +0.013 (+1.6) | +0.014 (+1.8) | +0.016 (+2.0) | +0.030 (+2.0) | 0.00-2.65 |
| pythia (70M) | -0.013 (-1.3) | -0.000 (-0.0) | +0.002 (+0.2) | +0.019 (+0.2) | 0.00-3.00 |

- **5M signal, by the pre-registered rule: it does not survive non-embedding counting as an excess, but it does not
  vanish either.** At c = 0 the Porian 5M model sits 0.07-0.08 nats BELOW the 1-term curve (z -2.9 / -3.3). The
  excess flips sign rather than going to noise.
- In the standard count (c = 2, all parameters) the 5M model is +0.022 / +0.031 above the curve (z +4.9 / +6.3).
- No c in [0, 3] puts Porian rw's smallest model within 2 sd. On owt2 only c = 0.05 does, the DM-3 branch where E
  is forced to 0.
- Chinchilla's and Pythia's smallest models are within noise at almost any c. Their smallest sizes, 202M and 70M,
  are past the range where Porian shows its excess.

## Pooled test (one shared c vs per-ladder c)
| test | c_sh | Lambda | dBIC_shared (BIC_shared - BIC_per) | bootstrap p |
|---|---|---|---|---|
| primary: 1-term, all 4 ladders | 3.00 (edge) | 8.04 | -4.43 | 0.065 (B 200) |
| secondary: 2-term, all 4 | 1.20 | 2.08 | -10.40 | - |
| secondary: 1-term, Porian pair | 3.00 | 0.00 | -3.33 | - |
| secondary: 1-term, without Pythia | 3.00 | 4.34 | -3.72 | - |

## Reproduction of DM-3 (gate, passed)
DM-3's grid [0, 2] and fitter give the following on rw / owt2. RSS1 matches dm3_profileA.csv to 9e-14 / 4e-14
relative (E1's fitter: 5e-6 / 6e-6).

| | rw | owt2 |
|---|---|---|
| c1 | 2.000 | 0.045 |
| 95% set | [0.05, 0.11] U [1.55, 2.0+] | [0.03, 0.08] U [1.83, 2.0+] |
| dBIC free | 19.53 | 20.81 |

All of these equal DM-3's RESULTS.md values to within its rounding.

## Controls (verify.py, ALL PASS)
1. DM-3 reproduction (above).
2. The vectorized 1-term fitter equals fit_batch exactly.
3. c recovery (planted c = 0.5, sd 0.005, 40 ladders):
   - Median bias -0.015. The 95% set covers the truth in 0.80. The smallest size is within 2 sd in 0.93.
   - The pre-registered precision criterion (median |c1 - 0.5| <= 0.1) FAILED at 0.150. Amendment 1 explains why:
     the estimator is unbiased but imprecise on 14 sizes.
4. LRT null: false detections 0.100 (4/40), at the 0.10 bound.
5. Planted fast term (0.3 nats at the smallest size): the LRT fires in 1.00 and the smallest-size residual is
   flagged in 1.00.
6. Shared-c null: the shared c is accepted in 0.90.
7. Different c planted (0.1 vs 1.5): dBIC_shared > 6 in 0.97.
8. Nesting, bootstrap sizes and NaN checks pass.

## Post hoc (labelled; not in any verdict)
- Pooled 1-term profile past the grid, from the pre-registered secondary RSS1 at c = 4, 5, 7, 10: +0.47, +4.0,
  +12.5, +22.9 relative to c = 3. Per ladder:
  - owt2's 1-term RSS keeps falling to about c = 5 (0.00242, against 0.00292 at c = 3).
  - rw's flattens at c = 3-4.
  - Chinchilla's and Pythia's rise.
- Seed noise (DM-2 `seed_sigma.csv`, Porian's separate "seed" runs, 7 seeds) at the smallest width, w96:
  - rw: sd 0.062, owt2: sd 0.012.
  - By that external scale, rw's +0.017 at c_sh is within noise (0.3 seed sd) and owt2's +0.026 is about 2.2 seed sd.
  - The pre-registered noise scale (the 2-term residual sd, 0.006-0.008) is much smaller than rw's seed sd at w96.
  - So V2's "not within noise" on rw depends on which noise scale is right for a single tuned-run point. It is
    robust on owt2 and in the leave-out residuals. This is the main caveat on V2.

## Caveats
- **Four ladders, of which only two (the Porian pair: same architecture, different data) have a tiny-model excess**.
  So "one c across ladders" is mostly a test of whether Chinchilla and Pythia object to the c Porian wants. They
  barely do, because they start at 70M-200M.
- Pythia is a fixed-D slice (300B tokens for every size, D/N from 25 to 4,284). Its counts come from the official
  HF configs (data/pythia_hf_configs.csv). They reproduce the paper's totals, within 1.5% of the nominal names.
- Chinchilla's y is interpolated between IsoFLOP points (DM-2).
- **ColPret LLaMA is excluded**: no width, depth or vocab on disk. No Marin Delphi points are on disk.
- c is in units of one V x d block. For the untied Porian and Pythia models, c = 2 is the full parameter count. For
  Chinchilla, c = 1 is DM-2's N_total, close to the reported count. A per-unique-embedding-parameter unit would halve
  c for the untied ladders. It cannot be defined for Chinchilla from the data on disk (tying not recorded).
- Bootstrap sizes are modest (B = 200 for the LRTs and the pooled test, 500 for residual sd). p = 0.005 is the
  floor for B = 200. Everything ran in ~2 minutes on skampere2, so nothing went to Sherlock.
- Single-ladder c point estimates move by ~0.15 between noise draws even at sd 0.005 (verify check 3). Read the sets.

## Files
- Code: `e1_data.py` -> `repro_dm3.py` -> `e1.py` -> `summarize_e1.py`, plus `verify.py` and the shared helpers in
  `e1lib.py`.
- `results/`:
  - Tables: `ladders.csv`, `e1_profile.csv`, `e1_ladder_summary.csv`, `e1_lrt_by_c.csv`,
    `e1_smallest_resid_by_c.csv`, `e1_smallest_resid_at_c_sh.csv`, `e1_resid_by_size.csv`, `e1_pooled.csv`,
    `e1_verdict.csv`, `e1_tables.md`.
  - Figures (all with grid): `e1_profiles.png`, `e1_resid_by_size.png`, `e1_smallest_z.png`.

**Verdict line:**
- V1 SUPPORTED (weakly: common c in [2.21, 2.66], dBIC_shared -4.4, pooled optimum at the c = 3 grid edge).
- V2 NOT BY EMBEDDING ACCOUNTING ALONE (the Porian 5M excess stays at z +3.2 / +4.8 at the shared c). On Porian, the
  second term survives c free (bootstrap LRT p = 0.005).
