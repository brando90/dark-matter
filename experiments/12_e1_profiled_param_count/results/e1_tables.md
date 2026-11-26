| ladder | sizes | c1 (1-term) | 1-term 95% set | excl. 1 / 0 | c2 (2-term) | 2-term 95% set | dBIC free (1+c vs 2+c) | c-free LRT p | sd 1-term+c / 2-term+c |
|---|---|---|---|---|---|---|---|---|---|
| porian-rw | 14 | 3.00 | [2.09, 3.00+] | yes / yes | 1.07 | [0.52, 2.10] | +15.0 | 0.005 | 0.0119 / 0.0065 |
| porian-owt2 | 14 | 3.00 | [0.04, 0.06] U [2.21, 3.00+] | yes / yes | 1.67 | [0.75, 3.00+] | +18.2 | 0.005 | 0.0171 / 0.0083 |
| chinchilla | 28 | 0.00 | [0.00, 2.66] | no / no | 0.00 | [0.00, 3.00+] | -4.3 | 0.154 | 0.0112 / 0.0113 |
| pythia | 8 | 0.79 | [0.00, 3.00+] | no / no | 0.79 | [0.00, 3.00+] | -4.2 | 1.000 | 0.0255 / 0.0361 |

| ladder | r_min at c = 0 (z) | at c_std (z) | at c_sh = 3.00 (z) | leave-out at c_sh (z) | c in own 95% set where within 2 sd |
|---|---|---|---|---|---|
| porian-rw | -0.080 (-3.3) | +0.022 (+4.9) | +0.017 (+3.2) | +0.042 (+3.2) | none |
| porian-owt2 | -0.068 (-2.9) | +0.031 (+6.3) | +0.026 (+4.8) | +0.066 (+4.8) | 0.05 |
| chinchilla | +0.013 (+1.6) | +0.014 (+1.8) | +0.016 (+2.0) | +0.030 (+2.0) | 0.00-2.65 |
| pythia | -0.013 (-1.3) | -0.000 (-0.0) | +0.002 (+0.2) | +0.019 (+0.2) | 0.00-3.00 |

| test | c_sh | Lambda | dBIC_shared (BIC_shared - BIC_per) | bootstrap p |
|---|---|---|---|---|
| primary_1term_all4 | 3.00 | 8.04 | -4.43 | 0.065 (B 200) |
| secondary_2term_all4 | 1.20 | 2.08 | -10.40 |  |
| secondary_1term_porian_pair | 3.00 | 0.00 | -3.33 |  |
| secondary_1term_without_pythia | 3.00 | 4.34 | -3.72 |  |

LRT (B = 200) detects a second term at c in {0, 0.1, ..., 3}:
- porian-rw: 30 of 31 grid values (0.1-3.0)
- porian-owt2: 30 of 31 grid values (0.1-3.0)
- chinchilla: 7 of 31 grid values (1.3-3.0)
- pythia: 0 of 31 grid values

Post hoc (labelled): pooled 1-term -2 log-lik relative to c = 3: c = 3: +0.00, c = 4: +0.47, c = 5: +4.01, c = 7: +12.50, c = 10: +22.90
Per-ladder RSS1 at c = 3 / 4 / 5 / 7 / 10: porian-rw 0.00142 / 0.00141 / 0.00156 / 0.00196 / 0.00244; porian-owt2 0.00292 / 0.00249 / 0.00242 / 0.00262 / 0.00307; chinchilla 0.00354 / 0.00373 / 0.00392 / 0.00431 / 0.00490; pythia 0.00413 / 0.00493 / 0.00565 / 0.00687 / 0.00830
