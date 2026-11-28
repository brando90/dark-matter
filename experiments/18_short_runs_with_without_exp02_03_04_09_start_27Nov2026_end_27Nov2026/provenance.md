# Provenance of the 245 Chinchilla rows and of the 11 short runs (11-27-2026)

**Doc link:** <https://github.com/brando90/dark-matter/blob/main/experiments/18_short_runs_with_without_exp02_03_04_09_start_27Nov2026_end_27Nov2026/provenance.md>

**TLDR:** The 245 rows are a reconstruction from a published scatter plot, not training logs, and their token counts are computed from FLOPs and parameter counts. The 11 runs under one token per parameter are the largest models of two FLOP columns, and the five lowest of them are the ones the Besiroglu et al. replication drops as unexplained outliers. Whether any of these runs was trained as the plot shows cannot be checked from public data.

Date: 11-27-2026. Owner: Claude Code, Claude Sonnet 5.5. Item 6 of the ranked next steps. Numbers below marked "file" come from `data/data_hitchhikers_guide/colpret_chinchilla.csv.gz` through `trace_short_runs.py` and are in `results/trace_summary.json` and `results/report_numbers.txt`; statements marked "published" come from web pages read on 11-27-2026 through a search and a summarising page fetch, so they should be checked against the papers before they are quoted anywhere.

## What the file is

1. The file has 245 rows with `original_paper` set to chinchilla and `data` set to MassiveText, with 43 distinct model names (file).
2. In every row FLOPs equal 6 N D to a relative deviation of at most 4.4e-16 (file). The token column is therefore derived from FLOPs and the parameter count; the file carries no independent token count.
3. Sorting log10 FLOPs and splitting at gaps above 0.05 dex gives 21 columns, 8 of which hold 8 or more runs and contain 220 of the 245 rows (file). That is the picture of horizontal FLOP bands that an IsoFLOP design produces.

## What the published sources say

1. Besiroglu, Erdil, Barnett and You, "Chinchilla Scaling: A replication attempt", arXiv:2404.10102 (version 2, 05-15-2024): the 245 points were reconstructed from Figure 4 of Hoffmann et al. The figure plots model size on one axis and training FLOP on the other, with the loss encoded as the colour of each point on a log scale from 2.00 to 5.00. They converted the PDF to SVG and read each point's position and fill colour. They name two noise sources: the FLOP axis has no tick marks, so each label's vertical centre was taken as its value, and the colour bar has 256 values, which limits loss precision to about 0.01. Neither an accuracy check against source data nor an independent validation is described.
2. The same paper says five high-loss points in a column near 1e19 FLOP, with the lowest token-to-parameter ratios (below 0.4), are often dropped, leaving 240 points, and that the reason for their high loss is unclear. The headline estimates use the 240 points: E 1.8172, A 482.01, B 2085.43, alpha 0.3478, beta 0.3658. Their appendix with all 245 points gives E 1.89, A 463.3, B 12530, alpha 0.345 and beta 0.452.
3. Hoffmann et al., "Training Compute-Optimal Large Language Models", arXiv:2203.15556: the abstract says they trained over 400 language models of 70 million to over 16 billion parameters on 5 to 500 billion tokens, and Approach 2 varies model size over nine FLOP budgets from 6e18 to 3e21. The paper's own data table is not part of what I read, so the individual budgets inside that range are not checked here.
4. A third-party blog post (shape-of-code.com, 04-28-2024) re-extracted the same figure from the PDF, reports 245 rows, and claims that the axis calibration in the SVG extraction was slightly off. I did not verify this claim.

## What the trace shows (file)

1. 10 of the 11 runs under one token per parameter lie in the column near 1e19 FLOP, which has 32 runs with FLOPs from 8.0e+18 to 1.2e+19, and they are its 10 largest model sizes, from 1.27 to 6.80 billion parameters. The eleventh, 12.57 billion parameters and 0.64 tokens per parameter, is the largest model of the column near 6e20 FLOP (14 runs).
2. At a fixed FLOP budget C, tokens per parameter are C divided by 6 N squared, so the largest model of a budget has the fewest tokens per parameter. That is what an IsoFLOP design predicts, and it explains why the short runs sit at the large-N end of their columns. It does not explain their losses.
3. The five lowest ratios, 0.036, 0.1732, 0.2332, 0.3085 and 0.4031, are all in the 1e19 column. Their losses are 5.006, 4.665, 3.766, 3.794 and 3.447, up to 1.686 times the median loss of runs within 0.1 dex of their FLOPs, which matches the "up to about 70% above" description in the replication. The fifth ratio is 0.4031, a little above the replication's "below 0.4".
4. Tokens run from 2.4e+08 to 3.2e+11. 37 of the 245 rows have fewer than 5e9 tokens, below the 5 to 500 billion range quoted in the Hoffmann abstract; 10 of those 37 are among the 11 short runs and 27 are not. That range may describe something narrower than the plotted points, or the extraction may be off; the data cannot decide.

## What is and is not reachable

Reachable: the 245 reconstructed rows and the figure, the text of both papers, and the replication's extraction code, which the replication paper links but which I did not fetch because the sandbox shell reaches only package registries and GitHub.

Not reachable: the original run logs and the table behind Figure 4. They are not public as far as I could find, so no one can confirm from outside that these runs trained with the token counts the plot implies, or why their losses are high.

## What this means for the dark-matter experiments

1. Every Chinchilla number in experiments 02, 03, 04 and 09 is conditional on the 245 figure-reconstructed rows.
2. The five lowest-ratio runs move the fitted token exponent from 0.4557 to 0.3669, close to the replication's 0.3658 for the same removal, and removing all 11 gives 0.3392 (see `results.md`, items 40 and 41). Reporting both the 245-run and the 234-run results, as folder 18 does, is the honest course; dropping runs because they hurt a fit is a degree of freedom that the dose-response and random-removal controls in `results.md` are there to bound.
3. A cleaner source would be the Epoch AI repository's CSV, compared row by row with the ColPret file. That comparison is the next step for this trace and was not done here.
