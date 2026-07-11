# Full plot set

Complete set of figures produced by the analysis pipeline, kept here for transparency and reproducibility. The figures used in the dissertation itself are in `../figures/`. This folder has everything the analysis generates.

Both platforms are included:

- `a4000/` is the RTX A4000 (16 GB, Ampere, bare-metal), 5 models
- `blackwell/` is the RTX PRO 6000 Blackwell (96 GB, Xen VM), 8 models

Per-run plots are included for one representative run only, to keep the repository size down. The runs are very consistent, which you can check in `aggregate/run_consistency.png`. The aggregate plots already combine all runs.

## Layout

Each platform folder contains:
aggregate/                     summaries computed across all runs
run_1/
comparison/
by_language/               energy per language, with a subfolder per language
by_difficulty/             energy and timeout by difficulty tier
<model>/plots/               per-model plots, one folder per model

### aggregate/
Summaries across all runs: mean energy per token with error bars (`mean_energy_errorbars.png`), run-to-run consistency (`run_consistency.png`), a coefficient-of-variation heatmap (`cv_heatmap.png`), and mean energy per language (`mean_energy_<language>.png`).

### run_1/comparison/
Cross-model comparison for one run. Model efficiency, effective energy and timeout rate are at the top level, with `by_language/` and `by_difficulty/` breakdowns below. Each language under `by_language/` also has its own subfolder.

### run_1/<model>/plots/
Per-model plots for each evaluated model.

## Reading notes

Energy is reported as net values, with the idle baseline subtracted. The main metric is energy per token (mWh/token).

Models with high timeout rates, especially the smaller Qwen coder variants, should be read alongside their timeout rate. The effective-energy plots divide by all attempted prompts instead of only the completed ones, which corrects for that.
