#!/usr/bin/env python3
"""
LLM Energy + Temperature Benchmark - Analysis v2.3

Bug fixes over user's version:
  - "comparison" added to SKIP_DIRS (was being processed as a model dir on re-runs)
  - colors[idx] replaced with enumerate() in scatter - after sort_values() the
    DataFrame index is non-sequential, causing IndexError with 4 models
  - set([m['name']...]) replaced with sorted() everywhere - set ordering is
    non-deterministic, causing inconsistent line colours between plots

New features:
  - Per-prompt model comparison: grid of subplots (one per prompt), each showing
    all models as grouped bars for energy, speed, and efficiency
  - Model-by-model prompt line plot: all models on one axes, x=prompt index,
    allowing direct visual comparison of how each model handled each prompt
  - summary.csv written to run root - one row per model, all metrics
  - per_prompt_summary.csv - one row per (model, prompt), all per-prompt metrics
"""

import sys
import warnings
from pathlib import Path

import matplotlib
# Use Agg (file renderer) before importing pyplot. This is required when
# the script runs as a subprocess of the GUI (or any headless context)
# where there is no X display connection. Agg writes directly to PNG
# files without needing a screen, so all plt.savefig() calls work fine.
matplotlib.use('Agg')
import matplotlib.pyplot as _plt_cfg

# ── Global figure style: readable at print/projector size ──────────────────
_plt_cfg.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Map raw model identifiers / gguf filenames to clean display names.
_MODEL_DISPLAY = {
    'gemma3_4b': 'Gemma 3 4B',
    'mistral_7b': 'Mistral 7B',
    'qwen2_5_7b': 'Qwen 2.5 7B',
    'qwen2_5_coder_7b': 'Qwen 2.5 Coder 7B',
    'qwen2_5_coder_32b': 'Qwen 2.5 Coder 32B',
    'qwen3_32b': 'Qwen 3 32B',
    'gpt-oss-20b': 'GPT-OSS 20B',
    'gpt-oss-120b': 'GPT-OSS 120B',
}

def display_name(raw):
    """Return a clean, human-readable model name for any raw id/filename."""
    s = str(raw)
    # strip quantisation and split-file suffixes
    for junk in ['.Q4_K_M', '-mxfp4', '.gguf']:
        s = s.replace(junk, '')
    import re as _re
    s = _re.sub(r'-?\d{5}-of-\d{5}', '', s)   # drop 00001-of-00003
    key = s.strip().rstrip('.').replace('.', '_')
    if key in _MODEL_DISPLAY:
        return _MODEL_DISPLAY[key]
    # try the un-suffixed base against the map
    base = key.split('_Q4')[0]
    if base in _MODEL_DISPLAY:
        return _MODEL_DISPLAY[base]
    # fallback: tidy underscores
    return s.replace('_', ' ').strip()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore", category=UserWarning)

if len(sys.argv) != 2:
    print("Usage: python3 analyze_results.py <run_directory>")
    sys.exit(1)

run_dir = Path(sys.argv[1])
print(f"\nAnalyzing results in: {run_dir}\n")

# Directories that are never model result folders
SKIP_DIRS = {"idle", "venv", "comparison"}

# ── Consistent colour assignment ─────────────────────────────────────────────
# Built once from sorted model names; reused in every plot so each model always
# gets the same colour regardless of which plot is being drawn.
def build_colour_map(model_names):
    cmap = plt.cm.tab10
    n = max(len(model_names) - 1, 1)
    return {name: cmap(i / n) for i, name in enumerate(sorted(model_names))}


# ── Data loading ─────────────────────────────────────────────────────────────
all_models_data = []
per_prompt_data  = {}   # prompt_index -> {model_name -> metric dict}
all_dfs          = {}   # model_name -> DataFrame (for per-prompt grid plots)

for model_dir in sorted(run_dir.iterdir()):
    if not model_dir.is_dir() or model_dir.name in SKIP_DIRS:
        continue

    prompt_file = model_dir / "prompt_energy.csv"
    if not prompt_file.exists():
        continue

    print(f"Processing model: {model_dir.name}")

    df = pd.read_csv(prompt_file, sep=';', na_values=['N/A', 'n/a', ''])

    numeric_cols = [
        'cpu_avg_W', 'cpu_max_W', 'gpu_avg_W', 'gpu_max_W',
        'duration_s', 'cpu_energy_Wh', 'gpu_energy_Wh',
        'cpu_net_avg_W', 'gpu_net_avg_W', 'cpu_net_Wh', 'gpu_net_Wh',
        'tokens', 'tokens_per_s', 'Wh_per_token', 'throttled',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['tokens']    = df['tokens'].fillna(0).astype(int)
    df['throttled'] = df['throttled'].fillna(0).astype(int)
    df['model']     = model_dir.name
    # Backward compat: if no language column, fill with "unknown"
    if 'language' not in df.columns:
        df['language'] = 'unknown'
    else:
        df['language'] = df['language'].fillna('unknown')

    # Backward compat: if no difficulty column, fill with "unclassified"
    if 'difficulty' not in df.columns:
        df['difficulty'] = 'unclassified'
    else:
        df['difficulty'] = df['difficulty'].fillna('unclassified')

    # tokens_per_s: the new llama.cpp (b9124+) writes timing to stdout as
    # "[ Prompt: X t/s | Generation: Y t/s ]" - the shell parses this directly.
    # Normalise locale decimal separators (comma → dot) in case the shell
    # didn't catch them all, then prefer the shell value when non-zero,
    # falling back to tokens/duration_s for older llama.cpp versions.
    if df['tokens_per_s'].dtype == object:
        df['tokens_per_s'] = (df['tokens_per_s'].astype(str)
                               .str.replace(',', '.', regex=False))
        df['tokens_per_s'] = pd.to_numeric(df['tokens_per_s'], errors='coerce').fillna(0.0)

    df['tokens_per_s'] = df.apply(
        lambda r: r['tokens_per_s'] if r['tokens_per_s'] > 0
        else (r['tokens'] / r['duration_s']
              if r['duration_s'] > 0 and r['tokens'] > 0 else 0.0),
        axis=1
    )
    energy_col = 'gpu_net_Wh' if 'gpu_net_Wh' in df.columns else 'gpu_energy_Wh'
    df['Wh_per_token'] = df.apply(
        lambda r: r[energy_col] / r['tokens']
        if r['tokens'] > 0 and pd.notna(r[energy_col]) and r[energy_col] > 0
        else np.nan,  # NaN rather than 0 so it doesn't drag down language averages
        axis=1
    )

    all_dfs[model_dir.name] = df

    # Load timeout log for this model (backward compat: file may not exist)
    timeout_file = model_dir / "timeout_log.csv"
    if timeout_file.exists():
        df_to = pd.read_csv(timeout_file, sep=";")
        for col, fill in [('language','unknown'), ('difficulty','unclassified')]:
            if col not in df_to.columns:
                df_to[col] = fill
            else:
                df_to[col] = df_to[col].fillna(fill)
        df_to['model'] = model_dir.name
    else:
        df_to = pd.DataFrame(columns=['timestamp','prompt_index',
                                      'language','difficulty','model'])
    all_dfs[model_dir.name + "__timeouts"] = df_to


    # Per-prompt lookup table
    for _, row in df.iterrows():
        prompt_idx = int(row['prompt_index'])
        if prompt_idx not in per_prompt_data:
            per_prompt_data[prompt_idx] = {}
        # Skip prompts with no generated tokens (likely returned empty
        # or were too fast for the energy window to capture meaningful data).
        # These would otherwise appear as zeros in rolling average plots.
        if row['tokens'] <= 0 or pd.isna(row['Wh_per_token']):
            continue
        per_prompt_data[prompt_idx][model_dir.name] = {
            'tokens':        row['tokens'],
            'tokens_per_s':  row['tokens_per_s'],
            'gpu_energy_Wh': float(row['gpu_net_Wh']) if 'gpu_net_Wh' in df.columns else 0,
            'Wh_per_token':  row['Wh_per_token'],
            'duration_s':    row['duration_s'],
            'throttled':     row['throttled'],
            'language':      row['language'],
            'difficulty':    row['difficulty'],
        }

    # Model-level summary
    valid_df     = df[(df['tokens'] > 0) & (df['Wh_per_token'].notna())]
    total_tokens = int(valid_df['tokens'].sum())
    total_energy = float(valid_df['gpu_net_Wh'].sum()) if 'gpu_net_Wh' in valid_df.columns else 0.0

    n_timeouts  = len(df_to)
    n_completed = len(valid_df)
    n_attempted = n_completed + n_timeouts
    timeout_pct = (n_timeouts / n_attempted * 100) if n_attempted > 0 else 0

    model_summary = {
        'name':                    model_dir.name,
        'total_energy_Wh':         total_energy,
        'avg_power_W':             float(df['gpu_net_avg_W'].mean()) if 'gpu_net_avg_W' in df.columns else 0,
        'total_tokens':            total_tokens,
        'avg_tokens_per_s':        float(valid_df['tokens_per_s'].mean()) if not valid_df.empty else 0,
        'avg_energy_per_token_mWh':float(valid_df['Wh_per_token'].mean() * 1000) if not valid_df.empty and valid_df['Wh_per_token'].mean() > 0 else 0,
        'throttle_pct':            float(df['throttled'].sum() / len(df) * 100) if len(df) > 0 else 0,
        'total_time_s':            float(df['duration_s'].sum()),
        'prompt_count':            n_completed,
        'n_timeouts':              n_timeouts,
        'n_attempted':             n_attempted,
        'timeout_pct':             round(timeout_pct, 1),
        'energy_per_1k_tokens_Wh': (total_energy / total_tokens * 1000) if total_tokens > 0 else 0,
        'effective_energy_per_prompt_Wh': total_energy / n_attempted if n_attempted > 0 else 0,
    }
    # Validate: if a model generated tokens but recorded zero GPU energy,
    # the energy measurement is invalid (likely driver failure). Flag it.
    if total_tokens > 0 and total_energy <= 0:
        print(f"  WARNING: {model_dir.name} generated {total_tokens} tokens "
              f"but recorded 0 Wh GPU energy. Measurement invalid — "
              f"likely nvidia-smi failure. Excluding from comparisons.")
        model_summary['invalid'] = True
    else:
        model_summary['invalid'] = False

    all_models_data.append(model_summary)

    # Per-model plots
    plots_dir = model_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Cumulative energy
    plt.figure(figsize=(12, 6))
    energy_col = 'gpu_net_Wh' if 'gpu_net_Wh' in df.columns else 'gpu_energy_Wh'
    df['cumulative_energy_Wh'] = df[energy_col].cumsum()
    plt.plot(df.index, df['cumulative_energy_Wh'], linewidth=2)
    plt.xlabel('Prompt Number')
    plt.ylabel('Cumulative Energy (Wh)')
    plt.title(f'{model_dir.name} - Cumulative Energy Consumption')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / 'cumulative_energy.png', dpi=200)
    plt.close()

    # Energy vs tokens scatter
    if 'gpu_net_Wh' in df.columns:
        valid = df[(df['tokens'] > 0) & (df['gpu_net_Wh'] > 0)]
        plt.figure(figsize=(12, 6))
        if len(valid) >= 2:
            plt.scatter(valid['tokens'], valid['gpu_net_Wh'], alpha=0.6)
            z = np.polyfit(valid['tokens'], valid['gpu_net_Wh'], 1)
            p = np.poly1d(z)
            plt.plot(valid['tokens'], p(valid['tokens']), 'r--', alpha=0.8,
                     label=f'Trend: {z[0]*1000:.2f} mWh/token')
            plt.xlabel('Tokens Generated')
            plt.ylabel('Net GPU Energy (Wh)')
            plt.title(f'{model_dir.name} - Energy vs Tokens')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / 'energy_vs_tokens.png', dpi=200)
        plt.close()

# ── Shared setup for comparison plots ────────────────────────────────────────
model_names  = sorted(all_dfs.keys())   # deterministic sorted list - used everywhere
colour_map   = build_colour_map(model_names)
prompt_indices = sorted(per_prompt_data.keys())

comparison_dir = run_dir / "comparison"
comparison_dir.mkdir(exist_ok=True)


# ── PER-PROMPT MODEL COMPARISON ──────────────────────────────────────────────
# With 2000+ prompts, individual dot plots are unreadable.
# Strategy:
#   - Rolling average lines (window=50) instead of raw dots
#   - Box plots per model grouped by language x difficulty
#   - Violin plots for Wh/token distribution per model
#   - per_prompt_model_grid removed (unusable at scale)

if per_prompt_data and len(model_names) >= 1 and all_dfs:
    # Build a combined DataFrame from all model data for aggregation
    _df_all = pd.concat(
        [v for k,v in all_dfs.items() if not k.endswith("__timeouts")],
        ignore_index=True
    )
    energy_col_plot = 'gpu_net_Wh' if 'gpu_net_Wh' in _df_all.columns else 'gpu_energy_Wh'

    ROLL = min(50, max(10, len(prompt_indices) // 20))  # adaptive window

    # 1. Rolling average energy per prompt
    fig, ax = plt.subplots(figsize=(14, 5))
    for name in model_names:
        vals = pd.Series([per_prompt_data[p].get(name, {}).get('gpu_energy_Wh', np.nan)
                          for p in prompt_indices])
        raw = vals
        smoothed = vals.rolling(ROLL, min_periods=1, center=True).mean()
        ax.plot(prompt_indices, raw, color=colour_map[name], alpha=0.12, linewidth=0.7)
        ax.plot(prompt_indices, smoothed, color=colour_map[name], linewidth=2.0,
                label=f"{name.replace('_Q4_K_M','').replace('_',' ')} ({ROLL}-prompt avg)")
    ax.set_xlabel('Prompt Index')
    ax.set_ylabel('Net GPU Energy (Wh)')
    ax.set_title(f'Energy per Prompt - Rolling {ROLL}-prompt Average (faint = raw)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(comparison_dir / 'per_prompt_energy.png', dpi=200)
    plt.close()

    # 2. Rolling average speed per prompt
    fig, ax = plt.subplots(figsize=(14, 5))
    for name in model_names:
        vals = pd.Series([per_prompt_data[p].get(name, {}).get('tokens_per_s', np.nan)
                          for p in prompt_indices])
        ax.plot(prompt_indices, vals, color=colour_map[name], alpha=0.12, linewidth=0.7)
        smoothed = vals.rolling(ROLL, min_periods=1, center=True).mean()
        ax.plot(prompt_indices, smoothed, color=colour_map[name], linewidth=2.0,
                label=name.replace('_Q4_K_M','').replace('_',' '))
    ax.set_xlabel('Prompt Index')
    ax.set_ylabel('Tokens / second')
    ax.set_title(f'Inference Speed - Rolling {ROLL}-prompt Average')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(comparison_dir / 'per_prompt_speed.png', dpi=200)
    plt.close()

    # 3. Rolling average efficiency per prompt
    fig, ax = plt.subplots(figsize=(14, 5))
    for name in model_names:
        vals = pd.Series([per_prompt_data[p].get(name, {}).get('Wh_per_token', np.nan)
                          for p in prompt_indices]) * 1000
        ax.plot(prompt_indices, vals, color=colour_map[name], alpha=0.12, linewidth=0.7)
        smoothed = vals.rolling(ROLL, min_periods=1, center=True).mean()
        ax.plot(prompt_indices, smoothed, color=colour_map[name], linewidth=2.0,
                label=name.replace('_Q4_K_M','').replace('_',' '))
    ax.set_xlabel('Prompt Index')
    ax.set_ylabel('mWh / token')
    ax.set_title(f'Energy Efficiency - Rolling {ROLL}-prompt Average (lower = better)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(comparison_dir / 'per_prompt_efficiency.png', dpi=200)
    plt.close()

    # 4. Box plots: Wh/token per model grouped by language x difficulty
    # Rows = languages, columns = difficulties; one box per model per cell
    _valid = _df_all[_df_all['Wh_per_token'].notna() & (_df_all['tokens'] > 0)].copy()
    _valid['Wh_per_token_mWh'] = _valid['Wh_per_token'] * 1000
    _langs_box = sorted(_valid['language'].dropna().unique())
    _diffs_box = [d for d in ['easy','medium','hard','unclassified']
                  if d in _valid['difficulty'].unique()]

    if _langs_box and _diffs_box:
        nrows = len(_langs_box)
        ncols = len(_diffs_box)
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(max(10, ncols*3.5), max(5, nrows*3)),
                                  squeeze=False, sharey='row')
        fig.suptitle('Wh/token distribution: language x difficulty x model',
                     fontsize=12, fontweight='bold')
        for ri, lang in enumerate(_langs_box):
            for ci, diff in enumerate(_diffs_box):
                ax = axes[ri][ci]
                sub = _valid[(_valid['language']==lang) & (_valid['difficulty']==diff)]
                data_by_model = [sub[sub['model']==m]['Wh_per_token_mWh'].dropna().values
                                 for m in model_names]
                bp = ax.boxplot(data_by_model, patch_artist=True,
                                medianprops=dict(color='white', linewidth=1.5),
                                whiskerprops=dict(color='#8b949e'),
                                capprops=dict(color='#8b949e'),
                                flierprops=dict(marker='.', markersize=2,
                                                color='#8b949e', alpha=0.4))
                for patch, name in zip(bp['boxes'], model_names):
                    patch.set_facecolor(colour_map[name])
                    patch.set_alpha(0.75)
                if ri == 0:
                    ax.set_title(diff.upper(), fontsize=9)
                if ci == 0:
                    ax.set_ylabel(f'{lang} / mWh per tok', fontsize=8)
                ax.set_xticks(range(1, len(model_names)+1))
                ax.set_xticklabels(
                    [m.replace('_Q4_K_M','').replace('_',' ') for m in model_names],
                    fontsize=6)
                ax.grid(True, alpha=0.25, axis='y')
        plt.tight_layout()
        plt.savefig(comparison_dir / 'boxplot_lang_diff.png', dpi=200, bbox_inches='tight')
        plt.close()


    # 5. Violin plot: Wh/token distribution per model (overall)
    if not _valid.empty:
        violin_models = [m for m in model_names
                         if len(_valid[_valid['model']==m]['Wh_per_token_mWh'].dropna()) >= 2]
        data_violin   = [_valid[_valid['model']==m]['Wh_per_token_mWh'].dropna().values
                         for m in violin_models]
        if data_violin:
            fig, ax = plt.subplots(figsize=(max(8, len(violin_models)*2), 6))
            parts = ax.violinplot(data_violin, positions=range(len(violin_models)),
                                  showmedians=True, showextrema=True)
            for i, (pc, name) in enumerate(zip(parts['bodies'], violin_models)):
                pc.set_facecolor(colour_map[name])
                pc.set_alpha(0.6)
            parts['cmedians'].set_color('white')
            parts['cmaxes'].set_color('#8b949e')
            parts['cmins'].set_color('#8b949e')
            parts['cbars'].set_color('#8b949e')
            ax.set_xticks(range(len(violin_models)))
            ax.set_xticklabels([m.replace('_Q4_K_M','').replace('_',' ')
                                for m in violin_models], fontsize=9)
            ax.set_ylabel('Wh / token (mWh)', fontsize=9)
            ax.set_title('Energy per token distribution per model - '
                         'width = density, line = median',
                         fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(comparison_dir / 'violin_efficiency.png', dpi=200, bbox_inches='tight')
            plt.close()



    # 5. Heatmap - models x prompt buckets (bucketed for scale)
    N_BUCKETS = min(50, len(prompt_indices))
    bucket_size = max(1, len(prompt_indices) // N_BUCKETS)
    buckets = [prompt_indices[i:i+bucket_size]
               for i in range(0, len(prompt_indices), bucket_size)]
    bucket_labels = [str(b[0]) for b in buckets]
    heatmap_data = []
    for m in model_names:
        row = []
        for bucket in buckets:
            vals = [per_prompt_data[p].get(m, {}).get('gpu_energy_Wh', np.nan)
                    for p in bucket]
            vals = [v for v in vals if v is not None and not np.isnan(v) and v > 0]
            row.append(float(np.mean(vals)) if vals else 0.0)
        heatmap_data.append(row)
    fig, ax = plt.subplots(figsize=(16, max(4, len(model_names) * 1.2)))
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    tick_step = max(1, len(bucket_labels) // 20)
    ax.set_xticks(np.arange(0, len(bucket_labels), tick_step))
    ax.set_xticklabels(bucket_labels[::tick_step], rotation=45, ha='right', fontsize=7)
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=8)
    ax.set_xlabel(f'Prompt Index (groups of {bucket_size})')
    ax.set_ylabel('Model')
    ax.set_title(f'Energy Heatmap - Mean GPU Wh per {bucket_size}-prompt bucket')
    fig.colorbar(im, ax=ax, label='Energy (Wh)')
    plt.tight_layout()
    plt.savefig(comparison_dir / 'energy_heatmap.png', dpi=200)
    plt.close()

    # 6. Relative efficiency per prompt (vs best model per prompt)
    if len(model_names) >= 2:
        plt.figure(figsize=(14, 6))
        for name in model_names:
            relative = []
            for p in prompt_indices:
                all_e = [per_prompt_data[p].get(m, {}).get('gpu_energy_Wh', 0)
                         for m in model_names]
                best    = min((e for e in all_e if e > 0), default=1)
                current = per_prompt_data[p].get(name, {}).get('gpu_energy_Wh', best)
                relative.append(current / best if best > 0 else 1)
            plt.plot(prompt_indices, relative, marker='o', linewidth=2, markersize=5,
                     label=f'{name} (avg {np.mean(relative):.2f}x)',
                     color=colour_map[name])
        plt.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='Best')
        plt.xlabel('Prompt Index')
        plt.ylabel('Energy relative to best model per prompt')
        plt.title('Relative Energy Efficiency per Prompt (lower = better)')
        plt.legend(fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(comparison_dir / 'relative_efficiency_per_prompt.png', dpi=200)
        plt.close()

# ── CROSS-MODEL SUMMARY PLOTS ─────────────────────────────────────────────────
if len(all_models_data) >= 1:
    df_all_models = pd.DataFrame(all_models_data)
    # Exclude models with invalid energy measurements from comparison charts
    invalid_models = df_all_models[df_all_models.get('invalid', False) == True]
    if not invalid_models.empty:
        print("\nExcluded from comparison (invalid energy data):")
        for _, row in invalid_models.iterrows():
            print(f"  {row['name']}: {row['total_tokens']} tokens, "
                  f"{row['total_energy_Wh']:.4f} Wh")
    df_compare = (df_all_models[df_all_models.get('invalid', False) != True]
                  .sort_values('total_energy_Wh')
                  .reset_index(drop=True))
    df_compare['display'] = df_compare['name'].apply(display_name)

    # 1. Total energy comparison
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(df_compare['display'], df_compare['total_energy_Wh'],
                  color=plt.cm.viridis(np.linspace(0, 0.8, len(df_compare))))
    ax.set_xlabel('Model')
    ax.set_ylabel('Total Net GPU Energy (Wh)')
    ax.set_title('Total Energy Consumption by Model')
    plt.xticks(rotation=30, ha='right')
    top_val = max(df_compare['total_energy_Wh'])
    for bar, (_, row) in zip(bars, df_compare.iterrows()):
        v  = row['total_energy_Wh']
        to = row.get('timeout_pct', 0)
        label = f'{v:.2f} Wh'
        if to > 0:
            label += f'  ({to:.0f}% timeout)'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + top_val * 0.01,
                label, ha='center', va='bottom', fontsize=8)
    if 'timeout_pct' in df_compare.columns and df_compare['timeout_pct'].max() > 0:
        ax.set_title('Total Energy Consumption by Model\n'
                     '(high timeout % = incomplete prompt coverage)')
    plt.tight_layout()
    plt.savefig(comparison_dir / 'total_energy_comparison.png', dpi=200)
    plt.close()

    # 2. Energy per 1K tokens (efficiency comparison)
    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.bar(df_compare['display'], df_compare['energy_per_1k_tokens_Wh'],
                  color=plt.cm.plasma(np.linspace(0, 0.8, len(df_compare))))
    ax.set_xlabel('Model')
    ax.set_ylabel('Energy per 1K Tokens (Wh)')
    ax.set_title('Energy Efficiency by Model (lower is better)')
    plt.xticks(rotation=30, ha='right')
    top_eff = max(df_compare['energy_per_1k_tokens_Wh']) if len(df_compare) > 0 else 1
    for bar, (_, row) in zip(bars, df_compare.iterrows()):
        v  = row['energy_per_1k_tokens_Wh']
        to = row.get('timeout_pct', 0)
        label = f'{v:.3f}'
        if to > 0:
            label += f'  ({to:.0f}% timeout)'
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + top_eff * 0.01,
                label, ha='center', va='bottom', fontsize=8)
    if 'timeout_pct' in df_compare.columns and df_compare['timeout_pct'].max() > 0:
        ax.set_title('Energy Efficiency by Model (lower is better)\n'
                     '(high timeout % = survivorship bias in averages)')
    plt.tight_layout()
    plt.savefig(comparison_dir / 'efficiency_comparison.png', dpi=200)
    plt.close()

    # 2b. Effective energy per attempted prompt (timeout-penalized)
    if 'effective_energy_per_prompt_Wh' in df_compare.columns:
        df_eff_sorted = df_compare.sort_values('effective_energy_per_prompt_Wh')
        df_eff_sorted['display'] = df_eff_sorted['name'].apply(display_name)
        fig, ax = plt.subplots(figsize=(14, 8))
        bars_ep = ax.bar(df_eff_sorted['display'],
                      df_eff_sorted['effective_energy_per_prompt_Wh'] * 1000,
                      color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_eff_sorted))))
        ax.set_xlabel('Model')
        ax.set_ylabel('Effective energy per attempted prompt (mWh)')
        ax.set_title('Effective Energy per Attempted Prompt (timeout-penalized)\n'
                     'Total energy / (completed + timed-out) -- lower is better')
        plt.xticks(rotation=30, ha='right')
        top_ep = df_eff_sorted['effective_energy_per_prompt_Wh'].max() * 1000
        top_ep = top_ep if top_ep > 0 else 1
        for bar_ep, (_, row) in zip(bars_ep, df_eff_sorted.iterrows()):
            v  = row['effective_energy_per_prompt_Wh'] * 1000
            to = row.get('timeout_pct', 0)
            n  = int(row.get('n_attempted', 0))
            label = f'{v:.1f} mWh\n{to:.0f}% timeout'
            ax.text(bar_ep.get_x() + bar_ep.get_width() / 2,
                    bar_ep.get_height() + top_ep * 0.01,
                    label, ha='center', va='bottom', fontsize=7)
        plt.tight_layout()
        plt.savefig(comparison_dir / 'effective_energy_comparison.png', dpi=200)
        plt.close()

    # 3. Speed vs efficiency scatter - enumerate() for safe colour indexing
    if len(df_compare) >= 2:
        fig, ax = plt.subplots(figsize=(12, 8))
        for i, (_, row) in enumerate(df_compare.iterrows()):
            colour = plt.cm.tab10(i / max(len(df_compare) - 1, 1))
            ax.scatter(row['avg_tokens_per_s'], row['energy_per_1k_tokens_Wh'],
                       s=200, label=row['name'], color=colour, alpha=0.8, zorder=3)
            ax.annotate(row['name'],
                        (row['avg_tokens_per_s'], row['energy_per_1k_tokens_Wh']),
                        xytext=(6, 6), textcoords='offset points', fontsize=9)
        ax.axhline(df_compare['energy_per_1k_tokens_Wh'].mean(),
                   color='gray', linestyle='--', alpha=0.4, label='Avg efficiency')
        ax.axvline(df_compare['avg_tokens_per_s'].mean(),
                   color='gray', linestyle=':',  alpha=0.4, label='Avg speed')
        ax.set_xlabel('Average Speed (tokens/s)')
        ax.set_ylabel('Energy per 1K Tokens (Wh)')
        ax.set_title('Speed vs Energy Efficiency Trade-off')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(comparison_dir / 'speed_vs_efficiency_scatter.png', dpi=200)
        plt.close()

    # 4. Radar chart - guarded against all-zero columns
    if len(df_compare) >= 2:
        def safe_norm(series):
            mx = series.max()
            return series / mx if mx > 0 else pd.Series([0.5] * len(series),
                                                         index=series.index)

        metrics_labels = [
            'Energy\nEfficiency\n(inv)',
            'Speed',
            'Stability\n(inv throttle)',
            'Energy/Prompt\n(inv)',
        ]

        per_prompt_e   = df_compare['total_energy_Wh'] / \
                         df_compare['prompt_count'].replace(0, 1)
        inv_eff        = safe_norm(1 - df_compare['energy_per_1k_tokens_Wh'] /
                                   max(df_compare['energy_per_1k_tokens_Wh'].max(), 1e-9))
        norm_speed     = safe_norm(df_compare['avg_tokens_per_s'])
        inv_throttle   = 1 - df_compare['throttle_pct'] / 100
        inv_per_prompt = safe_norm(1 - per_prompt_e /
                                   max(per_prompt_e.max(), 1e-9))

        angles = np.linspace(0, 2 * np.pi, len(metrics_labels), endpoint=False).tolist()
        angles_closed = angles + angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        for i, (_, row) in enumerate(df_compare.iterrows()):
            colour = plt.cm.tab10(i / max(len(df_compare) - 1, 1))
            values = [
                float(inv_eff.iloc[i]),
                float(norm_speed.iloc[i]),
                float(inv_throttle.iloc[i]),
                float(inv_per_prompt.iloc[i]),
            ]
            vals_closed = values + values[:1]
            ax.plot(angles_closed, vals_closed, 'o-', linewidth=2,
                    label=row['name'], color=colour)
            ax.fill(angles_closed, vals_closed, alpha=0.1, color=colour)

        ax.set_xticks(angles)
        ax.set_xticklabels(metrics_labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar (higher = better)', size=14, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(comparison_dir / 'performance_radar.png', dpi=200)
        plt.close()

# ── CSV exports ───────────────────────────────────────────────────────────────
if all_models_data:
    pd.DataFrame(all_models_data).to_csv(run_dir / 'summary.csv', index=False)
    print(f"Summary CSV written: {run_dir / 'summary.csv'}")

if all_dfs:
    combined = pd.concat([v for k,v in all_dfs.items() if not k.endswith("__timeouts")], ignore_index=True)
    cols = ['model', 'language', 'difficulty', 'prompt_index', 'duration_s', 'tokens',
            'tokens_per_s', 'gpu_energy_Wh', 'gpu_net_Wh', 'Wh_per_token',
            'cpu_energy_Wh', 'cpu_net_Wh', 'throttled']
    export_cols = [c for c in cols if c in combined.columns]
    combined[export_cols].to_csv(run_dir / 'per_prompt_summary.csv', index=False)
    print(f"Per-prompt CSV written: {run_dir / 'per_prompt_summary.csv'}")

# ── Language analysis ─────────────────────────────────────────────────────────
# Produces three levels of output:
#   1. comparison/by_language/<lang>/  - per-language folder with plots for
#                                        that language only (all models)
#   2. comparison/by_language/         - cross-language aggregates and the
#                                        best-model-per-language comparison
#   3. Console summary grouped by language
#
# Skipped entirely if no language column is present or only "unknown" exists.
# ─────────────────────────────────────────────────────────────────────────────

def _grouped_bar(ax, x, languages, model_names, lang_model_stats,
                 value_col, ylabel, title, colour_map, scale=1.0, fmt="{:.2f}",
                 log=False, annotate=False):
    """Draw a grouped bar chart of value_col per language per model.

    log=True uses a logarithmic y-axis so that one high-consuming model does
    not visually flatten the others (important for the language chart, where
    Qwen 2.5 Coder 7B sits far above the rest).
    annotate defaults to False: with eight models the per-bar numbers become
    unreadable clutter, so we rely on the axis instead.
    """
    n_m   = len(model_names)
    width = 0.8 / max(n_m, 1)
    for i, model in enumerate(model_names):
        vals = []
        for lang in languages:
            sub = lang_model_stats[
                (lang_model_stats['language'] == lang) &
                (lang_model_stats['model']    == model)
            ]
            vals.append(float(sub[value_col].iloc[0]) * scale
                        if not sub.empty else 0.0)
        offset = (i - n_m / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      label=display_name(model),
                      color=colour_map[model], alpha=0.9)
        if annotate:
            top = max(vals) if max(vals) > 0 else 1
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() * 1.02,
                            fmt.format(v), ha='center', va='bottom', fontsize=8)
    if log:
        ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([l.capitalize() for l in languages],
                       rotation=25, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # Legend outside the plot area so it never covers the bars.
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
              frameon=False, ncol=1)
    ax.grid(True, alpha=0.3, axis='y')


def _heatmap(ax, matrix, row_labels, col_labels, title, cbar_label, fmt="{:.3f}"):
    """Draw an annotated heatmap."""
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha='right', fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=10)
    median = np.nanmedian(matrix)
    for r in range(len(row_labels)):
        for c in range(len(col_labels)):
            v = matrix[r, c]
            if not np.isnan(v):
                ax.text(c, r, fmt.format(v), ha='center', va='center',
                        fontsize=8, fontweight='bold',
                        color='#0f1117' if v > median else '#e6edf3')
    return im


if all_dfs:
    combined_all = pd.concat([v for k,v in all_dfs.items() if not k.endswith("__timeouts")], ignore_index=True)
    # Exclude models flagged as invalid (zero energy despite generating tokens)
    _invalid_names = [m['name'] for m in all_models_data if m.get('invalid', False)]
    if _invalid_names:
        combined_all = combined_all[~combined_all['model'].isin(_invalid_names)]
    languages     = sorted(combined_all['language'].dropna().unique())
    has_languages = not (list(languages) == ['unknown'] or len(languages) == 0)

    if has_languages and len(languages) >= 1:
        energy_col = 'gpu_net_Wh' if 'gpu_net_Wh' in combined_all.columns                      else 'gpu_energy_Wh'

        model_names_lang = sorted(combined_all['model'].unique())
        cmap_lang = plt.cm.tab10
        colour_map_lang = {
            m: cmap_lang(i / max(len(model_names_lang) - 1, 1))
            for i, m in enumerate(model_names_lang)
        }

        # ── Aggregate stats: language × model ──────────────────────────────
        lang_model_stats = (
            combined_all[combined_all['tokens'] > 0]
            .groupby(['language', 'model'])
            .agg(
                mean_Wh_per_token =('Wh_per_token',   'mean'),
                std_Wh_per_token  =('Wh_per_token',   'std'),
                mean_tokens_per_s =('tokens_per_s',   'mean'),
                std_tokens_per_s  =('tokens_per_s',   'std'),
                total_energy_Wh   =(energy_col,       'sum'),
                prompt_count      =('prompt_index',   'count'),
            )
            .reset_index()
        )

        # ── 1. Per-language folders ─────────────────────────────────────────
        # Each language gets its own subfolder with two plots:
        #   - all models side-by-side for that language (efficiency + speed)
        #   - per-prompt energy line for that language across models
        print("\nGenerating per-language plots...")
        for lang in languages:
            ldir = run_dir / "comparison" / "by_language" / lang
            ldir.mkdir(parents=True, exist_ok=True)

            sub_stats = lang_model_stats[lang_model_stats['language'] == lang]
            sub_data  = combined_all[
                (combined_all['language'] == lang) &
                (combined_all['tokens']   >  0) &
                (combined_all['Wh_per_token'].notna())
            ]

            # Skip language if no data
            if sub_data.empty:
                continue

            x = np.arange(len(model_names_lang))

            # 1a. Efficiency + speed side-by-side bar chart for this language
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            fig.suptitle(f"Language: {lang.upper()} - all models",
                         fontsize=12, fontweight='bold')

            eff_vals   = []
            speed_vals = []
            colours    = []
            for model in model_names_lang:
                row = sub_stats[sub_stats['model'] == model]
                eff_vals.append(
                    float(row['mean_Wh_per_token'].iloc[0]) * 1000
                    if not row.empty else 0.0)
                speed_vals.append(
                    float(row['mean_tokens_per_s'].iloc[0])
                    if not row.empty else 0.0)
                colours.append(colour_map_lang[model])

            bars1 = ax1.bar(x, eff_vals, color=colours, alpha=0.85)
            ax1.set_xticks(x)
            ax1.set_xticklabels(
                [m.replace('_Q4_K_M','').replace('_','\n') for m in model_names_lang],
                fontsize=8)
            ax1.set_ylabel('Mean energy per token (mWh)')
            ax1.set_title('Efficiency (lower = better)')
            ax1.grid(True, alpha=0.3, axis='y')
            top1 = max(eff_vals) if max(eff_vals) > 0 else 1
            for bar, v in zip(bars1, eff_vals):
                if v > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + top1*0.01,
                             f'{v:.4f}', ha='center', va='bottom', fontsize=7)

            bars2 = ax2.bar(x, speed_vals, color=colours, alpha=0.85)
            ax2.set_xticks(x)
            ax2.set_xticklabels(
                [m.replace('_Q4_K_M','').replace('_','\n') for m in model_names_lang],
                fontsize=8)
            ax2.set_ylabel('Mean tokens / second')
            ax2.set_title('Speed (higher = better)')
            ax2.grid(True, alpha=0.3, axis='y')
            top2 = max(speed_vals) if max(speed_vals) > 0 else 1
            for bar, v in zip(bars2, speed_vals):
                if v > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + top2*0.01,
                             f'{v:.1f}', ha='center', va='bottom', fontsize=7)

            plt.tight_layout()
            plt.savefig(ldir / 'models_comparison.png', dpi=200,
                        bbox_inches='tight')
            plt.close()

            # 1b. Per-prompt energy line for this language
            prompt_indices_lang = sorted(sub_data['prompt_index'].unique())
            if len(prompt_indices_lang) > 1:
                fig, ax = plt.subplots(figsize=(13, 5))
                for model in model_names_lang:
                    model_sub = sub_data[sub_data['model'] == model]
                    vals = [
                        float(model_sub[model_sub['prompt_index'] == p]
                              [energy_col].mean())
                        if not model_sub[model_sub['prompt_index'] == p].empty
                        else np.nan
                        for p in prompt_indices_lang
                    ]
                    ax.plot(prompt_indices_lang, vals, marker='o', linewidth=1.8,
                            markersize=4, label=model,
                            color=colour_map_lang[model])
                ax.set_xlabel('Prompt index')
                ax.set_ylabel('Net GPU energy (Wh)')
                ax.set_title(f'{lang.upper()} - energy per prompt per model')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(ldir / 'per_prompt_energy.png', dpi=200,
                            bbox_inches='tight')
                plt.close()

            # Per-language CSV
            sub_stats.to_csv(ldir / 'summary.csv', index=False)
            print(f"  {lang}: {ldir}")

        # ── 2. Cross-language aggregate folder ─────────────────────────────
        if len(languages) > 1:
            agg_dir = run_dir / "comparison" / "by_language"
            agg_dir.mkdir(parents=True, exist_ok=True)

            lang_model_stats.to_csv(
                agg_dir / 'language_model_summary.csv', index=False)

            x_all = np.arange(len(languages))

            # 2a. Efficiency grouped bar - all languages, all models.
            # Log y-axis: without it, one high-consumption model (e.g. Qwen
            # 2.5 Coder 7B) flattens every other bar into an unreadable band.
            fig, ax = plt.subplots(figsize=(max(12, len(languages) * 1.6), 6.5))
            _grouped_bar(ax, x_all, languages, model_names_lang,
                         lang_model_stats,
                         value_col='mean_Wh_per_token',
                         ylabel='Mean energy per token (mWh, log scale)',
                         title='Energy per token by language and model '
                               '(lower = better)',
                         colour_map=colour_map_lang,
                         scale=1000, fmt="{:.2f}", log=True)
            plt.tight_layout()
            plt.savefig(agg_dir / 'efficiency_by_language.png', dpi=200,
                        bbox_inches='tight')
            plt.close()

            # 2b. Speed grouped bar
            fig, ax = plt.subplots(figsize=(max(10, len(languages) * 2.2), 6))
            _grouped_bar(ax, x_all, languages, model_names_lang,
                         lang_model_stats,
                         value_col='mean_tokens_per_s',
                         ylabel='Mean tokens / second',
                         title='Inference speed per language per model',
                         colour_map=colour_map_lang,
                         scale=1.0, fmt="{:.1f}")
            plt.tight_layout()
            plt.savefig(agg_dir / 'speed_by_language.png', dpi=200,
                        bbox_inches='tight')
            plt.close()

            # 2c. Heatmap: models × languages
            heat = np.array([
                [lang_model_stats[
                    (lang_model_stats['model']    == m) &
                    (lang_model_stats['language'] == l)
                 ]['mean_Wh_per_token'].mean() * 1000
                 if not lang_model_stats[
                    (lang_model_stats['model']    == m) &
                    (lang_model_stats['language'] == l)
                 ].empty else np.nan
                 for l in languages]
                for m in model_names_lang
            ])
            fig, ax = plt.subplots(
                figsize=(max(8, len(languages) * 1.5),
                         max(4, len(model_names_lang) * 1.2)))
            im = _heatmap(ax, heat, model_names_lang, languages,
                          title='Mean energy per token (mWh) - model × language',
                          cbar_label='mWh / token')
            fig.colorbar(im, ax=ax, label='mWh / token')
            plt.tight_layout()
            plt.savefig(agg_dir / 'efficiency_heatmap_lang.png', dpi=200,
                        bbox_inches='tight')
            plt.close()

            # 2d. Best model per language - efficiency and speed
            # For each language, find which model has the lowest mWh/token
            # and which has the highest tokens/s, then compare those winners
            best_eff_per_lang = (
                lang_model_stats
                .sort_values('mean_Wh_per_token')
                .groupby('language')
                .first()
                .reset_index()
            )
            best_spd_per_lang = (
                lang_model_stats
                .sort_values('mean_tokens_per_s', ascending=False)
                .groupby('language')
                .first()
                .reset_index()
            )

            fig, (ax1, ax2) = plt.subplots(1, 2,
                                           figsize=(max(12, len(languages)*2.2),
                                                    6))
            fig.suptitle('Best model per language', fontsize=12,
                         fontweight='bold')

            # Efficiency winner bars
            eff_colours = [colour_map_lang.get(m, 'steelblue')
                           for m in best_eff_per_lang['model']]
            bars1 = ax1.bar(range(len(languages)),
                            best_eff_per_lang['mean_Wh_per_token'] * 1000,
                            color=eff_colours, alpha=0.85)
            ax1.set_xticks(range(len(languages)))
            ax1.set_xticklabels(best_eff_per_lang['language'],
                                rotation=25, ha='right', fontsize=9)
            ax1.set_ylabel('Mean energy per token (mWh)')
            ax1.set_title('Most efficient model per language')
            ax1.grid(True, alpha=0.3, axis='y')
            top1 = best_eff_per_lang['mean_Wh_per_token'].max() * 1000
            for bar, (_, row) in zip(bars1, best_eff_per_lang.iterrows()):
                ax1.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + top1 * 0.01,
                         row['model'].replace('_Q4_K_M',''),
                         ha='center', va='bottom', fontsize=7, rotation=15)

            # Speed winner bars
            spd_colours = [colour_map_lang.get(m, 'steelblue')
                           for m in best_spd_per_lang['model']]
            bars2 = ax2.bar(range(len(languages)),
                            best_spd_per_lang['mean_tokens_per_s'],
                            color=spd_colours, alpha=0.85)
            ax2.set_xticks(range(len(languages)))
            ax2.set_xticklabels(best_spd_per_lang['language'],
                                rotation=25, ha='right', fontsize=9)
            ax2.set_ylabel('Mean tokens / second')
            ax2.set_title('Fastest model per language')
            ax2.grid(True, alpha=0.3, axis='y')
            top2 = best_spd_per_lang['mean_tokens_per_s'].max()
            for bar, (_, row) in zip(bars2, best_spd_per_lang.iterrows()):
                ax2.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + top2 * 0.01,
                         row['model'].replace('_Q4_K_M',''),
                         ha='center', va='bottom', fontsize=7, rotation=15)

            # Shared legend showing which colour = which model
            handles = [
                plt.Rectangle((0,0),1,1, color=colour_map_lang[m], alpha=0.85)
                for m in model_names_lang
            ]
            fig.legend(handles, model_names_lang,
                       loc='lower center', ncol=len(model_names_lang),
                       fontsize=8, bbox_to_anchor=(0.5, -0.05))
            plt.tight_layout()
            plt.savefig(agg_dir / 'best_model_per_language.png', dpi=200,
                        bbox_inches='tight')
            plt.close()

        # ── Console summary by language ─────────────────────────────────────
        print("\n" + "=" * 80)
        print("PER-LANGUAGE SUMMARY  (mean mWh/token | mean tok/s | prompts)")
        print("=" * 80)
        for lang in languages:
            print(f"\n  {lang.upper()}")
            sub = lang_model_stats[
                lang_model_stats['language'] == lang
            ].sort_values('mean_Wh_per_token')
            for _, row in sub.iterrows():
                winner = " < best efficiency"                     if row['model'] == sub.iloc[0]['model'] else ""
                print(f"    {row['model']:<34} "
                      f"{row['mean_Wh_per_token']*1000:.4f} mWh/tok  "
                      f"{row['mean_tokens_per_s']:.1f} tok/s  "
                      f"({int(row['prompt_count'])} prompts){winner}")

# ── NEW: Tier 1 + 2 language × model analysis ───────────────────────────────
# Runs whenever language data is present (same guard as the block above).
# Adds to comparison/by_language/ and per-model subfolders.
# ─────────────────────────────────────────────────────────────────────────────

if all_dfs:
    combined_all2 = pd.concat([v for k,v in all_dfs.items() if not k.endswith("__timeouts")], ignore_index=True)
    languages2    = sorted(combined_all2['language'].dropna().unique())
    _has_lang2    = not (list(languages2) == ['unknown'] or len(languages2) == 0)

    if _has_lang2 and len(languages2) >= 1:
        energy_col2 = 'gpu_net_Wh' if 'gpu_net_Wh' in combined_all2.columns                       else 'gpu_energy_Wh'
        model_names2 = sorted(combined_all2['model'].unique())
        cmap2 = plt.cm.tab10
        cmap_m2 = {m: cmap2(i / max(len(model_names2)-1, 1))
                   for i, m in enumerate(model_names2)}
        agg_dir2 = run_dir / "comparison" / "by_language"
        agg_dir2.mkdir(parents=True, exist_ok=True)

        # Aggregate with std so we can draw error bars
        lms2 = (
            combined_all2[(combined_all2['tokens'] > 0) &
                          (combined_all2['Wh_per_token'].notna())]
            .groupby(['language', 'model'])
            .agg(
                mean_eff  =('Wh_per_token',  'mean'),
                std_eff   =('Wh_per_token',  'std'),
                mean_spd  =('tokens_per_s',  'mean'),
                std_spd   =('tokens_per_s',  'std'),
                mean_tpp  =('tokens',         'mean'),   # tokens per prompt
                std_tpp   =('tokens',         'std'),
                mean_epp  =(energy_col2,       'mean'),  # energy per prompt
                std_epp   =(energy_col2,       'std'),
                n         =('prompt_index',   'count'),
            )
            .reset_index()
        )

        # ── TIER 1a: Language overhead plot ──────────────────────────────────
        # For each model: how much more/less energy per token does each language
        # cost relative to the reference language (alphabetically first)?
        # Value > 1 means that language costs more energy than the reference.
        if len(languages2) > 1:
            ref_lang = languages2[0]
            print(f"\n  Language overhead reference: '{ref_lang}'")

            overhead_records = []
            for model in model_names2:
                ref_row = lms2[(lms2['model'] == model) &
                               (lms2['language'] == ref_lang)]
                if ref_row.empty or ref_row['mean_eff'].iloc[0] == 0:
                    continue
                ref_val = ref_row['mean_eff'].iloc[0]
                for lang in languages2:
                    row = lms2[(lms2['model'] == model) & (lms2['language'] == lang)]
                    if row.empty:
                        continue
                    overhead_records.append({
                        'model': model, 'language': lang,
                        'overhead': row['mean_eff'].iloc[0] / ref_val,
                        'std_eff':  row['std_eff'].iloc[0],
                        'ref_val':  ref_val,
                    })

            if overhead_records:
                df_oh = pd.DataFrame(overhead_records)
                x_oh  = np.arange(len(languages2))
                n_m2  = len(model_names2)
                w2    = 0.8 / max(n_m2, 1)

                fig, ax = plt.subplots(figsize=(max(11, len(languages2)*2.2), 6))
                for i, model in enumerate(model_names2):
                    sub = df_oh[df_oh['model'] == model]
                    vals = [float(sub[sub['language']==l]['overhead'].iloc[0])
                            if not sub[sub['language']==l].empty else np.nan
                            for l in languages2]
                    offset = (i - n_m2/2 + 0.5) * w2
                    bars = ax.bar(x_oh + offset, vals, width=w2*0.9,
                                  label=model.replace('_Q4_K_M',''),
                                  color=cmap_m2[model], alpha=0.85)
                    # Error bars scaled to reference
                    errs = []
                    for lang in languages2:
                        r = sub[sub['language']==lang]
                        if r.empty or r['ref_val'].iloc[0] == 0:
                            errs.append(0)
                        else:
                            errs.append(r['std_eff'].iloc[0] / r['ref_val'].iloc[0])
                    ax.errorbar(x_oh + offset, vals, yerr=errs,
                                fmt='none', color='#e6edf3', capsize=2,
                                linewidth=0.8, alpha=0.6)

                ax.axhline(1.0, color='#8b949e', linewidth=1.0,
                           linestyle='--', label=f'Baseline ({ref_lang})')
                ax.set_xticks(x_oh)
                ax.set_xticklabels(languages2, rotation=25, ha='right', fontsize=9)
                ax.set_ylabel(f'Energy overhead relative to {ref_lang}', fontsize=9)
                ax.set_title('Language energy overhead per model\n'
                             '(1.0 = same as reference; >1.0 = more energy)',
                             fontsize=11, fontweight='bold')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3, axis='y')
                plt.tight_layout()
                plt.savefig(agg_dir2 / 'language_overhead.png', dpi=200,
                            bbox_inches='tight')
                plt.close()
                print(f"  Saved: language_overhead.png")

        # ── TIER 1b: Tokens-per-prompt by language ────────────────────────────
        # Shows whether a model generates more tokens for some languages,
        # which explains part of the energy difference independent of efficiency.
        x_tp  = np.arange(len(languages2))
        n_m2  = len(model_names2)
        w_tp  = 0.8 / max(n_m2, 1)

        fig, ax = plt.subplots(figsize=(max(11, len(languages2)*2.2), 6))
        for i, model in enumerate(model_names2):
            vals = []
            errs = []
            for lang in languages2:
                row = lms2[(lms2['model']==model) & (lms2['language']==lang)]
                vals.append(float(row['mean_tpp'].iloc[0]) if not row.empty else 0)
                errs.append(float(row['std_tpp'].iloc[0])
                            if not row.empty and not np.isnan(row['std_tpp'].iloc[0])
                            else 0)
            offset = (i - n_m2/2 + 0.5) * w_tp
            ax.bar(x_tp + offset, vals, width=w_tp*0.9,
                   label=model.replace('_Q4_K_M',''),
                   color=cmap_m2[model], alpha=0.85)
            ax.errorbar(x_tp + offset, vals, yerr=errs,
                        fmt='none', color='#e6edf3', capsize=2,
                        linewidth=0.8, alpha=0.6)
        ax.set_xticks(x_tp)
        ax.set_xticklabels(languages2, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel('Mean tokens generated per prompt', fontsize=9)
        ax.set_title('Output token count per language per model\n'
                     '(separates tokenisation cost from inference efficiency)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(agg_dir2 / 'tokens_per_prompt_by_language.png', dpi=200,
                    bbox_inches='tight')
        plt.close()
        print(f"  Saved: tokens_per_prompt_by_language.png")

        # ── TIER 1c: Error-bar efficiency chart ───────────────────────────────
        # Same as efficiency_by_language but with ±1 std error bars.
        fig, ax = plt.subplots(figsize=(max(11, len(languages2)*2.2), 6))
        for i, model in enumerate(model_names2):
            vals = []
            errs = []
            for lang in languages2:
                row = lms2[(lms2['model']==model) & (lms2['language']==lang)]
                vals.append(float(row['mean_eff'].iloc[0])*1000 if not row.empty else 0)
                errs.append(float(row['std_eff'].iloc[0])*1000
                            if not row.empty and not np.isnan(row['std_eff'].iloc[0])
                            else 0)
            offset = (i - n_m2/2 + 0.5) * w_tp
            ax.bar(x_tp + offset, vals, width=w_tp*0.9,
                   label=model.replace('_Q4_K_M',''),
                   color=cmap_m2[model], alpha=0.85)
            ax.errorbar(x_tp + offset, vals, yerr=errs,
                        fmt='none', color='#e6edf3', capsize=3,
                        linewidth=1.0)
        ax.set_xticks(x_tp)
        ax.set_xticklabels(languages2, rotation=25, ha='right', fontsize=9)
        ax.set_ylabel('Mean energy per token (mWh) ± 1 std', fontsize=9)
        ax.set_title('Energy efficiency per language per model - with variance\n'
                     '(error bars show ±1 std across prompts)',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(agg_dir2 / 'efficiency_by_language_errorbars.png', dpi=200,
                    bbox_inches='tight')
        plt.close()
        print(f"  Saved: efficiency_by_language_errorbars.png")

        # ── TIER 2a: Consistency heatmap (CV = std/mean) ─────────────────────
        # Lower CV = more consistent/predictable energy cost.
        # High CV means the model's energy varies a lot across prompts for
        # that language - less reliable for real-world deployment.
        cv_matrix = np.array([
            [float(lms2[(lms2['model']==m) & (lms2['language']==l)]['std_eff'].iloc[0]) /
             float(lms2[(lms2['model']==m) & (lms2['language']==l)]['mean_eff'].iloc[0])
             if not lms2[(lms2['model']==m) & (lms2['language']==l)].empty and
                float(lms2[(lms2['model']==m) & (lms2['language']==l)]['mean_eff'].iloc[0]) > 0
             else np.nan
             for l in languages2]
            for m in model_names2
        ])
        fig, ax = plt.subplots(
            figsize=(max(8, len(languages2)*1.5),
                     max(4, len(model_names2)*1.2)))
        im = ax.imshow(cv_matrix, cmap='RdYlGn_r', aspect='auto',
                       vmin=0, vmax=min(1.0, np.nanmax(cv_matrix)))
        ax.set_xticks(range(len(languages2)))
        ax.set_yticks(range(len(model_names2)))
        ax.set_xticklabels(languages2, rotation=30, ha='right', fontsize=9)
        ax.set_yticklabels(model_names2, fontsize=9)
        ax.set_title('Consistency heatmap - CV of energy per token (std/mean)\n'
                     'Green = consistent (low CV), Red = variable (high CV)',
                     fontsize=10, fontweight='bold')
        fig.colorbar(im, ax=ax, label='CV (lower = more consistent)')
        median_cv = np.nanmedian(cv_matrix)
        for r in range(len(model_names2)):
            for c in range(len(languages2)):
                v = cv_matrix[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f'{v:.2f}', ha='center', va='center',
                            fontsize=8, fontweight='bold',
                            color='#0f1117' if v > median_cv else '#e6edf3')
        plt.tight_layout()
        plt.savefig(agg_dir2 / 'consistency_heatmap.png', dpi=200,
                    bbox_inches='tight')
        plt.close()
        print(f"  Saved: consistency_heatmap.png")

        # ── TIER 2b: Per-model language radar ─────────────────────────────────
        # One radar chart per model. Axes = languages.
        # Value = normalised efficiency (1.0 = best across all models for that lang).
        # Makes it immediately visible which languages a model handles well.
        if len(languages2) >= 3:
            # Normalise: for each language, best (lowest) mWh/token = 1.0,
            # worst = 0.0. So a higher value on the radar is always better.
            eff_pivot = lms2.pivot(index='model', columns='language',
                                   values='mean_eff')
            col_min = eff_pivot.min(axis=0)
            col_max = eff_pivot.max(axis=0)
            # Invert: lower energy = better → flip to higher = better
            eff_norm = 1 - (eff_pivot - col_min) / (col_max - col_min).replace(0, 1)

            angles = np.linspace(0, 2*np.pi, len(languages2),
                                 endpoint=False).tolist()
            angles_c = angles + angles[:1]

            n_models_r = len(model_names2)
            ncols = min(3, n_models_r)
            nrows = (n_models_r + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(5.5*ncols, 5*nrows),
                                     subplot_kw=dict(projection='polar'))
            axes_flat = np.array(axes).flatten() if n_models_r > 1 else [axes]
            fig.suptitle('Per-model language profile\n'
                         '(higher = more efficient for that language)',
                         fontsize=12, fontweight='bold')

            for idx, model in enumerate(model_names2):
                ax_r = axes_flat[idx]
                vals_r = [float(eff_norm.loc[model, l])
                          if (model in eff_norm.index and l in eff_norm.columns
                              and not np.isnan(eff_norm.loc[model, l]))
                          else 0.0
                          for l in languages2]
                vals_c = vals_r + vals_r[:1]
                ax_r.plot(angles_c, vals_c, 'o-', linewidth=2,
                          color=cmap_m2[model])
                ax_r.fill(angles_c, vals_c, alpha=0.15, color=cmap_m2[model])
                ax_r.set_xticks(angles)
                ax_r.set_xticklabels(languages2, fontsize=8)
                ax_r.set_ylim(0, 1)
                ax_r.set_title(model.replace('_Q4_K_M','').replace('_',' '),
                               fontsize=9, pad=12, color=cmap_m2[model])
                ax_r.grid(True)

            for idx in range(n_models_r, len(axes_flat)):
                axes_flat[idx].set_visible(False)

            plt.tight_layout()
            plt.savefig(agg_dir2 / 'per_model_language_radar.png', dpi=200,
                        bbox_inches='tight')
            plt.close()
            print(f"  Saved: per_model_language_radar.png")

        # ── RECAP TABLE ───────────────────────────────────────────────────────
        # A single wide CSV and a rendered PNG table for easy reading.
        # Rows: models. Columns: one mWh/token column per language + overall.
        # Also includes tokens/s and CV columns.
        print("\n  Generating recap table...")

        recap_rows = []
        for model in model_names2:
            row = {'model': model.replace('_Q4_K_M','')}
            overall_eff_vals = []
            for lang in languages2:
                sub = lms2[(lms2['model']==model) & (lms2['language']==lang)]
                if not sub.empty:
                    eff_mwh = sub['mean_eff'].iloc[0] * 1000
                    spd     = sub['mean_spd'].iloc[0]
                    cv      = (sub['std_eff'].iloc[0] / sub['mean_eff'].iloc[0]
                               if sub['mean_eff'].iloc[0] > 0 else np.nan)
                    row[f'{lang}_mWh_per_tok'] = round(eff_mwh, 4)
                    row[f'{lang}_tok_per_s']   = round(spd, 2)
                    row[f'{lang}_cv']          = round(cv, 3) if not np.isnan(cv) else None
                    overall_eff_vals.append(eff_mwh)
                else:
                    row[f'{lang}_mWh_per_tok'] = None
                    row[f'{lang}_tok_per_s']   = None
                    row[f'{lang}_cv']          = None
            row['overall_mean_mWh_per_tok'] = round(np.nanmean(overall_eff_vals), 4) \
                if overall_eff_vals else None
            # Add timeout info from model_summary
            ms = [m for m in all_models_data if m['name'] == model]
            if ms:
                row['timeout_pct'] = ms[0].get('timeout_pct', 0)
                row['n_attempted'] = ms[0].get('n_attempted', 0)
            else:
                row['timeout_pct'] = 0
                row['n_attempted'] = 0
            recap_rows.append(row)

        df_recap = pd.DataFrame(recap_rows)
        recap_csv = agg_dir2 / 'recap_table.csv'
        df_recap.to_csv(recap_csv, index=False)
        print(f"  Saved: recap_table.csv")

        # Rendered PNG table - efficiency columns only for readability
        eff_cols = [c for c in df_recap.columns if c.endswith('_mWh_per_tok')]
        col_labels = ['Model'] + [c.replace('_mWh_per_tok','') for c in eff_cols] + ['Overall mWh/tok', 'Timeout %']
        cell_data  = []
        for _, r in df_recap.iterrows():
            cells = [r['model'].replace('_',' ')]
            for c in eff_cols:
                v = r[c]
                cells.append(f"{v:.4f}" if v is not None else "-")
            cells.append(f"{r['overall_mean_mWh_per_tok']:.4f}"
                         if r['overall_mean_mWh_per_tok'] is not None else "-")
            cells.append(f"{r.get('timeout_pct', 0):.0f}%")
            cell_data.append(cells)

        # Colour cells: green = best in column, red = worst
        n_rows_t = len(cell_data)
        n_cols_t = len(col_labels)
        cell_colours = [['#161b22'] * n_cols_t for _ in range(n_rows_t)]
        for ci in range(1, n_cols_t):      # skip model name column
            try:
                col_vals = [float(cell_data[ri][ci])
                            for ri in range(n_rows_t)
                            if cell_data[ri][ci] != '-']
                if not col_vals:
                    continue
                best  = min(col_vals)
                worst = max(col_vals)
                for ri in range(n_rows_t):
                    v_str = cell_data[ri][ci]
                    if v_str == '-':
                        continue
                    v = float(v_str)
                    if v == best:
                        cell_colours[ri][ci] = '#1a4a2e'   # dark green
                    elif v == worst:
                        cell_colours[ri][ci] = '#4a1a1a'   # dark red
            except (ValueError, IndexError):
                pass

        fig_t, ax_t = plt.subplots(figsize=(max(10, n_cols_t*1.6), n_rows_t*0.7 + 1.2))
        ax_t.axis('off')
        tbl = ax_t.table(
            cellText=cell_data,
            colLabels=col_labels,
            cellColours=cell_colours,
            colColours=['#21262d'] * n_cols_t,
            loc='center', cellLoc='center',
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.6)

        # Style header and cells
        for (row_i, col_i), cell in tbl.get_celld().items():
            cell.set_edgecolor('#30363d')
            cell.set_text_props(color='#c9d1d9' if row_i > 0 else '#58a6ff',
                                fontweight='bold' if row_i == 0 else 'normal')

        fig_t.patch.set_facecolor('#0f1117')
        ax_t.set_title('Recap: mean energy per token (mWh) - model x language\n'
                        'Green = best in column, Red = worst in column',
                        fontsize=10, color='#c9d1d9', pad=10)
        plt.tight_layout()
        plt.savefig(agg_dir2 / 'recap_table.png', dpi=200, bbox_inches='tight',
                    facecolor='#0f1117')
        plt.close()
        print(f"  Saved: recap_table.png")

# -- Difficulty analysis ─────────────────────────────────────────────────────
# Groups prompts by easy/medium/hard and compares energy and speed per model.
# Only runs when at least one difficulty tier other than "unclassified" exists.

if all_dfs:
    combined_diff = pd.concat([v for k,v in all_dfs.items() if not k.endswith("__timeouts")], ignore_index=True)
    diff_levels_all = [d for d in combined_diff['difficulty'].dropna().unique()
                       if d != 'unclassified']
    has_difficulty = len(diff_levels_all) > 0

    if has_difficulty:
        DIFF_ORDER = ['easy', 'medium', 'hard']
        diff_levels_sorted = [d for d in DIFF_ORDER if d in diff_levels_all] + \
                             [d for d in diff_levels_all if d not in DIFF_ORDER]

        energy_col_d = 'gpu_net_Wh' if 'gpu_net_Wh' in combined_diff.columns \
                       else 'gpu_energy_Wh'
        model_names_d = sorted(combined_diff['model'].unique())
        cmap_d = plt.cm.tab10
        colour_map_d = {m: cmap_d(i / max(len(model_names_d)-1, 1))
                        for i, m in enumerate(model_names_d)}

        diff_dir = run_dir / "comparison" / "by_difficulty"
        diff_dir.mkdir(parents=True, exist_ok=True)

        diff_stats = (
            combined_diff[(combined_diff['tokens'] > 0) &
                          (combined_diff['Wh_per_token'].notna())]
            .groupby(['difficulty', 'model'])
            .agg(
                mean_eff =('Wh_per_token', 'mean'),
                std_eff  =('Wh_per_token', 'std'),
                mean_spd =('tokens_per_s', 'mean'),
                std_spd  =('tokens_per_s', 'std'),
                mean_dur =('duration_s',   'mean'),
                std_dur  =('duration_s',   'std'),
                n        =('prompt_index', 'count'),
            )
            .reset_index()
        )
        diff_stats.to_csv(diff_dir / 'difficulty_model_summary.csv', index=False)

        x_d  = np.arange(len(diff_levels_sorted))
        n_md = len(model_names_d)
        w_d  = 0.8 / max(n_md, 1)

        def _diff_bar(ax, value_col, std_col, ylabel, title, scale=1.0):
            for i, model in enumerate(model_names_d):
                vals = []
                errs = []
                for diff in diff_levels_sorted:
                    row = diff_stats[(diff_stats['model'] == model) &
                                     (diff_stats['difficulty'] == diff)]
                    vals.append(float(row[value_col].iloc[0]) * scale
                                if not row.empty else 0.0)
                    errs.append(float(row[std_col].iloc[0]) * scale
                                if not row.empty and not np.isnan(row[std_col].iloc[0])
                                else 0.0)
                offset = (i - n_md / 2 + 0.5) * w_d
                bars = ax.bar(x_d + offset, vals, width=w_d * 0.9,
                              label=model.replace('_Q4_K_M', ''),
                              color=colour_map_d[model], alpha=0.85)
                ax.errorbar(x_d + offset, vals, yerr=errs,
                            fmt='none', color='#333333', capsize=3,
                            linewidth=0.8, alpha=0.7)
                top = max(vals) if vals else 1
                for bar, v in zip(bars, vals):
                    if v > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + top * 0.01,
                                f'{v:.3f}', ha='center', va='bottom', fontsize=6)
            ax.set_xticks(x_d)
            ax.set_xticklabels(diff_levels_sorted, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis='y')

        # 1. Energy efficiency by difficulty
        fig, ax = plt.subplots(figsize=(max(9, len(diff_levels_sorted) * 2.5), 6))
        _diff_bar(ax, 'mean_eff', 'std_eff',
                  'Mean energy per token (mWh)',
                  'Energy efficiency per difficulty tier (lower = better)', 1000)
        plt.tight_layout()
        plt.savefig(diff_dir / 'efficiency_by_difficulty.png', dpi=200, bbox_inches='tight')
        plt.close()

        # 2. Inference speed by difficulty
        fig, ax = plt.subplots(figsize=(max(9, len(diff_levels_sorted) * 2.5), 6))
        _diff_bar(ax, 'mean_spd', 'std_spd',
                  'Mean tokens / second', 'Inference speed per difficulty tier', 1.0)
        plt.tight_layout()
        plt.savefig(diff_dir / 'speed_by_difficulty.png', dpi=200, bbox_inches='tight')
        plt.close()

        # 3. Duration by difficulty
        fig, ax = plt.subplots(figsize=(max(9, len(diff_levels_sorted) * 2.5), 6))
        _diff_bar(ax, 'mean_dur', 'std_dur',
                  'Mean duration (seconds)', 'Response time per difficulty tier', 1.0)
        plt.tight_layout()
        plt.savefig(diff_dir / 'duration_by_difficulty.png', dpi=200, bbox_inches='tight')
        plt.close()

        # 4. Heatmap: model x difficulty
        heat_d = np.array([
            [diff_stats[(diff_stats['model'] == m) &
                        (diff_stats['difficulty'] == d)]['mean_eff'].mean() * 1000
             if not diff_stats[(diff_stats['model'] == m) &
                               (diff_stats['difficulty'] == d)].empty
             else np.nan
             for d in diff_levels_sorted]
            for m in model_names_d
        ])
        fig, ax = plt.subplots(figsize=(max(6, len(diff_levels_sorted) * 2),
                                        max(4, len(model_names_d) * 1.2)))
        im = ax.imshow(heat_d, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(diff_levels_sorted)))
        ax.set_yticks(range(len(model_names_d)))
        ax.set_xticklabels(diff_levels_sorted, fontsize=10)
        ax.set_yticklabels(model_names_d, fontsize=9)
        ax.set_title('Mean energy per token (mWh) - model x difficulty', fontsize=10)
        fig.colorbar(im, ax=ax, label='mWh / token')
        median_d = np.nanmedian(heat_d)
        for r in range(len(model_names_d)):
            for c in range(len(diff_levels_sorted)):
                v = heat_d[r, c]
                if not np.isnan(v):
                    ax.text(c, r, f'{v:.3f}', ha='center', va='center',
                            fontsize=9, fontweight='bold',
                            color='#0f1117' if v > median_d else '#e6edf3')
        plt.tight_layout()
        plt.savefig(diff_dir / 'efficiency_heatmap_difficulty.png', dpi=200, bbox_inches='tight')
        plt.close()

        # 5. Combined language x difficulty breakdown (if both present)
        languages_d = [l for l in combined_diff['language'].dropna().unique()
                       if l != 'unknown']
        if len(languages_d) >= 1 and len(diff_levels_sorted) >= 2:
            lang_diff_stats = (
                combined_diff[combined_diff['tokens'] > 0]
                .groupby(['language', 'difficulty', 'model'])
                .agg(mean_eff=('Wh_per_token', 'mean'), n=('prompt_index', 'count'))
                .reset_index()
            )
            lang_diff_stats.to_csv(diff_dir / 'language_difficulty_summary.csv', index=False)
            n_langs_d = len(languages_d)
            fig, axes = plt.subplots(1, n_langs_d,
                                     figsize=(max(8, n_langs_d * 5), 5),
                                     sharey=True, squeeze=False)
            fig.suptitle('Energy per token (mWh) by difficulty per language',
                         fontsize=11, fontweight='bold')
            for li, lang in enumerate(sorted(languages_d)):
                ax = axes[0][li]
                for model in model_names_d:
                    vals = [
                        float(lang_diff_stats[
                            (lang_diff_stats['language'] == lang) &
                            (lang_diff_stats['difficulty'] == diff) &
                            (lang_diff_stats['model'] == model)
                        ]['mean_eff'].iloc[0]) * 1000
                        if not lang_diff_stats[
                            (lang_diff_stats['language'] == lang) &
                            (lang_diff_stats['difficulty'] == diff) &
                            (lang_diff_stats['model'] == model)
                        ].empty else np.nan
                        for diff in diff_levels_sorted
                    ]
                    ax.plot(diff_levels_sorted, vals, marker='o', linewidth=2,
                            label=model.replace('_Q4_K_M', ''),
                            color=colour_map_d[model])
                ax.set_title(lang.upper(), fontsize=10)
                ax.set_xlabel('Difficulty', fontsize=9)
                if li == 0:
                    ax.set_ylabel('mWh / token', fontsize=9)
                ax.grid(True, alpha=0.3)
            axes[0][-1].legend(fontsize=7, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(diff_dir / 'language_difficulty_breakdown.png',
                        dpi=200, bbox_inches='tight')
            plt.close()
            print(f"  Saved: language_difficulty_breakdown.png")

        # Console summary
        print("\n" + "=" * 70)
        print("DIFFICULTY SUMMARY  (mean mWh/token | mean tok/s | n prompts)")
        print("=" * 70)
        for diff in diff_levels_sorted:
            print(f"\n  {diff.upper()}")
            sub = diff_stats[diff_stats['difficulty'] == diff].sort_values('mean_eff')
            for _, row in sub.iterrows():
                print(f"    {row['model']:<34} "
                      f"{row['mean_eff']*1000:.4f} mWh/tok  "
                      f"{row['mean_spd']:.1f} tok/s  "
                      f"({int(row['n'])} prompts)")

# -- Timeout analysis ---------------------------------------------------------
# Loads timeout_log.csv from each model directory and produces:
#   - Timeout rate per model (bar chart)
#   - Timeout heatmap: model x language
#   - Timeout rate by difficulty
#   - timeout_summary.csv in comparison/
# ------------------------------------------------------------------------------

if all_dfs:
    timeout_frames = []
    prompt_counts  = {}
    for key, df_item in all_dfs.items():
        if key.endswith("__timeouts"):
            if not df_item.empty:
                timeout_frames.append(df_item)
        else:
            prompt_counts[key] = len(df_item)

    if timeout_frames:
        combined_to = pd.concat(timeout_frames, ignore_index=True)
        for col, fill in [('language','unknown'), ('difficulty','unclassified')]:
            if col not in combined_to.columns:
                combined_to[col] = fill
            else:
                combined_to[col] = combined_to[col].fillna(fill)

        to_dir = run_dir / "comparison"
        to_dir.mkdir(exist_ok=True)

        model_names_to = sorted(combined_to['model'].unique())
        cmap_to = plt.cm.tab10
        colour_map_to = {m: cmap_to(i / max(len(model_names_to)-1, 1))
                         for i, m in enumerate(model_names_to)}

        combined_to.groupby(['model','language','difficulty']).size() \
            .reset_index(name='n_timeouts') \
            .to_csv(to_dir / 'timeout_summary.csv', index=False)

        # 1. Overall timeout rate per model
        fig, ax = plt.subplots(figsize=(max(8, len(model_names_to) * 1.8), 5))
        overall_rates = []
        for model in model_names_to:
            n_to   = len(combined_to[combined_to['model'] == model])
            n_done = prompt_counts.get(model, 0)
            overall_rates.append(n_to / max(n_to + n_done, 1) * 100)
        bars = ax.bar(range(len(model_names_to)), overall_rates,
                      color=[colour_map_to[m] for m in model_names_to], alpha=0.85)
        ax.set_xticks(range(len(model_names_to)))
        ax.set_xticklabels([m.replace('_Q4_K_M','').replace('_','\n')
                            for m in model_names_to], fontsize=9)
        ax.set_ylabel('Timeout rate (%)', fontsize=9)
        ax.set_title('Prompt timeout rate per model\n'
                     '(timeouts / total attempted prompts)',
                     fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        top_to = max(overall_rates) if overall_rates else 1
        for bar, v in zip(bars, overall_rates):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + top_to * 0.01,
                    f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(to_dir / 'timeout_rate_per_model.png', dpi=200, bbox_inches='tight')
        plt.close()

        # 2. Timeout heatmap: model x language
        languages_to = [l for l in sorted(combined_to['language'].unique())
                        if l != 'unknown']
        if languages_to:
            heat_to = np.zeros((len(model_names_to), len(languages_to)))
            for ri, model in enumerate(model_names_to):
                n_done = prompt_counts.get(model, 0)
                for ci, lang in enumerate(languages_to):
                    n_to = len(combined_to[(combined_to['model'] == model) &
                                           (combined_to['language'] == lang)])
                    heat_to[ri, ci] = n_to / max(n_to + n_done, 1) * 100
            fig, ax = plt.subplots(figsize=(max(6, len(languages_to)*1.8),
                                            max(4, len(model_names_to)*1.2)))
            im = ax.imshow(heat_to, cmap='YlOrRd', aspect='auto', vmin=0)
            ax.set_xticks(range(len(languages_to)))
            ax.set_yticks(range(len(model_names_to)))
            ax.set_xticklabels(languages_to, rotation=25, ha='right', fontsize=9)
            ax.set_yticklabels(model_names_to, fontsize=9)
            ax.set_title('Timeout rate (%) - model x language',
                         fontsize=10, fontweight='bold')
            fig.colorbar(im, ax=ax, label='Timeout rate (%)')
            median_to = np.nanmedian(heat_to)
            for r in range(len(model_names_to)):
                for c in range(len(languages_to)):
                    v = heat_to[r, c]
                    ax.text(c, r, f'{v:.1f}%', ha='center', va='center',
                            fontsize=8, fontweight='bold',
                            color='#0f1117' if v > median_to else '#e6edf3')
            plt.tight_layout()
            plt.savefig(to_dir / 'timeout_heatmap_language.png', dpi=200, bbox_inches='tight')
            plt.close()

        # 3. Timeout rate by difficulty
        diffs_to = [d for d in ['easy','medium','hard']
                    if d in combined_to['difficulty'].unique()]
        if diffs_to:
            fig, ax = plt.subplots(figsize=(max(7, len(diffs_to)*2.5), 5))
            x_d = np.arange(len(diffs_to))
            w_d = 0.8 / max(len(model_names_to), 1)
            for i, model in enumerate(model_names_to):
                n_done = prompt_counts.get(model, 0)
                vals = []
                for diff in diffs_to:
                    n_to = len(combined_to[(combined_to['model'] == model) &
                                           (combined_to['difficulty'] == diff)])
                    vals.append(n_to / max(n_to + n_done, 1) * 100)
                offset = (i - len(model_names_to)/2 + 0.5) * w_d
                ax.bar(x_d + offset, vals, width=w_d*0.9,
                       label=model.replace('_Q4_K_M',''),
                       color=colour_map_to[model], alpha=0.85)
            ax.set_xticks(x_d)
            ax.set_xticklabels(diffs_to, fontsize=10)
            ax.set_ylabel('Timeout rate (%)', fontsize=9)
            ax.set_title('Timeout rate per difficulty tier',
                         fontsize=11, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            plt.savefig(to_dir / 'timeout_rate_by_difficulty.png', dpi=200, bbox_inches='tight')
            plt.close()

        # Console summary
        print("\n" + "=" * 70)
        print("TIMEOUT SUMMARY")
        print("=" * 70)
        for model in model_names_to:
            n_to   = len(combined_to[combined_to['model'] == model])
            n_done = prompt_counts.get(model, 0)
            rate   = n_to / max(n_to + n_done, 1) * 100
            print(f"  {model:<34} {n_to:>4} timeouts / {n_to+n_done:>4} attempted "
                  f"({rate:.1f}%)")
    else:
        print("  No timeouts recorded in this run.")

# ── Console summary ───────────────────────────────────────────────────────────
if all_models_data:
    df_compare = (pd.DataFrame(all_models_data)
                  .sort_values('total_energy_Wh')
                  .reset_index(drop=True))

    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<30} {'Total Wh':<12} {'Wh/1K tok':<13} {'tok/s':<10} {'Throttle%'}")
    print("-" * 80)
    for _, row in df_compare.iterrows():
        print(f"{row['name']:<30} {row['total_energy_Wh']:<12.3f} "
              f"{row['energy_per_1k_tokens_Wh']:<13.4f} "
              f"{row['avg_tokens_per_s']:<10.1f} "
              f"{row['throttle_pct']:.1f}%")

    if per_prompt_data:
        print("\n" + "=" * 80)
        print("FIRST 5 PROMPTS - PER-MODEL BREAKDOWN")
        print("=" * 80)
        for p in sorted(per_prompt_data.keys())[:5]:
            print(f"\nPrompt {p}:")
            for m in sorted(per_prompt_data[p].keys()):
                d = per_prompt_data[p][m]
                print(f"  {m:<32} {d['gpu_energy_Wh']:.5f} Wh  "
                      f"{d['tokens_per_s']:.1f} tok/s  "
                      f"{d['Wh_per_token']*1000:.3f} mWh/tok")

    if len(df_compare) >= 2:
        best_eff   = df_compare.loc[df_compare['energy_per_1k_tokens_Wh'].idxmin()]
        best_speed = df_compare.loc[df_compare['avg_tokens_per_s'].idxmax()]
        lowest_e   = df_compare.loc[df_compare['total_energy_Wh'].idxmin()]

        print("\n" + "=" * 80)
        print("BEST IN CLASS")
        print("=" * 80)
        print(f"Best efficiency : {best_eff['name']}  "
              f"({best_eff['energy_per_1k_tokens_Wh']:.4f} Wh/1K tokens)")
        print(f"Fastest         : {best_speed['name']}  "
              f"({best_speed['avg_tokens_per_s']:.1f} tokens/s)")
        print(f"Lowest energy   : {lowest_e['name']}  "
              f"({lowest_e['total_energy_Wh']:.3f} Wh total)")

print("\nAll graphs generated successfully.\n")
