#!/usr/bin/env python3
"""
aggregate_analysis.py -- Cross-run statistical analysis for the LLM Energy Benchmark.

Expects a session directory containing run_1/, run_2/, ... run_N/ subdirectories,
each produced by analyze_results.py after its corresponding run.

Outputs go to <session_dir>/aggregate/:
  summary_stats.csv          -- mean, std, cv, p25, p75 per (model, language, difficulty)
  mean_energy_errorbars.png  -- bar chart with error bars (mean +- std)
  cv_heatmap.png             -- coefficient of variation: consistency across runs
  outlier_report.csv         -- prompts flagged as outliers (IQR method)
  per_run_comparison.png     -- overlaid bar chart showing each run side by side
  run_consistency.png        -- how much each model's energy varies across runs

Usage:
  python3 aggregate_analysis.py <session_dir>
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

# Clean display names for models (shared convention with analyze_results.py)
_MODEL_DISPLAY = {
    'gemma3_4b': 'Gemma 3 4B', 'mistral_7b': 'Mistral 7B',
    'qwen2_5_7b': 'Qwen 2.5 7B', 'qwen2_5_coder_7b': 'Qwen 2.5 Coder 7B',
    'qwen2_5_coder_32b': 'Qwen 2.5 Coder 32B', 'qwen3_32b': 'Qwen 3 32B',
    'gpt-oss-20b': 'GPT-OSS 20B', 'gpt-oss-120b': 'GPT-OSS 120B',
}
def display_name(raw):
    import re as _re
    s = str(raw)
    for junk in ['.Q4_K_M', '_Q4_K_M', '-mxfp4', '.gguf']:
        s = s.replace(junk, '')
    s = _re.sub(r'-?\d{5}-of-\d{5}', '', s)
    key = s.strip().rstrip('.').replace('.', '_')
    if key in _MODEL_DISPLAY: return _MODEL_DISPLAY[key]
    base = key.split('_Q4')[0]
    if base in _MODEL_DISPLAY: return _MODEL_DISPLAY[base]
    return s.replace('_', ' ').strip()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

SKIP_DIRS = {"idle", "venv", "aggregate", "comparison"}
DIFF_ORDER = ["easy", "medium", "hard", "unclassified"]

# ── Colour config (matches analyze_results.py palette) ───────────────────────
def build_colour_map(names):
    cmap = plt.cm.tab10
    n = max(len(names) - 1, 1)
    return {name: cmap(i / n) for i, name in enumerate(sorted(names))}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_run(run_dir: Path) -> pd.DataFrame | None:
    """Load all prompt_energy.csv files from a single run directory."""
    frames = []
    for model_dir in sorted(run_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name in SKIP_DIRS:
            continue
        csv = model_dir / "prompt_energy.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv, sep=";", na_values=["N/A", "n/a", ""])
        df["model"] = model_dir.name
        df["run"]   = run_dir.name
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def load_all_runs(session_dir: Path) -> pd.DataFrame | None:
    """Load data from all run_N/ subdirectories."""
    run_dirs = sorted(
        d for d in session_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_")
    )
    if not run_dirs:
        print("No run_N/ directories found.")
        return None

    frames = []
    for run_dir in run_dirs:
        df = load_run(run_dir)
        if df is not None:
            frames.append(df)
            print(f"  Loaded: {run_dir.name} ({len(df)} prompt rows)")

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)

    # Normalise columns
    numeric = [
        "cpu_avg_W", "cpu_max_W", "gpu_avg_W", "gpu_max_W",
        "duration_s", "cpu_energy_Wh", "gpu_energy_Wh",
        "cpu_net_Wh", "gpu_net_Wh", "tokens", "tokens_per_s",
        "Wh_per_token", "throttled",
    ]
    for col in numeric:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined["tokens"]       = combined["tokens"].fillna(0).astype(int)
    combined["throttled"]    = combined["throttled"].fillna(0).astype(int)
    combined["language"]     = combined["language"].fillna("unknown") \
        if "language" in combined.columns else "unknown"
    combined["difficulty"]   = combined["difficulty"].fillna("unclassified") \
        if "difficulty" in combined.columns else "unclassified"

    # Recompute tokens_per_s and Wh_per_token (same logic as analyze_results.py)
    energy_col = "gpu_net_Wh" if "gpu_net_Wh" in combined.columns else "gpu_energy_Wh"
    combined["tokens_per_s"] = combined.apply(
        lambda r: r["tokens_per_s"] if r["tokens_per_s"] > 0
        else (r["tokens"] / r["duration_s"] if r["duration_s"] > 0 and r["tokens"] > 0 else 0.0),
        axis=1
    )
    combined["Wh_per_token"] = combined.apply(
        lambda r: r[energy_col] / r["tokens"]
        if r["tokens"] > 0 and pd.notna(r[energy_col]) and r[energy_col] > 0
        else np.nan,
        axis=1
    )
    combined["energy_col"] = energy_col
    return combined


# ── Outlier detection (IQR method) ───────────────────────────────────────────

def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag rows where Wh_per_token is outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
    within each (model, prompt_index, language) group.
    With only 3 runs, IQR is conservative -- it mainly catches real failures.
    """
    records = []
    group_cols = ["model", "prompt_index"]
    if "language" in df.columns:
        group_cols.append("language")

    for keys, grp in df.groupby(group_cols):
        vals = grp["Wh_per_token"].dropna()
        if len(vals) < 2:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = (grp["Wh_per_token"] < lo) | (grp["Wh_per_token"] > hi)
        for idx in grp[outlier_mask].index:
            row = grp.loc[idx]
            record = {
                "run":          row.get("run", ""),
                "model":        row["model"],
                "prompt_index": row["prompt_index"],
                "Wh_per_token": row["Wh_per_token"],
                "iqr_lo":       round(lo, 8),
                "iqr_hi":       round(hi, 8),
            }
            if "language" in row:
                record["language"] = row["language"]
            if "difficulty" in row:
                record["difficulty"] = row["difficulty"]
            records.append(record)

    return pd.DataFrame(records)


# ── Aggregate statistics ──────────────────────────────────────────────────────

def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (model, language, difficulty): mean, std, CV, p25, p75, n_runs.

    The coefficient of variation here measures consistency ACROSS RUNS, not the
    spread of individual prompts. We first reduce each run to a single mean, then
    take the standard deviation over those run-means. This matches how CV is
    defined in the dissertation (inter-run variability); computing it over all
    pooled prompts instead would report per-prompt spread, which is much larger
    for short-output and high-timeout models and does not describe reproducibility.
    """
    group_cols = ["model"]
    if "language" in df.columns:
        group_cols.append("language")
    if "difficulty" in df.columns:
        group_cols.append("difficulty")

    valid = df[df["tokens"] > 0]

    # One mean per run first.
    per_run = (
        valid.groupby(group_cols + ["run"])
        .agg(
            run_mean_eff=("Wh_per_token", "mean"),
            run_mean_spd=("tokens_per_s", "mean"),
        )
        .reset_index()
    )

    # Then spread across the run-means.
    stats = (
        per_run.groupby(group_cols)
        .agg(
            mean_eff   =("run_mean_eff", "mean"),
            std_eff    =("run_mean_eff", "std"),
            p25_eff    =("run_mean_eff", lambda x: x.quantile(0.25)),
            p75_eff    =("run_mean_eff", lambda x: x.quantile(0.75)),
            mean_spd   =("run_mean_spd", "mean"),
            std_spd    =("run_mean_spd", "std"),
            n_runs     =("run",          "count"),
        )
        .reset_index()
    )
    stats["cv_eff"] = stats["std_eff"] / stats["mean_eff"].replace(0, np.nan)
    return stats


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_mean_errorbars(stats: pd.DataFrame, out_dir: Path, colour_map: dict):
    """
    Bar chart: mean Wh/token per model with +- 1 std error bars.
    One chart overall, one per language if language data present.
    """
    def _one_chart(sub, title, fname):
        models = sorted(sub["model"].unique())
        x      = np.arange(len(models))
        vals   = [sub[sub["model"] == m]["mean_eff"].mean() * 1000 for m in models]
        errs   = [sub[sub["model"] == m]["std_eff"].mean() * 1000 for m in models]

        fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.8), 6))
        bars = ax.bar(x, vals, color=[colour_map[m] for m in models],
                      alpha=0.85, width=0.6)
        ax.errorbar(x, vals, yerr=errs, fmt="none", color="#333333",
                    capsize=5, linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_Q4_K_M", "").replace("_", "\n")
                            for m in models], fontsize=9)
        ax.set_ylabel("Mean energy per token (mWh) +- 1 std", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        top = max(vals) if vals else 1
        for bar, v, e in zip(bars, vals, errs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + top * 0.02,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=200, bbox_inches="tight")
        plt.close()

    _one_chart(stats, "Mean energy per token across all runs (lower = better)",
               "mean_energy_errorbars.png")

    if "language" in stats.columns:
        for lang in sorted(stats["language"].unique()):
            if lang == "unknown":
                continue
            sub = stats[stats["language"] == lang]
            _one_chart(sub,
                       f"Mean energy per token -- {lang.upper()}",
                       f"mean_energy_{lang}.png")


def plot_cv_heatmap(stats: pd.DataFrame, out_dir: Path):
    """
    Heatmap of coefficient of variation (std/mean) for Wh/token.
    Low CV = consistent across runs (good). High CV = variable (unreliable).
    Rows: models. Columns: languages (or single "all" column).
    """
    models = sorted(stats["model"].unique())
    if "language" in stats.columns:
        langs = [l for l in sorted(stats["language"].unique()) if l != "unknown"]
    else:
        langs = ["all"]

    matrix = np.full((len(models), len(langs)), np.nan)
    for ri, model in enumerate(models):
        for ci, lang in enumerate(langs):
            if lang == "all":
                sub = stats[stats["model"] == model]
            else:
                sub = stats[(stats["model"] == model) & (stats["language"] == lang)]
            if not sub.empty:
                matrix[ri, ci] = sub["cv_eff"].mean()

    fig, ax = plt.subplots(figsize=(max(6, len(langs) * 1.8),
                                    max(4, len(models) * 1.2)))
    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=min(1.0, np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1.0))
    ax.set_xticks(range(len(langs)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(langs, rotation=25, ha="right", fontsize=9)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_title("Run consistency (CV = std/mean of Wh/token)\n"
                 "Green = consistent across runs, Red = variable",
                 fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, label="CV (lower = more consistent)")
    median_cv = np.nanmedian(matrix)
    for r in range(len(models)):
        for c in range(len(langs)):
            v = matrix[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="#0f1117" if v > median_cv else "#e6edf3")
    plt.tight_layout()
    plt.savefig(out_dir / "cv_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_per_run_comparison(df: pd.DataFrame, out_dir: Path, colour_map: dict):
    """
    For each model, show one bar per run so you can see if energy is stable
    or drifting (e.g. thermal effects in run 3 vs run 1).
    """
    models  = sorted(df["model"].unique())
    runs    = sorted(df["run"].unique())
    energy_col = df["energy_col"].iloc[0] if "energy_col" in df.columns else "gpu_net_Wh"

    n_models = len(models)
    n_runs   = len(runs)
    x        = np.arange(n_models)
    w        = 0.8 / max(n_runs, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_models * 2), 6))
    run_colours = plt.cm.Set2(np.linspace(0, 0.8, n_runs))

    for ri, run in enumerate(runs):
        vals = []
        for model in models:
            sub = df[(df["model"] == model) & (df["run"] == run) &
                     (df["tokens"] > 0) & (df["Wh_per_token"].notna())]
            vals.append(sub["Wh_per_token"].mean() * 1000 if not sub.empty else 0)
        offset = (ri - n_runs / 2 + 0.5) * w
        ax.bar(x + offset, vals, width=w * 0.9,
               label=run, color=run_colours[ri], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_Q4_K_M", "").replace("_", "\n")
                        for m in models], fontsize=9)
    ax.set_ylabel("Mean mWh per token")
    ax.set_title("Energy per token per run per model\n"
                 "(stable bars = reproducible results)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, title="Run")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "per_run_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_timeout_analysis(df, combined_to, out_dir, colour_map):
    """Cross-run timeout analysis: overall rate, per-run rate, model x language heatmap."""
    if combined_to is None or combined_to.empty:
        print("  No timeouts recorded across any run.")
        return

    models = sorted(df["model"].unique())
    runs   = sorted(df["run"].unique())
    prompt_counts = df.groupby(["model","run"]).size().reset_index(name="n_done")

    # 1. Overall timeout rate per model
    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.8), 5))
    rates = []
    for model in models:
        n_to   = len(combined_to[combined_to["model"] == model])
        n_done = int(prompt_counts[prompt_counts["model"] == model]["n_done"].sum())
        rates.append(n_to / max(n_to + n_done, 1) * 100)
    bars = ax.bar(range(len(models)), rates,
                  color=[colour_map[m] for m in models], alpha=0.85)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace("_Q4_K_M","").replace("_"," ")
                        for m in models], fontsize=9)
    ax.set_ylabel("Timeout rate (%) across all runs", fontsize=9)
    ax.set_title("Aggregate timeout rate per model", fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    top = max(rates) if rates else 1
    for bar, v in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + top * 0.01,
                f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "timeout_rate_aggregate.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 2. Timeout rate per run per model
    n_runs = len(runs)
    x      = np.arange(len(models))
    w      = 0.8 / max(n_runs, 1)
    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2), 5))
    run_colours = plt.cm.Set2(np.linspace(0, 0.8, n_runs))
    for ri, run in enumerate(runs):
        vals = []
        for model in models:
            n_to = len(combined_to[(combined_to["model"] == model) &
                                   (combined_to["run"] == run)])
            sub  = prompt_counts[(prompt_counts["model"] == model) &
                                  (prompt_counts["run"]   == run)]["n_done"]
            n_done = int(sub.iloc[0]) if not sub.empty else 0
            vals.append(n_to / max(n_to + n_done, 1) * 100)
        offset = (ri - n_runs/2 + 0.5) * w
        ax.bar(x + offset, vals, width=w*0.9,
               label=run, color=run_colours[ri], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_Q4_K_M","").replace("_"," ")
                        for m in models], fontsize=9)
    ax.set_ylabel("Timeout rate (%)", fontsize=9)
    ax.set_title("Timeout rate per run per model", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, title="Run")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "timeout_per_run.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 3. Heatmap: model x language
    if "language" in combined_to.columns:
        langs = [l for l in sorted(combined_to["language"].unique()) if l != "unknown"]
        if langs:
            heat = np.zeros((len(models), len(langs)))
            for ri, model in enumerate(models):
                n_done = int(prompt_counts[prompt_counts["model"]==model]["n_done"].sum())
                for ci, lang in enumerate(langs):
                    n_to = len(combined_to[(combined_to["model"]==model) &
                                           (combined_to["language"]==lang)])
                    heat[ri, ci] = n_to / max(n_to + n_done, 1) * 100
            fig, ax = plt.subplots(figsize=(max(6, len(langs)*1.8),
                                            max(4, len(models)*1.2)))
            im = ax.imshow(heat, cmap="YlOrRd", aspect="auto", vmin=0)
            ax.set_xticks(range(len(langs)))
            ax.set_yticks(range(len(models)))
            ax.set_xticklabels(langs, rotation=25, ha="right", fontsize=9)
            ax.set_yticklabels(models, fontsize=9)
            ax.set_title("Aggregate timeout rate (%) - model x language",
                         fontsize=10, fontweight="bold")
            fig.colorbar(im, ax=ax, label="Timeout rate (%)")
            median_h = np.nanmedian(heat)
            for r in range(len(models)):
                for c in range(len(langs)):
                    v = heat[r, c]
                    ax.text(c, r, f"{v:.1f}%", ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="#0f1117" if v > median_h else "#e6edf3")
            plt.tight_layout()
            plt.savefig(out_dir / "timeout_heatmap_language.png",
                        dpi=200, bbox_inches="tight")
            plt.close()


plt.rcParams.update({'font.size': 13, 'axes.titlesize': 15, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11})


def plot_run_consistency(df: pd.DataFrame, out_dir: Path, colour_map: dict):
    """
    Line plot: one line per model, x=run number, y=mean Wh/token.
    A flat line = perfectly reproducible. Rising line = thermal drift.
    """
    models = sorted(df["model"].unique())
    runs   = sorted(df["run"].unique())

    fig, ax = plt.subplots(figsize=(max(9, len(runs) * 2.4), 5.5))
    for model in models:
        vals = []
        for run in runs:
            sub = df[(df["model"] == model) & (df["run"] == run) &
                     (df["tokens"] > 0) & (df["Wh_per_token"].notna())]
            vals.append(sub["Wh_per_token"].mean() * 1000 if not sub.empty else np.nan)
        ax.plot(runs, vals, marker="o", linewidth=2, markersize=7,
                label=display_name(model),
                color=colour_map[model])
        # Annotate first and last point
        if vals and not np.isnan(vals[0]):
            ax.annotate(f"{vals[0]:.3f}", (runs[0], vals[0]),
                        textcoords="offset points", xytext=(-18, 6), fontsize=7)
        if vals and not np.isnan(vals[-1]):
            ax.annotate(f"{vals[-1]:.3f}", (runs[-1], vals[-1]),
                        textcoords="offset points", xytext=(4, 6), fontsize=7)

    ax.set_xlabel("Run")
    ax.set_ylabel("Mean mWh per token")
    ax.set_title("Energy consistency across runs\n"
                 "(flat line = reproducible, rising = thermal drift)")
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5),
              frameon=False)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "run_consistency.png", dpi=200, bbox_inches="tight")
    plt.close()


# ── Console summary ───────────────────────────────────────────────────────────

def print_summary(stats, outliers, n_runs, n_outliers_total,
                  combined_to=None, df=None):
    print("\n" + "=" * 70)
    print(f"AGGREGATE SUMMARY  ({n_runs} runs)")
    print("=" * 70)
    print(f"\n{'Model':<34} {'Mean mWh/tok':<14} {'Std':<10} {'CV':<8} {'tok/s'}")
    print("-" * 70)
    for _, row in stats.groupby("model").agg(
        mean_eff=("mean_eff", "mean"),
        std_eff =("std_eff",  "mean"),
        cv_eff  =("cv_eff",   "mean"),
        mean_spd=("mean_spd", "mean"),
    ).reset_index().sort_values("mean_eff").iterrows():
        print(f"  {row['model']:<32} "
              f"{row['mean_eff']*1000:<14.4f} "
              f"{row['std_eff']*1000:<10.4f} "
              f"{row['cv_eff']:<8.3f} "
              f"{row['mean_spd']:.1f}")

    print(f"\nOutlier prompts detected: {n_outliers_total}")
    if n_outliers_total > 0:
        print("  See aggregate/outlier_report.csv for details.")

    if combined_to is not None and not combined_to.empty and df is not None:
        print("\nTimeout summary (across all runs):")
        prompt_counts = df.groupby("model").size().reset_index(name="n_done")
        for _, row in prompt_counts.iterrows():
            model  = row["model"]
            n_to   = len(combined_to[combined_to["model"] == model])
            n_done = row["n_done"]
            rate   = n_to / max(n_to + n_done, 1) * 100
            print(f"  {model:<34} {n_to:>4} timeouts / {n_to+n_done:>4} "
                  f"attempted ({rate:.1f}%)")

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 aggregate_analysis.py <session_dir>")
        sys.exit(1)

    session_dir = Path(sys.argv[1])
    print(f"\nAggregate analysis: {session_dir}\n")

    df = load_all_runs(session_dir)
    if df is None or df.empty:
        print("No data found.")
        sys.exit(1)

    combined_to = df.attrs.get("timeouts",
                               __import__("pandas").DataFrame(
                                   columns=["timestamp","prompt_index",
                                            "language","difficulty","model","run"]))

    out_dir = session_dir / "aggregate"
    out_dir.mkdir(exist_ok=True)

    models     = sorted(df["model"].unique())
    colour_map = build_colour_map(models)
    n_runs     = df["run"].nunique()

    print(f"\nModels: {len(models)}")
    print(f"Runs:   {n_runs}")
    print(f"Total prompt rows: {len(df)}")

    # Stats
    stats = compute_stats(df)
    stats.to_csv(out_dir / "summary_stats.csv", index=False)
    print(f"\nSummary stats -> aggregate/summary_stats.csv")

    # Outliers
    outliers = detect_outliers(df)
    outliers.to_csv(out_dir / "outlier_report.csv", index=False)
    print(f"Outliers found: {len(outliers)} -> aggregate/outlier_report.csv")

    # Plots
    print("\nGenerating aggregate plots...")
    plot_mean_errorbars(stats, out_dir, colour_map)
    print("  mean_energy_errorbars.png")

    plot_cv_heatmap(stats, out_dir)
    print("  cv_heatmap.png")

    plot_per_run_comparison(df, out_dir, colour_map)
    print("  per_run_comparison.png")

    plot_run_consistency(df, out_dir, colour_map)
    print("  run_consistency.png")

    plot_timeout_analysis(df, combined_to, out_dir, colour_map)
    if not combined_to.empty:
        print("  timeout_rate_aggregate.png")
        print("  timeout_per_run.png")

    print_summary(stats, outliers, n_runs, len(outliers), combined_to, df)
    print(f"\nAll aggregate outputs saved to: {out_dir}\n")


if __name__ == "__main__":
    main()
