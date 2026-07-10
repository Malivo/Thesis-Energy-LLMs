#!/usr/bin/env python3
"""
benchmark_gui.py - Configuration GUI for the LLM Energy Benchmark.

Reads benchmark_config.sh on startup, lets the user edit all settings,
writes the config back, and can launch benchmark.sh directly.

Both files must live in the same directory.
Run with:  python3 benchmark_gui.py
"""

import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
CONFIG_FILE  = SCRIPT_DIR / "benchmark_config.sh"
INSTALL_SCRIPT = SCRIPT_DIR / "install_models.sh"
RERUN_SCRIPT   = SCRIPT_DIR / "rerun_analysis.sh"
# BENCH_FILE is resolved at runtime from the config (BENCH_SCRIPT setting)

# ── Installable model catalogue ───────────────────────────────────────────────
# Must match the definitions in install_models.sh.
# gguf_rel: path relative to MODELS_BASE_DIR (~HOME/models by default)
INSTALLABLE_MODELS = [
    # id, display name, total params (B), gguf path relative to ~/models,
    # actual_vram_gb (None = use formula), notes
    {"id": "gemma3_4b",        "name": "Gemma 3 4B (IT)",        "params_b": 4,   "actual_vram_gb": None, "gguf_rel": "gemma/gemma3_4b.Q4_K_M.gguf",               "notes": "Requires HF access approval"},
    {"id": "qwen2_5_7b",       "name": "Qwen 2.5 7B",            "params_b": 7,   "actual_vram_gb": None, "gguf_rel": "qwen/qwen2_5_7b.Q4_K_M.gguf",               "notes": ""},
    {"id": "qwen2_5_coder_7b", "name": "Qwen 2.5 Coder 7B",      "params_b": 7,   "actual_vram_gb": None, "gguf_rel": "qwen/qwen2_5_coder_7b.Q4_K_M.gguf",         "notes": ""},
    {"id": "mistral_7b",       "name": "Mistral 7B Instruct",     "params_b": 7,   "actual_vram_gb": None, "gguf_rel": "mistral/mistral_7b.Q4_K_M.gguf",            "notes": ""},
    {"id": "qwen3_32b",        "name": "Qwen 3 32B",              "params_b": 32,  "actual_vram_gb": None, "gguf_rel": "qwen3/qwen3_32b.Q4_K_M.gguf",               "notes": "~19 GB VRAM"},
    {"id": "qwen2_5_coder_32b","name": "Qwen 2.5 Coder 32B",       "params_b": 32,  "actual_vram_gb": None, "gguf_rel": "qwen/qwen2_5_coder_32b.Q4_K_M.gguf",       "notes": "~19 GB VRAM"},
    # gpt-oss: MoE models with native MXFP4 quantization.
    # Pre-built GGUFs downloaded from ggml-org - no conversion or re-quantization.
    # Require --jinja flag at runtime for the harmony response format.
    # VRAM figures are empirically confirmed, not formula-estimated.
    {"id": "gpt_oss_20b",      "name": "GPT OSS 20B (MoE)",       "params_b": 21,  "actual_vram_gb": 16,   "gguf_rel": "gpt-oss-20b/gguf/gpt-oss-20b-mxfp4.gguf",  "notes": "MXFP4 native. Needs --jinja. 16 GB VRAM."},
    {"id": "gpt_oss_120b",     "name": "GPT OSS 120B (MoE)",      "params_b": 117, "actual_vram_gb": 80,   "gguf_rel": "gpt-oss-120b/gguf/gpt-oss-120b-mxfp4.gguf","notes": "MXFP4 native. Needs --jinja. 80 GB VRAM (H100)."},
]

def _estimate_vram_gb(model: dict) -> float:
    """Return VRAM estimate in GB.
    Uses actual_vram_gb if provided (empirically known),
    otherwise falls back to Q4_K_M formula: params × 4.5/8 + 1.5 GB."""
    if model.get("actual_vram_gb") is not None:
        return float(model["actual_vram_gb"])
    return model["params_b"] * 4.5 / 8 + 1.5

def _detect_vram_mb() -> int:
    """Return total GPU VRAM in MB, or 0 if nvidia-smi not available."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return 0

def _detect_ram_mb() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 0

def _detect_disk_gb(path: str) -> int:
    try:
        r = subprocess.run(["df", "-BG", path],
                           capture_output=True, text=True, timeout=5)
        return int(r.stdout.splitlines()[1].split()[3].replace("G", ""))
    except Exception:
        return 0

# ── Colour palette (dark industrial) ─────────────────────────────────────────
BG       = "#0f1117"   # window background
BG2      = "#161b22"   # frame / input background
BORDER   = "#30363d"   # subtle borders
FG       = "#c9d1d9"   # primary text
FG_DIM   = "#8b949e"   # secondary / label text
ACCENT   = "#58a6ff"   # blue accent (focus rings, headings)
GREEN    = "#3fb950"   # launch button
RED      = "#f78166"   # destructive / warning
MONO     = ("Courier New", 10) if sys.platform == "win32" else ("Monospace", 10)
MONO_LG  = ("Courier New", 11) if sys.platform == "win32" else ("Monospace", 11)

# ── Config parser ─────────────────────────────────────────────────────────────

def _parse_config(path: Path) -> dict:
    """Parse a bash config file into a Python dict.
    Handles scalar KEY="value" and the MODELS=(...) array."""
    cfg = {
        "DURATION":       "3600",
        "INTERVAL":       "1",
        "IDLE_DURATION":  "60",
        "PROMPT_TIMEOUT": "300",
        "MIN_FREE_MB":    "500",
        "BASE_LOGDIR":    "",
        "LLAMA_BIN":      "",
        "PROMPT_DIR":     "",
        "WARMUP_PROMPT":  "Say hello.",
        "BENCH_SCRIPT":   "run_energy_benchmark.sh",
        "RUN_MODE":       "duration",
        "MODELS":         [],
    }
    if not path.exists():
        return cfg

    text = path.read_text()

    # MODELS array - capture everything between ( and closing )
    m = re.search(r'MODELS=\(\s*(.*?)\s*\)', text, re.DOTALL)
    if m:
        raw = m.group(1)
        cfg["MODELS"] = [
            s.strip().strip('"').strip("'")
            for s in raw.splitlines()
            if s.strip() and not s.strip().startswith("#")
        ]

    # Scalar KEY="value" or KEY=value
    for key in ["DURATION", "INTERVAL", "IDLE_DURATION", "PROMPT_TIMEOUT",
                "MIN_FREE_MB", "BASE_LOGDIR", "LLAMA_BIN", "PROMPT_DIR",
                "WARMUP_PROMPT", "BENCH_SCRIPT", "RUN_MODE"]:
        pat = rf'^{key}=["\']?(.*?)["\']?\s*(?:#.*)?$'
        hit = re.search(pat, text, re.MULTILINE)
        if hit:
            cfg[key] = hit.group(1).strip().strip('"').strip("'")

    return cfg


def _write_config(path: Path, cfg: dict):
    """Write the config dict back to a bash-sourceable file."""
    models_lines = "\n".join(f'  "{m}"' for m in cfg["MODELS"] if m.strip())
    content = f"""\
# ============================================
# Benchmark Configuration
# Generated by benchmark_gui.py - edit here
# or use the GUI to update these values.
# ============================================

# ----- Timing --------------------------------
DURATION={cfg['DURATION']}
INTERVAL={cfg['INTERVAL']}
IDLE_DURATION={cfg['IDLE_DURATION']}
PROMPT_TIMEOUT={cfg['PROMPT_TIMEOUT']}
RUN_MODE={cfg['RUN_MODE']}
N_RUNS={cfg['N_RUNS']}

# ----- Disk space guard ----------------------
MIN_FREE_MB={cfg['MIN_FREE_MB']}

# ----- Paths ---------------------------------
BASE_LOGDIR="{cfg['BASE_LOGDIR']}"
LLAMA_BIN="{cfg['LLAMA_BIN']}"
PROMPT_DIR="{cfg['PROMPT_DIR']}"
BENCH_SCRIPT="{cfg['BENCH_SCRIPT']}"

# ----- Warmup --------------------------------
WARMUP_PROMPT="{cfg['WARMUP_PROMPT']}"

# ----- Models --------------------------------
# One path per element. The GUI rewrites this entire block.
MODELS=(
{models_lines}
)
"""
    path.write_text(content)


# ── Reusable widget helpers ───────────────────────────────────────────────────

def _label(parent, text, small=False):
    font = (MONO[0], 9) if small else MONO
    return tk.Label(parent, text=text, bg=BG2, fg=FG_DIM,
                    font=font, anchor="w")


def _entry(parent, width=30, **kw):
    e = tk.Entry(parent, width=width, bg=BG2, fg=FG,
                 insertbackground=FG, relief="flat",
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT, font=MONO, **kw)
    return e


def _section(parent, text):
    """A coloured section divider label."""
    return tk.Label(parent, text=f"  {text}",
                    bg=BG, fg=ACCENT, font=(MONO[0], 9, "bold"),
                    anchor="w", pady=4)


def _hline(parent):
    return tk.Frame(parent, bg=BORDER, height=1)


def _button(parent, text, command, colour=ACCENT, width=14):
    return tk.Button(
        parent, text=text, command=command,
        bg=colour, fg=BG, activebackground=FG, activeforeground=BG,
        relief="flat", font=(MONO[0], 10, "bold"),
        width=width, cursor="hand2", pady=4,
    )


# ── Main application ──────────────────────────────────────────────────────────

class BenchmarkGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LLM Energy Benchmark - Configuration")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)
        self.root.minsize(640, 520)

        self.cfg = _parse_config(CONFIG_FILE)
        self._proc         = None   # running benchmark subprocess
        self._install_proc = None   # running installer subprocess
        self._output_queue = queue.Queue()  # thread-safe stdout buffer
        self._install_queue = queue.Queue()

        self._build_ui()
        self._populate()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # Title bar
        title = tk.Label(self.root,
                         text="  LLM Energy Benchmark",
                         bg=BG, fg=ACCENT,
                         font=(MONO[0], 14, "bold"), anchor="w", padx=16, pady=10)
        title.pack(fill="x")
        _hline(self.root).pack(fill="x")

        # Notebook
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook",
                        background=BG, borderwidth=0, tabmargins=[0, 0, 0, 0])
        style.configure("TNotebook.Tab",
                        background=BG2, foreground=FG_DIM,
                        font=MONO_LG, padding=[16, 6], borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", BG)],
                  foreground=[("selected", ACCENT)])
        style.configure("TFrame", background=BG)

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=0, pady=0)

        self._tab_basic    = ttk.Frame(nb)
        self._tab_advanced = ttk.Frame(nb)
        self._tab_run      = ttk.Frame(nb)
        self._tab_install  = ttk.Frame(nb)

        nb.add(self._tab_basic,    text="  Basic  ")
        nb.add(self._tab_advanced, text="  Advanced  ")
        nb.add(self._tab_run,      text="  Run  ")
        nb.add(self._tab_install,  text="  Install  ")

        self._build_basic(self._tab_basic)
        self._build_advanced(self._tab_advanced)
        self._build_run(self._tab_run)
        self._build_install(self._tab_install)

        # Bottom bar
        _hline(self.root).pack(fill="x")
        bar = tk.Frame(self.root, bg=BG, pady=8, padx=12)
        bar.pack(fill="x")

        self._status = tk.Label(bar, text="Ready.", bg=BG, fg=FG_DIM,
                                font=(MONO[0], 9), anchor="w")
        self._status.pack(side="left", fill="x", expand=True)

        _button(bar, "Rerun analysis", self._rerun_analysis, colour=ACCENT, width=14).pack(side="right", padx=4)
        _button(bar, "Save config", self._save, colour=ACCENT, width=12).pack(side="right", padx=4)
        _button(bar, "▶  Launch", self._launch, colour=GREEN, width=12).pack(side="right", padx=4)
        _button(bar, "■  Stop", self._stop, colour=RED, width=8).pack(side="right", padx=4)

    def _build_basic(self, parent):
        f = tk.Frame(parent, bg=BG, padx=20, pady=14)
        f.pack(fill="both", expand=True)

        _section(f, "Timing").grid(row=0, column=0, columnspan=2,
                                   sticky="ew", pady=(0, 4))

        fields = [
            ("DURATION",       "Run duration per model (seconds)"),
            ("IDLE_DURATION",  "Idle baseline duration (seconds)"),
            ("PROMPT_TIMEOUT", "Per-prompt timeout (seconds)"),
            ("INTERVAL",       "Sampling interval (seconds)"),
            ("MIN_FREE_MB",    "Minimum free disk space (MB)"),
        ]

        self._basic_vars = {}
        for i, (key, label) in enumerate(fields, start=1):
            _label(f, label).grid(row=i, column=0, sticky="w", pady=3, padx=(8, 16))
            e = _entry(f, width=12)
            e.grid(row=i, column=1, sticky="w", pady=3)
            self._basic_vars[key] = e

        row = len(fields) + 1
        _hline(f).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        _section(f, "Warmup").grid(row=row + 1, column=0, columnspan=2,
                                   sticky="ew", pady=(0, 4))

        _label(f, "Warmup prompt text").grid(row=row + 2, column=0,
                                             sticky="w", pady=3, padx=(8, 16))
        self._warmup_var = _entry(f, width=40)
        self._warmup_var.grid(row=row + 2, column=1, sticky="ew", pady=3)

        _hline(f).grid(row=row + 3, column=0, columnspan=2, sticky="ew", pady=10)
        _section(f, "Run mode").grid(row=row + 4, column=0, columnspan=2,
                                     sticky="ew", pady=(0, 4))

        _label(f, "Run mode").grid(row=row + 5, column=0,
                                   sticky="w", pady=3, padx=(8, 16))
        self._run_mode_var = tk.StringVar(value="duration")
        mode_frame = tk.Frame(f, bg=BG2)
        mode_frame.grid(row=row + 5, column=1, sticky="w", pady=3)
        for val, label in [("duration", "Duration  (cycle prompts until time runs out)"),
                           ("all_once", "All once  (every prompt runs exactly once)")]:
            tk.Radiobutton(
                mode_frame, text=label, variable=self._run_mode_var, value=val,
                bg=BG2, fg=FG, selectcolor=BG, activebackground=BG2,
                activeforeground=ACCENT, font=MONO,
            ).pack(anchor="w")


        _hline(f).grid(row=row + 6, column=0, columnspan=2, sticky='ew', pady=10)
        _section(f, 'Repetitions').grid(row=row + 7, column=0, columnspan=2,
                                        sticky='ew', pady=(0, 4))
        _label(f, 'Number of runs (N_RUNS)').grid(row=row + 8, column=0,
                                                   sticky='w', pady=3, padx=(8, 16))
        self._n_runs_var = _entry(f, width=6)
        self._n_runs_var.grid(row=row + 8, column=1, sticky='w', pady=3)
        _label(f, 'Per-run plots after each run. Aggregate plots at the end.',
               small=True).grid(row=row + 9, column=0, columnspan=2,
                                sticky='w', padx=(8, 0))

        f.columnconfigure(1, weight=1)

    def _build_advanced(self, parent):
        f = tk.Frame(parent, bg=BG, padx=20, pady=14)
        f.pack(fill="both", expand=True)

        _section(f, "Paths").grid(row=0, column=0, columnspan=2,
                                  sticky="ew", pady=(0, 4))

        path_fields = [
            ("BENCH_SCRIPT", "Benchmark script filename"),
            ("BASE_LOGDIR",  "Output directory (BASE_LOGDIR)"),
            ("LLAMA_BIN",    "llama-cli binary path (LLAMA_BIN)"),
            ("PROMPT_DIR",   "Prompts directory (PROMPT_DIR)"),
        ]
        self._path_vars = {}
        for i, (key, label) in enumerate(path_fields, start=1):
            _label(f, label).grid(row=i, column=0, sticky="w", pady=3, padx=(8, 16))
            e = _entry(f, width=50)
            e.grid(row=i, column=1, sticky="ew", pady=3)
            self._path_vars[key] = e

        row = len(path_fields) + 1
        _hline(f).grid(row=row, column=0, columnspan=2, sticky="ew", pady=10)
        _section(f, "Models").grid(row=row + 1, column=0, columnspan=2,
                                   sticky="ew", pady=(0, 4))

        _label(f, "One .gguf path per line.", small=True).grid(
            row=row + 2, column=0, columnspan=2, sticky="w", padx=(8, 0))

        self._models_text = scrolledtext.ScrolledText(
            f, height=8, bg=BG2, fg=FG,
            insertbackground=FG, relief="flat",
            highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT, font=MONO,
            wrap="none",
        )
        self._models_text.grid(row=row + 3, column=0, columnspan=2,
                               sticky="nsew", pady=4, padx=(8, 0))

        btn_row = tk.Frame(f, bg=BG)
        btn_row.grid(row=row + 4, column=0, columnspan=2,
                     sticky="w", padx=(8, 0), pady=(0, 4))
        _button(btn_row, "+ Add row", self._add_model_row,
                colour=ACCENT, width=10).pack(side="left", padx=(0, 6))
        _button(btn_row, " Clear all", self._clear_models,
                colour=RED, width=10).pack(side="left")

        f.columnconfigure(1, weight=1)
        f.rowconfigure(row + 3, weight=1)

    def _build_run(self, parent):
        f = tk.Frame(parent, bg=BG, padx=20, pady=14)
        f.pack(fill="both", expand=True)

        _section(f, "Live output").pack(fill="x", pady=(0, 6))

        self._run_log = scrolledtext.ScrolledText(
            f, bg=BG2, fg=FG,
            insertbackground=FG, relief="flat",
            highlightthickness=1, highlightbackground=BORDER,
            font=(MONO[0], 9), state="disabled", wrap="word",
        )
        self._run_log.pack(fill="both", expand=True)

        _label(f, "Launch or Stop the run using the buttons at the bottom.",
               small=True).pack(anchor="w", pady=(6, 0))

    # ── Install tab ──────────────────────────────────────────────────────────

    def _build_install(self, parent):
        f = tk.Frame(parent, bg=BG, padx=14, pady=10)
        f.pack(fill="both", expand=True)

        # ── System info bar ───────────────────────────────────────────────
        _section(f, "System").pack(fill="x", pady=(0, 4))
        info_row = tk.Frame(f, bg=BG2, pady=6, padx=10)
        info_row.pack(fill="x")

        self._lbl_vram = tk.Label(info_row, text="VRAM: -", bg=BG2, fg=FG,
                                  font=MONO, padx=12)
        self._lbl_vram.pack(side="left")
        self._lbl_ram = tk.Label(info_row, text="RAM: -", bg=BG2, fg=FG,
                                 font=MONO, padx=12)
        self._lbl_ram.pack(side="left")
        self._lbl_disk = tk.Label(info_row, text="Disk: -", bg=BG2, fg=FG,
                                  font=MONO, padx=12)
        self._lbl_disk.pack(side="left")
        _button(info_row, "⟳ Refresh", self._install_refresh,
                colour=ACCENT, width=10).pack(side="right", padx=4)

        # ── Model table ───────────────────────────────────────────────────
        _section(f, "Available models").pack(fill="x", pady=(8, 4))

        tree_frame = tk.Frame(f, bg=BG)
        tree_frame.pack(fill="x")

        cols = ("name", "params", "est_vram", "fit", "installed", "notes")
        self._install_tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings",
            height=len(INSTALLABLE_MODELS), selectmode="extended",
        )
        style = ttk.Style()
        style.configure("Treeview",
                        background=BG2, foreground=FG,
                        fieldbackground=BG2, rowheight=22,
                        font=MONO)
        style.configure("Treeview.Heading",
                        background=BG, foreground=ACCENT, font=MONO)
        style.map("Treeview", background=[("selected", "#1f2937")])

        col_cfg = [
            ("name",      "Model",       200, "w"),
            ("params",    "Params",       60, "center"),
            ("est_vram",  "Est. VRAM",    80, "center"),
            ("fit",       "Fits?",        60, "center"),
            ("installed", "Installed",    80, "center"),
            ("notes",     "Notes",       220, "w"),
        ]
        for cid, heading, width, anchor in col_cfg:
            self._install_tree.heading(cid, text=heading)
            self._install_tree.column(cid, width=width, anchor=anchor,
                                      stretch=(cid == "notes"))

        self._install_tree.pack(fill="x", side="left")

        sb = ttk.Scrollbar(tree_frame, orient="vertical",
                           command=self._install_tree.yview)
        self._install_tree.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")

        # ── Action buttons ────────────────────────────────────────────────
        btn_row = tk.Frame(f, bg=BG, pady=6)
        btn_row.pack(fill="x")
        _button(btn_row, "Install deps", self._install_deps,
                colour=ACCENT, width=14).pack(side="left", padx=(0, 6))
        _button(btn_row, "Install selected", self._install_selected,
                colour=GREEN, width=16).pack(side="left", padx=(0, 6))
        _button(btn_row, "Install all that fit", self._install_all_fit,
                colour=GREEN, width=18).pack(side="left", padx=(0, 6))
        _button(btn_row, "■  Stop", self._install_stop,
                colour=RED, width=8).pack(side="right")

        # ── Output log ────────────────────────────────────────────────────
        _section(f, "Output").pack(fill="x", pady=(6, 4))
        self._install_log = scrolledtext.ScrolledText(
            f, height=10, bg=BG2, fg=FG,
            insertbackground=FG, relief="flat",
            highlightthickness=1, highlightbackground=BORDER,
            font=(MONO[0], 9), state="disabled", wrap="word",
        )
        self._install_log.pack(fill="both", expand=True)

        # Populate with placeholder rows; refresh will fill real data
        for m in INSTALLABLE_MODELS:
            self._install_tree.insert("", "end", iid=m["id"], values=(
                m["name"],
                f"{m['params_b']}B",
                f"{_estimate_vram_gb(m):.1f} GB",
                "-", "-", m["notes"],
            ))

        # Trigger first refresh after window is drawn
        self.root.after(500, self._install_refresh)

    def _install_refresh(self):
        """Re-detect system specs and model install status."""
        vram_mb  = _detect_vram_mb()
        ram_mb   = _detect_ram_mb()
        cfg      = self._collect()
        base_dir = Path(cfg.get("BASE_LOGDIR", str(Path.home() / "models"))).parent / "models"
        # Try to resolve a sensible models base dir
        models_base = Path.home() / "models"
        disk_gb  = _detect_disk_gb(str(models_base))

        self._lbl_vram.config(
            text=f"VRAM: {vram_mb} MB ({vram_mb/1024:.1f} GB)"
            if vram_mb else "VRAM: not detected")
        self._lbl_ram.config(text=f"RAM: {ram_mb} MB ({ram_mb/1024:.1f} GB)")
        self._lbl_disk.config(text=f"Disk free: {disk_gb} GB ({models_base})")

        for m in INSTALLABLE_MODELS:
            est_gb = _estimate_vram_gb(m)
            fits   = vram_mb > 0 and (est_gb * 1024 * 0.9 <= vram_mb)
            fit_str = " Yes" if fits else (" No" if vram_mb > 0 else "?")

            gguf_path = models_base / m["gguf_rel"]
            inst_str = " Yes" if gguf_path.exists() else "No"

            self._install_tree.item(m["id"], values=(
                m["name"],
                f"{m['params_b']}B",
                f"{est_gb:.1f} GB",
                fit_str,
                inst_str,
                m["notes"],
            ))
            # Colour rows
            tag = "fits" if fits else "nofits"
            self._install_tree.item(m["id"], tags=(tag,))

        self._install_tree.tag_configure("fits",   foreground=FG)
        self._install_tree.tag_configure("nofits", foreground=FG_DIM)

    def _install_run(self, extra_args: list[str]):
        """Launch install_models.sh with given args, output to install log."""
        if self._install_proc and self._install_proc.poll() is None:
            messagebox.showwarning("Already running",
                                   "An installation is already in progress.")
            return
        if not INSTALL_SCRIPT.exists():
            messagebox.showerror("Not found",
                                 f"install_models.sh not found at:\n{INSTALL_SCRIPT}")
            return

        self._install_log_clear()
        self._install_log_append(
            f"Running: bash {INSTALL_SCRIPT.name} {' '.join(extra_args)}\n\n")

        try:
            self._install_proc = subprocess.Popen(
                ["bash", str(INSTALL_SCRIPT)] + extra_args,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, preexec_fn=os.setpgrp,
            )
        except Exception as exc:
            messagebox.showerror("Launch failed", str(exc))
            return

        self._install_queue = queue.Queue()
        threading.Thread(target=self._install_read_output, daemon=True).start()
        self.root.after(100, self._install_poll)

    def _install_deps(self):
        self._install_run(["--deps-only"])

    def _install_selected(self):
        sel = self._install_tree.selection()
        if not sel:
            messagebox.showinfo("Nothing selected",
                                "Select one or more models in the table first.")
            return
        self._install_run(list(sel))

    def _install_all_fit(self):
        self._install_run([])

    def _install_stop(self):
        if not (self._install_proc and self._install_proc.poll() is None):
            return
        try:
            pgid = os.getpgid(self._install_proc.pid)
            os.killpg(pgid, __import__("signal").SIGINT)
        except ProcessLookupError:
            pass
        self._install_log_append("\n[GUI] Install stopped.\n")

    def _install_read_output(self):
        try:
            for line in self._install_proc.stdout:
                self._install_queue.put(line)
        finally:
            self._install_queue.put(None)

    def _install_poll(self):
        done = False
        for _ in range(50):
            try:
                line = self._install_queue.get_nowait()
            except queue.Empty:
                break
            if line is None:
                done = True
                break
            self._install_log_append(line)
        if done:
            self._install_proc.wait()
            self._install_log_append("\n[GUI] Installation finished.\n")
            # Auto-refresh model status and config
            self._install_refresh()
            self.cfg = _parse_config(CONFIG_FILE)
            self._populate()
        else:
            self.root.after(100, self._install_poll)

    def _install_log_append(self, text: str):
        self._install_log.config(state="normal")
        self._install_log.insert("end", text)
        self._install_log.see("end")
        self._install_log.config(state="disabled")

    def _install_log_clear(self):
        self._install_log.config(state="normal")
        self._install_log.delete("1.0", "end")
        self._install_log.config(state="disabled")

    # ── Populate from config ──────────────────────────────────────────────────

    def _populate(self):
        for key, widget in self._basic_vars.items():
            widget.delete(0, "end")
            widget.insert(0, self.cfg.get(key, ""))

        self._warmup_var.delete(0, "end")
        self._warmup_var.insert(0, self.cfg.get("WARMUP_PROMPT", ""))

        self._run_mode_var.set(self.cfg.get("RUN_MODE", "duration"))

        self._n_runs_var.delete(0, "end")
        self._n_runs_var.insert(0, self.cfg.get("N_RUNS", "3"))

        for key, widget in self._path_vars.items():
            widget.delete(0, "end")
            widget.insert(0, self.cfg.get(key, ""))

        self._models_text.config(state="normal")
        self._models_text.delete("1.0", "end")
        self._models_text.insert("end", "\n".join(self.cfg.get("MODELS", [])))
        self._models_text.config(state="normal")

    # ── Collect values from widgets ───────────────────────────────────────────

    def _collect(self) -> dict:
        cfg = {}
        for key, widget in self._basic_vars.items():
            cfg[key] = widget.get().strip()
        cfg["WARMUP_PROMPT"] = self._warmup_var.get().strip()
        cfg["RUN_MODE"]       = self._run_mode_var.get()
        cfg["N_RUNS"]         = self._n_runs_var.get().strip()
        for key, widget in self._path_vars.items():
            cfg[key] = widget.get().strip()
        raw_models = self._models_text.get("1.0", "end").strip()
        cfg["MODELS"] = [
            line.strip() for line in raw_models.splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        return cfg

    def _validate(self, cfg: dict) -> list[str]:
        errors = []
        for key in ["DURATION", "IDLE_DURATION", "PROMPT_TIMEOUT",
                    "INTERVAL", "MIN_FREE_MB"]:
            v = cfg.get(key, "")
            try:
                if int(v) <= 0:
                    raise ValueError
            except (ValueError, TypeError):
                errors.append(f"{key} must be a positive integer (got: '{v}')")
        try:
            if int(cfg.get("N_RUNS", "1")) < 1:
                raise ValueError
        except (ValueError, TypeError):
            errors.append(
                "N_RUNS must be a positive integer (got: '{}')".format(
                    cfg.get("N_RUNS", "")))
        if not cfg.get("BENCH_SCRIPT"):

            errors.append("Benchmark script filename cannot be empty.")
        if not cfg.get("LLAMA_BIN"):
            errors.append("LLAMA_BIN path cannot be empty.")
        if not cfg.get("BASE_LOGDIR"):
            errors.append("BASE_LOGDIR path cannot be empty.")
        if not cfg.get("MODELS"):
            errors.append("At least one model path is required.")
        return errors

    # ── Actions ───────────────────────────────────────────────────────────────

    def _save(self):
        cfg = self._collect()
        errors = self._validate(cfg)
        if errors:
            messagebox.showerror("Validation error", "\n".join(errors))
            return
        _write_config(CONFIG_FILE, cfg)
        self.cfg = cfg
        self._set_status(f"Config saved → {CONFIG_FILE.name}", colour=GREEN)

    def _add_model_row(self):
        self._models_text.config(state="normal")
        current = self._models_text.get("1.0", "end").rstrip("\n")
        if current:
            self._models_text.insert("end", "\n/path/to/model.gguf")
        else:
            self._models_text.insert("end", "/path/to/model.gguf")

    def _clear_models(self):
        if messagebox.askyesno("Clear models", "Remove all model paths?"):
            self._models_text.delete("1.0", "end")

    def _ask_sudo_password(self) -> "str | None":
        """Modal dialog to collect the sudo password.
        Returns the password string, or None if the user cancelled."""
        dialog = tk.Toplevel(self.root)
        dialog.title("sudo authentication")
        dialog.configure(bg=BG)
        dialog.resizable(False, False)
        dialog.grab_set()

        tk.Label(dialog,
                 text="  sudo password required to run powerstat:",
                 bg=BG, fg=FG_DIM, font=MONO, anchor="w",
                 pady=10, padx=12).pack(fill="x")

        pwd_var = tk.StringVar()
        entry = _entry(dialog, width=32, show="*", textvariable=pwd_var)
        entry.pack(padx=20, pady=(0, 10))
        entry.focus_set()

        result = {"value": None}

        def _ok(event=None):
            result["value"] = pwd_var.get()
            dialog.destroy()

        def _cancel(event=None):
            dialog.destroy()

        entry.bind("<Return>", _ok)
        entry.bind("<Escape>", _cancel)

        btn_row = tk.Frame(dialog, bg=BG)
        btn_row.pack(pady=(0, 12))
        _button(btn_row, "OK",     _ok,     colour=GREEN, width=8).pack(side="left", padx=6)
        _button(btn_row, "Cancel", _cancel, colour=RED,   width=8).pack(side="left", padx=6)

        self.root.wait_window(dialog)
        return result["value"]

    def _verify_sudo(self, password: str) -> bool:
        """Test the password with sudo -S true. Returns True if accepted."""
        try:
            result = subprocess.run(
                ["sudo", "-S", "-p", "", "true"],
                input=password + "\n",
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _rerun_analysis(self):
        """Open a folder picker, then run rerun_analysis.sh on the selected session."""
        from tkinter import filedialog

        if self._proc and self._proc.poll() is None:
            messagebox.showwarning("Busy", "A benchmark is running. Wait or stop it first.")
            return

        cfg = self._collect()
        initial_dir = cfg.get("BASE_LOGDIR", str(Path.home() / "benchmark_results"))

        session_dir = filedialog.askdirectory(
            title="Select session folder (the timestamp directory)",
            initialdir=initial_dir,
        )
        if not session_dir:
            return

        if not RERUN_SCRIPT.exists():
            messagebox.showerror("Not found",
                                 f"rerun_analysis.sh not found at:\n{RERUN_SCRIPT}")
            return

        # Switch to Run tab to show output
        for tab in self.root.winfo_children():
            if hasattr(tab, 'select'):
                tab.select(self._tab_run)
                break

        self._log_clear()
        self._log_append(f"Re-running analysis on: {session_dir}\n\n")
        self._set_status("Re-running analysis...", colour=GREEN)

        try:
            self._proc = subprocess.Popen(
                ["bash", str(RERUN_SCRIPT), session_dir],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, preexec_fn=os.setpgrp,
            )
        except Exception as exc:
            messagebox.showerror("Failed", str(exc))
            self._set_status("Rerun failed.", colour=RED)
            return

        self._output_queue = queue.Queue()
        threading.Thread(target=self._read_output, daemon=True).start()
        self.root.after(100, self._poll_output)

    def _launch(self):
        if self._proc and self._proc.poll() is None:
            messagebox.showwarning("Already running",
                                   "A benchmark is already running. Stop it first.")
            return

        # Auto-save before launching
        cfg = self._collect()
        errors = self._validate(cfg)
        if errors:
            messagebox.showerror("Validation error", "\n".join(errors))
            return
        _write_config(CONFIG_FILE, cfg)

        bench_file = SCRIPT_DIR / cfg.get("BENCH_SCRIPT", "run_energy_benchmark.sh")
        if not bench_file.exists():
            messagebox.showerror("Not found",
                                 f"Benchmark script not found at:\n{bench_file}\n\n"
                                 "Check the filename in the Advanced tab.")
            return

        # Always ask for the sudo password before launching.
        # We don't rely on a cached timestamp because the benchmark can run
        # for a long time (N_RUNS × models × prompts) and a stale timestamp
        # would cause powerstat to fail silently mid-run.
        # The keepalive loop inside the script then handles renewals.
        while True:
            password = self._ask_sudo_password()
            if password is None:
                self._set_status("Launch cancelled.", colour=FG_DIM)
                return
            if self._verify_sudo(password):
                # Set a fresh timestamp immediately before handing off to bash
                subprocess.run(
                    ["sudo", "-S", "-p", "", "-v"],
                    input=password + "\n",
                    capture_output=True, text=True,
                )
                break
            messagebox.showerror("Authentication failed",
                                 "Incorrect password, please try again.")

        # Switch to Run tab
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Notebook):
                widget.select(2)
                break

        self._log_clear()
        self._log_append(f"Launching: bash {bench_file}\n\n")
        self._set_status("Running…", colour=GREEN)

        try:
            self._proc = subprocess.Popen(
                ["bash", str(bench_file)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                # os.setpgrp creates a new process group (so SIGINT via
                # killpg only hits the benchmark, not the GUI) while staying
                # in the same session as the GUI - which is what allows the
                # sudo timestamp set by the GUI to be valid inside the script.
                # start_new_session=True would call setsid() and break the
                # sudo timestamp by moving into a new session.
                preexec_fn=os.setpgrp,
            )
        except Exception as exc:
            messagebox.showerror("Launch failed", str(exc))
            return

        # Read stdout in a background thread so the GUI never blocks
        self._output_queue = queue.Queue()
        t = threading.Thread(target=self._read_output, daemon=True)
        t.start()
        self.root.after(100, self._poll_output)


    def _read_output(self):
        """Background thread: reads lines from the process and queues them.
        Runs until the process exits and stdout is exhausted."""
        try:
            for line in self._proc.stdout:
                self._output_queue.put(line)
        finally:
            self._output_queue.put(None)  # sentinel: process is done


    def _stop(self):
        if not (self._proc and self._proc.poll() is None):
            self._set_status("No running process.", colour=FG_DIM)
            return

        try:
            pgid = os.getpgid(self._proc.pid)
        except ProcessLookupError:
            self._set_status("Process already gone.", colour=FG_DIM)
            return

        self._log_append("\n[GUI] Sending SIGINT to process group "
                         f"(pgid {pgid}) - same as Ctrl+C...\n")

        try:
            # SIGINT triggers the trap in benchmark.sh exactly like Ctrl+C,
            # so powerstat, temp logger, and llama-cli are all cleaned up
            # via the script's own trap handler.
            os.killpg(pgid, __import__("signal").SIGINT)
        except ProcessLookupError:
            pass

        # Give the trap handler up to 10 s to finish gracefully
        self.root.after(10_000, lambda: self._force_kill_if_running(pgid))
        self._set_status("Stopping… (waiting for cleanup)", colour=RED)

    def _force_kill_if_running(self, pgid: int):
        """Escalate to SIGKILL if the process group is still alive after
        the graceful shutdown window."""
        if self._proc and self._proc.poll() is None:
            self._log_append("[GUI] Process still running after 10s - "
                             "sending SIGKILL.\n")
            try:
                os.killpg(pgid, __import__("signal").SIGKILL)
            except ProcessLookupError:
                pass
            self._set_status("Killed.", colour=RED)

    def _poll_output(self):
        """Called by tkinter every 100 ms. Drains the output queue without
        blocking so the UI stays fully responsive during long runs."""
        if self._proc is None:
            return

        done = False
        # Drain up to 50 lines per tick to keep the UI snappy
        for _ in range(50):
            try:
                line = self._output_queue.get_nowait()
            except queue.Empty:
                break
            if line is None:  # sentinel from _read_output
                done = True
                break
            self._log_append(line)

        if done:
            rc = self._proc.wait()
            self._log_append(f"\n[GUI] Process exited (code {rc}).\n")
            colour = GREEN if rc == 0 else RED
            self._set_status(f"Finished (exit code {rc}).", colour=colour)
        else:
            self.root.after(100, self._poll_output)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log_append(self, text: str):
        self._run_log.config(state="normal")
        self._run_log.insert("end", text)
        self._run_log.see("end")
        self._run_log.config(state="disabled")

    def _log_clear(self):
        self._run_log.config(state="normal")
        self._run_log.delete("1.0", "end")
        self._run_log.config(state="disabled")

    def _set_status(self, text: str, colour: str = FG_DIM):
        self._status.config(text=text, fg=colour)

    def _select_tab(self, index: int):
        """Select notebook tab by index."""
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Notebook):
                widget.select(index)
                return


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.configure(bg=BG)

    # Try to set a dark window icon if available
    try:
        root.tk.call("wm", "iconphoto", root._w,
                     tk.PhotoImage(data=""))
    except Exception:
        pass

    app = BenchmarkGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
