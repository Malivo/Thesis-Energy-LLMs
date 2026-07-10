#!/bin/bash
# ============================================================
# setup.sh -- One-shot setup for the LLM Energy Benchmark
#
# Run this once on a new machine. It:
#   1. Installs system dependencies
#   2. Clones and builds llama.cpp with CUDA support
#   3. Sets up the Python conversion venv
#   4. Installs HuggingFace CLI
#   5. Writes machine-specific paths to benchmark_config.sh
#   6. Verifies the setup
#
# Usage:
#   chmod +x setup.sh && ./setup.sh
#
# After this, run install_models.sh to download and install models.
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/benchmark_config.sh"
LOG_FILE="$SCRIPT_DIR/setup.log"
LLAMA_DIR="$HOME/llama.cpp"
MODELS_BASE_DIR="$HOME/models"

log()     { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
ok()      { echo "[$(date '+%H:%M:%S')] [OK] $*" | tee -a "$LOG_FILE"; }
fail()    { echo "[$(date '+%H:%M:%S')] [FAIL] $*" | tee -a "$LOG_FILE"; exit 1; }
section() {
  echo "" | tee -a "$LOG_FILE"
  echo "=============================================" | tee -a "$LOG_FILE"
  echo "  $*" | tee -a "$LOG_FILE"
  echo "=============================================" | tee -a "$LOG_FILE"
}

: > "$LOG_FILE"
log "Setup started on $(hostname)"
log "Home: $HOME"
log "Script dir: $SCRIPT_DIR"

# ── 1. System dependencies ────────────────────────────────────────────────
section "System dependencies"

sudo apt-get update -qq
sudo apt-get install -y \
  build-essential cmake git wget curl unzip \
  python3 python3-pip python3-venv pipx \
  libcurl4-openssl-dev \
  lm-sensors powerstat \
  2>&1 | tee -a "$LOG_FILE"

# CUDA toolkit - install only if nvidia-smi is present
if command -v nvidia-smi &>/dev/null; then
  if ! command -v nvcc &>/dev/null; then
    log "NVIDIA GPU detected, no CUDA toolkit found - installing..."
    sudo apt-get install -y nvidia-cuda-toolkit 2>&1 | tee -a "$LOG_FILE" || \
      log "WARN: CUDA toolkit install failed - will build CPU-only llama.cpp"
  else
    log "NVIDIA GPU detected, CUDA toolkit already installed: $(nvcc --version 2>&1 | grep release)"
  fi
  CUDA_AVAILABLE=1
else
  log "No NVIDIA GPU detected - will build CPU-only llama.cpp"
  CUDA_AVAILABLE=0
fi

ok "System dependencies installed"

# ── 2. llama.cpp ─────────────────────────────────────────────────────────
section "llama.cpp"

if [ ! -d "$LLAMA_DIR" ]; then
  log "Cloning llama.cpp..."
  git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR" 2>&1 | tee -a "$LOG_FILE"
else
  log "llama.cpp directory exists - pulling latest..."
  git -C "$LLAMA_DIR" pull --ff-only 2>&1 | tee -a "$LOG_FILE" || \
    log "WARN: git pull failed - using existing version"
fi

mkdir -p "$LLAMA_DIR/build"
cd "$LLAMA_DIR/build"

if [ "$CUDA_AVAILABLE" -eq 1 ]; then
  log "Configuring with CUDA support..."
  cmake .. -DGGML_CUDA=ON 2>&1 | tee -a "$LOG_FILE"
else
  log "Configuring CPU-only build..."
  cmake .. 2>&1 | tee -a "$LOG_FILE"
fi

log "Building (this may take several minutes)..."
cmake --build . --config Release -j "$(nproc)" 2>&1 | tee -a "$LOG_FILE"

LLAMA_COMPLETION="$LLAMA_DIR/build/bin/llama-completion"
LLAMA_BIN="$LLAMA_DIR/build/bin/llama-cli"

if [ ! -f "$LLAMA_COMPLETION" ]; then
  fail "llama-completion not found after build. Check $LOG_FILE for errors."
fi
ok "llama.cpp built: $LLAMA_COMPLETION"

cd "$SCRIPT_DIR"

# ── 3. Python conversion venv ─────────────────────────────────────────────
section "Python conversion environment"

CONV_VENV="$LLAMA_DIR/venv"
if [ ! -d "$CONV_VENV" ]; then
  python3 -m venv "$CONV_VENV"
fi
"$CONV_VENV/bin/pip" install --quiet --upgrade pip
"$CONV_VENV/bin/pip" install --quiet \
  torch transformers accelerate sentencepiece safetensors huggingface_hub
ok "Conversion venv ready: $CONV_VENV"

# ── 4. HuggingFace CLI ────────────────────────────────────────────────────
section "HuggingFace CLI"

# Ensure ~/.local/bin is on PATH (pipx installs binaries there)
export PATH="$HOME/.local/bin:$PATH"

if ! command -v huggingface-cli &>/dev/null; then
  pipx install huggingface_hub
  pipx ensurepath
  export PATH="$HOME/.local/bin:$PATH"
  log "HuggingFace CLI installed via pipx"
  log "NOTE: Run 'source ~/.bashrc' or open a new terminal for PATH to persist."
else
  log "HuggingFace CLI already installed"
fi

if huggingface-cli whoami &>/dev/null; then
  ok "HuggingFace: logged in as $(huggingface-cli whoami 2>/dev/null)"
else
  log ""
  log "NOTE: Not logged into HuggingFace."
  log "Run 'huggingface-cli login' before running install_models.sh."
fi

# ── 5. Write machine-specific config ──────────────────────────────────────
section "Writing benchmark_config.sh"

# Detect GPU VRAM
VRAM_MB=0
if command -v nvidia-smi &>/dev/null; then
  VRAM_MB=$(nvidia-smi --query-gpu=memory.total \
    --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
fi
log "Detected VRAM: ${VRAM_MB} MB"

# Write a fresh config with machine-specific paths
cat > "$CONFIG_FILE" << CFGEOF
# ============================================
# Benchmark Configuration
# Generated by setup.sh on $(hostname) - $(date)
# Edit here or use the GUI to update values.
# ============================================

# ----- Machine info --------------------------
MACHINE_HOSTNAME="$(hostname)"
MACHINE_VRAM_MB="$VRAM_MB"

# ----- Timing --------------------------------
DURATION=3600
INTERVAL=1
IDLE_DURATION=30
PROMPT_TIMEOUT=20
# RUN_MODE: "duration" = cycle prompts until DURATION expires
#           "all_once" = run every prompt exactly once per model
RUN_MODE=all_once
N_RUNS=3

# ----- Disk space guard ----------------------
MIN_FREE_MB=2000

# ----- Paths ---------------------------------
MODELS_BASE_DIR="\$HOME/models"
LLAMA_DIR="\$HOME/llama.cpp"
BASE_LOGDIR="\$HOME/benchmark_results"
LLAMA_BIN="\$HOME/llama.cpp/build/bin/llama-cli"
BENCH_SCRIPT="benchmark.sh"
PROMPT_DIR="\$HOME/prompts"

# ----- Warmup --------------------------------
WARMUP_PROMPT="Say hello."

# ----- Difficulty classification -------------
DIFFICULTY_BLOCK_SIZE=30

# ----- Models --------------------------------
# Populated by install_models.sh after installation.
# One full path per element.
MODELS=(
)
CFGEOF

ok "Config written: $CONFIG_FILE"

# ── 6. Verify setup ───────────────────────────────────────────────────────
section "Verification"

PASS=0
FAIL=0

check() {
  local desc="$1" cmd="$2"
  if eval "$cmd" &>/dev/null; then
    ok "$desc"
    PASS=$((PASS+1))
  else
    log "[FAIL] $desc"
    FAIL=$((FAIL+1))
  fi
}

check "llama-completion binary"  "[ -f '$LLAMA_COMPLETION' ]"
check "llama-quantize binary"    "[ -f '$LLAMA_DIR/build/bin/llama-quantize' ]"
check "powerstat available"      "command -v powerstat"
check "nvidia-smi available"     "command -v nvidia-smi"
check "python3 available"        "command -v python3"
check "requirements.txt exists"  "[ -f '$SCRIPT_DIR/requirements.txt' ]"
check "analyze_results.py exists" "[ -f '$SCRIPT_DIR/analyze_results.py' ]"
check "aggregate_analysis.py exists" "[ -f '$SCRIPT_DIR/aggregate_analysis.py' ]"

echo ""
log "Setup complete. Passed: $PASS  Failed: $FAIL"
if [ "$FAIL" -gt 0 ]; then
  log "Some checks failed. Review $LOG_FILE for details."
else
  ok "Machine is ready."
  log ""
  log "Next steps:"
  log "  1. huggingface-cli login        (if not already logged in)"
  log "  2. ./install_models.sh          (download and install models)"
  log "  3. ./benchmark_gui.py           (launch the GUI) or"
  log "     ./benchmark.sh               (run directly from terminal)"
fi
