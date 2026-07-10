#!/bin/bash
# ============================================================
# install_models.sh — LLM Model Installer
# Downloads, converts and quantizes models to Q4_K_M.
# Checks VRAM compatibility before downloading.
# Auto-updates benchmark_config.sh with compatible models.
#
# Usage:
#   ./install_models.sh                        # install all that fit
#   ./install_models.sh --deps-only            # only install system deps + build llama.cpp
#   ./install_models.sh --check-only           # print what fits, do nothing
#   ./install_models.sh gemma3_4b mistral_7b   # install specific models by ID
# ============================================================

set -u
set +m

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/benchmark_config.sh"
LOG_FILE="$SCRIPT_DIR/install_models.log"

# ── Load config for paths ──────────────────────────────────────────────────
if [ -f "$CONFIG_FILE" ]; then
  source "$CONFIG_FILE"
fi
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
MODELS_BASE_DIR="${MODELS_BASE_DIR:-$HOME/models}"
LLAMA_BIN="${LLAMA_BIN:-$LLAMA_DIR/build/bin/llama-cli}"
LLAMA_COMPLETION="$(dirname "$LLAMA_BIN")/llama-completion"
LLAMA_QUANTIZE="$(dirname "$LLAMA_BIN")/llama-quantize"

# ── Logging ────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }
log_section() {
  echo "" | tee -a "$LOG_FILE"
  echo "=============================================" | tee -a "$LOG_FILE"
  echo "  $*" | tee -a "$LOG_FILE"
  echo "=============================================" | tee -a "$LOG_FILE"
}

# ── Model catalogue ────────────────────────────────────────────────────────
# Fields: id|display_name|hf_repo|local_subdir|gguf_basename|params_B|notes
#
# local_subdir  — relative to MODELS_BASE_DIR, used for download + source of conversion
# gguf_basename — relative to MODELS_BASE_DIR, without .gguf / .Q4_K_M.gguf suffix
#
# VRAM estimate (Q4_K_M): params_B × 4.5 / 8 + 1.5 GB overhead
# ──────────────────────────────────────────────────────────────────────────
declare -A MODEL_NAME MODEL_REPO MODEL_SUBDIR MODEL_GGUF MODEL_PARAMS MODEL_NOTES MODEL_DIRECT_GGUF MODEL_EXTRA_FLAGS

_register() {
  # Args: id name hf_repo local_subdir gguf_basename params notes [direct_gguf=0] [extra_flags=""]
  local id="$1" name="$2" repo="$3" subdir="$4" gguf="$5" params="$6" notes="$7"
  local direct="${8:-0}" flags="${9:-}"
  MODEL_NAME["$id"]="$name"
  MODEL_REPO["$id"]="$repo"
  MODEL_SUBDIR["$id"]="$subdir"
  MODEL_GGUF["$id"]="$gguf"
  MODEL_PARAMS["$id"]="$params"
  MODEL_NOTES["$id"]="$notes"
  MODEL_DIRECT_GGUF["$id"]="$direct"  # 1 = download pre-built GGUF, skip convert/quantize
  MODEL_EXTRA_FLAGS["$id"]="$flags"   # extra llama-cli flags for testing/running
}

_register gemma3_4b \
  "Gemma 3 4B (IT)" \
  "google/gemma-3-4b-it" \
  "gemma/gemma-3-4b" \
  "gemma/gemma3_4b" \
  4 \
  "Requires HuggingFace access approval at huggingface.co/google/gemma-3-4b-it"

_register qwen2_5_7b \
  "Qwen 2.5 7B" \
  "Qwen/Qwen2.5-7B" \
  "qwen/Qwen2.5-7B" \
  "qwen/qwen2_5_7b" \
  7 \
  ""

_register qwen2_5_coder_7b \
  "Qwen 2.5 Coder 7B" \
  "Qwen/Qwen2.5-Coder-7B" \
  "qwen/Qwen2.5-Coder-7B" \
  "qwen/qwen2_5_coder_7b" \
  7 \
  ""

_register mistral_7b \
  "Mistral 7B Instruct v0.2" \
  "mistralai/Mistral-7B-Instruct-v0.2" \
  "mistral/Mistral-7B-Instruct" \
  "mistral/mistral_7b" \
  7 \
  ""

_register qwen3_32b \
  "Qwen 3 32B" \
  "Qwen/Qwen3-32B" \
  "qwen3/Qwen3-32B" \
  "qwen3/qwen3_32b" \
  32 \
  "Requires ~19 GB VRAM"

_register qwen2_5_coder_32b \
  "Qwen 2.5 Coder 32B Instruct" \
  "Qwen/Qwen2.5-Coder-32B-Instruct" \
  "qwen/Qwen2.5-Coder-32B-Instruct" \
  "qwen/qwen2_5_coder_32b" \
  32 \
  "Requires ~19 GB VRAM. Pairs with Qwen2.5-Coder-7B for scale comparison."

# gpt-oss models use MXFP4 quantization natively. Their FFN weights do not
# quantize nicely to Q4_K_M, so we download pre-built GGUFs from ggml-org
# instead of using the convert→quantize pipeline.
# They also require --jinja when running llama-cli (harmony format).
# The GGUF files are named *-mxfp4.gguf after download.
_register gpt_oss_20b \
  "GPT OSS 20B (MoE)" \
  "ggml-org/gpt-oss-20b-GGUF" \
  "gpt-oss-20b/gguf" \
  "gpt-oss-20b/gpt-oss-20b-mxfp4" \
  21 \
  "MXFP4 native quant. Needs --jinja flag. ~16 GB VRAM. GGUF direct download." 1 "--jinja"

_register gpt_oss_120b \
  "GPT OSS 120B (MoE)" \
  "ggml-org/gpt-oss-120b-GGUF" \
  "gpt-oss-120b/gguf" \
  "gpt-oss-120b/gpt-oss-120b-mxfp4" \
  117 \
  "MXFP4 native quant. Needs --jinja flag. ~80 GB VRAM (H100-class). GGUF direct download." 1 "--jinja"

# Ordered list for display
ALL_MODEL_IDS=(
  gemma3_4b
  qwen2_5_7b
  qwen2_5_coder_7b
  mistral_7b
  qwen3_32b
  qwen2_5_coder_32b
  gpt_oss_20b
  gpt_oss_120b
)

# ── System detection ───────────────────────────────────────────────────────
detect_vram_mb() {
  nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits \
    2>/dev/null | head -1 | tr -d ' '
}

detect_ram_mb() {
  awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo
}

detect_free_disk_gb() {
  local path="${1:-$MODELS_BASE_DIR}"
  mkdir -p "$path" 2>/dev/null
  df -BG "$path" | awk 'NR==2 {gsub("G","",$4); print $4}'
}

# Estimate VRAM required for Q4_K_M in MB
# Formula: params_B × 4.5 / 8 × 1024 + 1536 MB overhead
# For MoE models (gpt-oss) the native MXFP4 quant is more compact — use
# the empirically known figures instead of the formula.
declare -A MODEL_VRAM_OVERRIDE
MODEL_VRAM_OVERRIDE["gpt_oss_20b"]=$((16 * 1024))   # 16 GB
MODEL_VRAM_OVERRIDE["gpt_oss_120b"]=$((80 * 1024))  # 80 GB

estimate_vram_mb() {
  local id="$1" params_b="$2"
  if [[ -v MODEL_VRAM_OVERRIDE["$id"] ]]; then
    echo "${MODEL_VRAM_OVERRIDE[$id]}"
    return
  fi
  awk -v p="$params_b" 'BEGIN{printf "%d", p * 4.5 / 8 * 1024 + 1536}'
}

# ── Compatibility check ────────────────────────────────────────────────────
model_fits_estimate() {
  local id="$1" vram_mb="$2"
  local required_mb; required_mb=$(estimate_vram_mb "$id" "${MODEL_PARAMS[$id]}")
  # Leave 10% headroom
  local usable_mb; usable_mb=$(awk -v v="$vram_mb" 'BEGIN{printf "%d", v * 0.90}')
  [ "$required_mb" -le "$usable_mb" ]
}

model_installed() {
  local id="$1"
  local q4_path="$MODELS_BASE_DIR/${MODEL_GGUF[$id]}.Q4_K_M.gguf"
  [ -f "$q4_path" ]
}

model_in_config() {
  local id="$1"
  local q4_path="$MODELS_BASE_DIR/${MODEL_GGUF[$id]}.Q4_K_M.gguf"
  grep -qF "$q4_path" "$CONFIG_FILE" 2>/dev/null
}

# ── Dependency installation ────────────────────────────────────────────────
install_system_deps() {
  log_section "Installing system dependencies"
  sudo apt-get update -qq
  sudo apt-get install -y \
    build-essential cmake git wget unzip \
    python3 python3-pip python3-venv pipx \
    libcurl4-openssl-dev \
    lm-sensors powerstat

  # Only install nvidia-cuda-toolkit from apt if no CUDA toolkit is already
  # present. A newer version (e.g. 12.8 from NVIDIA's repo) should not be
  # overwritten by the older Ubuntu package (12.0).
  if ! command -v nvcc &>/dev/null; then
    log "No CUDA toolkit found, installing nvidia-cuda-toolkit from apt..."
    sudo apt-get install -y nvidia-cuda-toolkit
  else
    log "CUDA toolkit already installed: $(nvcc --version | grep release)"
  fi
  log "System dependencies installed."
}

build_llama_cpp() {
  log_section "Building llama.cpp"

  if [ ! -d "$LLAMA_DIR" ]; then
    log "Cloning llama.cpp..."
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMA_DIR"
  else
    log "llama.cpp directory exists — pulling latest..."
    git -C "$LLAMA_DIR" pull --ff-only || log "WARN: git pull failed, using existing version"
  fi

  mkdir -p "$LLAMA_DIR/build"
  cd "$LLAMA_DIR/build"

  log "Configuring with CUDA support..."
  cmake .. -DGGML_CUDA=ON

  log "Building (this may take several minutes)..."
  cmake --build . --config Release -j "$(nproc)"

  if [ -f "$LLAMA_DIR/build/bin/llama-cli" ]; then
    log "llama.cpp built successfully."
    log "  Binary: $LLAMA_DIR/build/bin/llama-cli"
  else
    log "ERROR: llama-cli binary not found after build."
    return 1
  fi

  cd "$SCRIPT_DIR"
}

setup_conversion_venv() {
  local venv_dir="$LLAMA_DIR/venv"
  if [ ! -d "$venv_dir" ]; then
    log "Creating Python venv for GGUF conversion..."
    python3 -m venv "$venv_dir"
  fi
  "$venv_dir/bin/pip" install --quiet --upgrade pip
  "$venv_dir/bin/pip" install --quiet \
    torch transformers accelerate sentencepiece safetensors huggingface_hub
  log "Conversion venv ready: $venv_dir"
}

ensure_hf_cli() {
  # Ensure ~/.local/bin is on PATH (pipx installs binaries there)
  export PATH="$HOME/.local/bin:$PATH"

  # Determine which CLI binary is available.
  # Prefer "hf" over "huggingface-cli" as the latter is deprecated
  # in newer versions and may print warnings or refuse to work.
  if command -v hf &>/dev/null; then
    HF_CLI="hf"
  elif command -v huggingface-cli &>/dev/null; then
    HF_CLI="huggingface-cli"
  else
    log "Installing huggingface_hub via pipx..."
    pipx install huggingface_hub
    pipx ensurepath
    export PATH="$HOME/.local/bin:$PATH"
    if command -v hf &>/dev/null; then
      HF_CLI="hf"
    elif command -v huggingface-cli &>/dev/null; then
      HF_CLI="huggingface-cli"
    else
      log "ERROR: Could not find huggingface-cli or hf after install."
      return 1
    fi
  fi
  log "Using HuggingFace CLI: $HF_CLI"

  # Check if logged in (try both CLI entry points)
  local hf_user=""
  if hf whoami &>/dev/null; then
    hf_user=$(hf whoami 2>/dev/null)
  elif huggingface-cli whoami &>/dev/null; then
    hf_user=$(huggingface-cli whoami 2>/dev/null)
  elif [ -f "$HOME/.cache/huggingface/token" ] || [ -f "$HOME/.huggingface/token" ]; then
    hf_user="(token found on disk)"
  fi

  if [ -z "$hf_user" ]; then
    log ""
    log "HuggingFace login required. Run one of:"
    log "  huggingface-cli login"
    log "  hf auth login"
    log "then re-run this script."
    return 1
  fi
  log "HuggingFace CLI: logged in as $hf_user"
}

# ── Per-model installation ─────────────────────────────────────────────────
install_model() {
  local id="$1"
  local name="${MODEL_NAME[$id]}"
  local repo="${MODEL_REPO[$id]}"
  local subdir="${MODEL_SUBDIR[$id]}"
  local gguf_base="${MODEL_GGUF[$id]}"
  local notes="${MODEL_NOTES[$id]}"
  local direct="${MODEL_DIRECT_GGUF[$id]:-0}"
  local extra_flags="${MODEL_EXTRA_FLAGS[$id]:-}"

  log_section "Installing: $name"
  [ -n "$notes" ] && log "NOTE: $notes"

  if [ "$direct" = "1" ]; then
    # ── Direct GGUF download path (gpt-oss and similar) ───────────────────
    # These models distribute pre-built, natively quantized GGUFs.
    # Converting and re-quantizing degrades quality, so we skip both steps.
    local gguf_dir="$MODELS_BASE_DIR/$subdir"
    local gguf_path="$MODELS_BASE_DIR/${gguf_base}.gguf"

    if [ -f "$gguf_path" ]; then
      log "Already installed: $gguf_path"
      _add_to_config "$id" "$gguf_path"
      return 0
    fi

    mkdir -p "$gguf_dir"
    log "Downloading pre-built GGUF from $repo..."
    if ! "$HF_CLI" download "$repo" \
          --include "*.gguf" \
          --local-dir "$gguf_dir"; then
      log "ERROR: GGUF download failed for $repo"
      return 1
    fi

    # Find the mxfp4 GGUF. For split models (multiple .gguf files),
    # llama.cpp must be pointed at the FIRST split (00001-of-NNNNN).
    local found_gguf
    found_gguf=$(find "$gguf_dir" -name "*mxfp4*00001*.gguf" | head -1)
    [ -z "$found_gguf" ] && found_gguf=$(find "$gguf_dir" -name "*mxfp4*.gguf" | sort | head -1)
    [ -z "$found_gguf" ] && found_gguf=$(find "$gguf_dir" -name "*.gguf" | sort | head -1)

    if [ -z "$found_gguf" ]; then
      log "ERROR: No GGUF file found in $gguf_dir"
      return 1
    fi

    # Symlink or copy to the expected path
    if [ "$found_gguf" != "$gguf_path" ]; then
      ln -sf "$found_gguf" "$gguf_path"
      log "Linked: $found_gguf → $gguf_path"
    fi

    # Test
    log "Testing model load..."
    # shellcheck disable=SC2086
    if timeout 120 "$LLAMA_COMPLETION" \
        -m "$gguf_path" \
        --n-gpu-layers 999 \
        --ctx-size 256 \
        --n-predict 1 \
        -p "Hello" \
        --single-turn \
        $extra_flags \
        >/dev/null 2>&1; then
      log "Test PASSED."
      _add_to_config "$id" "$gguf_path"
      log "✓ $name added to benchmark_config.sh"
    else
      log "Test FAILED — model may not fit in VRAM."
      log "  GGUF kept at: $gguf_path"
      log "  Extra flags needed at runtime: $extra_flags"
    fi
  else
    # ── Standard convert→quantize path ────────────────────────────────────
    local src_dir="$MODELS_BASE_DIR/$subdir"
    local gguf_path="$MODELS_BASE_DIR/${gguf_base}.gguf"
    local q4_path="$MODELS_BASE_DIR/${gguf_base}.Q4_K_M.gguf"

    if [ -f "$q4_path" ]; then
      log "Already installed: $q4_path"
      _add_to_config "$id" "$q4_path"
      return 0
    fi

    mkdir -p "$(dirname "$gguf_path")"

    # Step 1: Download
    if [ ! -d "$src_dir" ] || [ -z "$(ls -A "$src_dir" 2>/dev/null)" ]; then
      log "Downloading $repo..."
      mkdir -p "$src_dir"
      if ! "$HF_CLI" download "$repo" --local-dir "$src_dir"; then
        log "ERROR: Download failed for $repo"
        log "  Check the repo ID is correct and you have access."
        return 1
      fi
    else
      log "Source directory exists, skipping download: $src_dir"
    fi

    # Step 2: Convert to GGUF
    if [ ! -f "$gguf_path" ]; then
      log "Converting to GGUF: $gguf_path"
      if ! "$LLAMA_DIR/venv/bin/python" \
            "$LLAMA_DIR/convert_hf_to_gguf.py" \
            "$src_dir" \
            --outfile "$gguf_path"; then
        log "ERROR: GGUF conversion failed for $name"
        return 1
      fi
    else
      log "GGUF already exists, skipping conversion."
    fi

    # Step 3: Quantize to Q4_K_M
    log "Quantizing to Q4_K_M: $q4_path"
    if ! "$LLAMA_QUANTIZE" "$gguf_path" "$q4_path" Q4_K_M; then
      log "ERROR: Quantization failed for $name"
      return 1
    fi
    log "Quantization complete."

    # Step 4: Real VRAM test
    log "Testing model load..."
    # shellcheck disable=SC2086
    if timeout 120 "$LLAMA_COMPLETION" \
        -m "$q4_path" \
        --n-gpu-layers 999 \
        --ctx-size 256 \
        --n-predict 1 \
        -p "Hello" \
        --single-turn \
        $extra_flags \
        >/dev/null 2>&1; then
      log "Test PASSED — model loads on GPU."
      _add_to_config "$id" "$q4_path"
      log "✓ $name added to benchmark_config.sh"
    else
      log "Test FAILED — does not fit in VRAM with full GPU offload."
      log "  GGUF kept at: $q4_path"
    fi
  fi
}

# ── Config update ──────────────────────────────────────────────────────────
_write_paths_to_config() {
  # Write LLAMA_DIR and MODELS_BASE_DIR to config so benchmark.sh
  # can find llama-completion and models on any machine.
  [ -f "$CONFIG_FILE" ] || return
  for key_val in "LLAMA_DIR=$LLAMA_DIR" "MODELS_BASE_DIR=$MODELS_BASE_DIR"; do
    local key="${key_val%%=*}"
    local val="${key_val#*=}"
    if grep -q "^$key=" "$CONFIG_FILE"; then
      sed -i "s|^$key=.*|$key="$val"|" "$CONFIG_FILE"
    else
      echo "$key="$val"" >> "$CONFIG_FILE"
    fi
  done
  log "Config updated: LLAMA_DIR=$LLAMA_DIR, MODELS_BASE_DIR=$MODELS_BASE_DIR"
}

_add_to_config() {
  local id="$1"
  local q4_path="$2"

  if [ ! -f "$CONFIG_FILE" ]; then
    log "WARN: $CONFIG_FILE not found - cannot auto-update."
    return
  fi

  if grep -qF "$q4_path" "$CONFIG_FILE"; then
    log "Already in config: $q4_path"
    return
  fi

  # Use Python for reliable parsing — it's always available in our venv
  python3 - "$CONFIG_FILE" "$q4_path" << 'PYEOF'
import sys, re

config_file = sys.argv[1]
model_path  = sys.argv[2]

with open(config_file, 'r') as f:
    content = f.read()

if model_path in content:
    print(f"  Already in config: {model_path}")
    sys.exit(0)

# Find the closing ) of the MODELS=( ... ) block and insert before it
pattern = re.compile(r'(MODELS=\()(.*?)(\n\))', re.DOTALL)
m = pattern.search(content)
if m:
    new_entry = f'\n  "{model_path}"'
    new_content = (
        content[:m.start(3)]
        + new_entry
        + content[m.start(3):]
    )
    with open(config_file, 'w') as f:
        f.write(new_content)
    print(f"  Config updated: {model_path}")
else:
    print("  ERROR: Could not find MODELS array in config.")
    sys.exit(1)
PYEOF
}

# ── Check-only mode ────────────────────────────────────────────────────────
print_compatibility_report() {
  local vram_mb; vram_mb=$(detect_vram_mb)
  local ram_mb;  ram_mb=$(detect_ram_mb)
  local disk_gb; disk_gb=$(detect_free_disk_gb)

  echo ""
  echo "System:"
  printf "  GPU VRAM : %d MB (%.1f GB)\n" "$vram_mb" "$(awk -v v=$vram_mb 'BEGIN{printf "%.1f",v/1024}')"
  printf "  RAM      : %d MB (%.1f GB)\n" "$ram_mb"  "$(awk -v v=$ram_mb  'BEGIN{printf "%.1f",v/1024}')"
  printf "  Disk free: %d GB (%s)\n" "$disk_gb" "$MODELS_BASE_DIR"
  echo ""
  printf "  %-26s  %7s  %9s  %-10s  %s\n" "Model" "Params" "Est. VRAM" "Fit" "Installed"
  printf "  %-26s  %7s  %9s  %-10s  %s\n" "─────────────────────────" "───────" "─────────" "──────────" "─────────"

  for id in "${ALL_MODEL_IDS[@]}"; do
    local params="${MODEL_PARAMS[$id]}"
    local req_mb; req_mb=$(estimate_vram_mb "$id" "$params")
    local req_gb; req_gb=$(awk -v v=$req_mb 'BEGIN{printf "%.1f",v/1024}')
    local fits; fits="✗ No"
    model_fits_estimate "$id" "$vram_mb" && fits="✓ Yes"
    local installed; installed="No"
    model_installed "$id" && installed="Yes"
    printf "  %-26s  %5dB  %7.1f GB  %-10s  %s\n" \
      "${MODEL_NAME[$id]}" "$params" "$req_gb" "$fits" "$installed"
  done
  echo ""
}

# ── Argument parsing ───────────────────────────────────────────────────────
DEPS_ONLY=0
CHECK_ONLY=0
REQUESTED_IDS=()

for arg in "$@"; do
  case "$arg" in
    --deps-only)  DEPS_ONLY=1 ;;
    --check-only) CHECK_ONLY=1 ;;
    *)
      # Validate it's a known model ID
      if [[ -v MODEL_NAME["$arg"] ]]; then
        REQUESTED_IDS+=("$arg")
      else
        echo "Unknown model ID: $arg"
        echo "Known IDs: ${ALL_MODEL_IDS[*]}"
        exit 1
      fi
      ;;
  esac
done

# ── Main ───────────────────────────────────────────────────────────────────
: > "$LOG_FILE"
log "=== LLM Model Installer ==="
log "LLAMA_DIR       : $LLAMA_DIR"
log "MODELS_BASE_DIR : $MODELS_BASE_DIR"

if [ "$CHECK_ONLY" -eq 1 ]; then
  print_compatibility_report
  exit 0
fi

if [ "$DEPS_ONLY" -eq 1 ]; then
  install_system_deps
  build_llama_cpp
  setup_conversion_venv
  ensure_hf_cli
  log "Dependencies installed."
  exit 0
fi

# Check prerequisites
if [ ! -f "$LLAMA_COMPLETION" ]; then
  log "llama-completion not found at $LLAMA_COMPLETION"
  log "Run with --deps-only first to build llama.cpp, or set LLAMA_BIN in benchmark_config.sh"
  exit 1
fi

if ! ensure_hf_cli; then
  exit 1
fi

setup_conversion_venv

# Detect VRAM
VRAM_MB=$(detect_vram_mb)
if [ -z "$VRAM_MB" ] || [ "$VRAM_MB" -eq 0 ]; then
  log "WARNING: Could not detect VRAM via nvidia-smi. Skipping estimate check."
  VRAM_MB=999999
fi
log "Detected VRAM: ${VRAM_MB} MB"

print_compatibility_report

# Determine which models to install
INSTALL_IDS=()
if [ "${#REQUESTED_IDS[@]}" -gt 0 ]; then
  # Explicit list from CLI args
  INSTALL_IDS=("${REQUESTED_IDS[@]}")
else
  # Default: all models that pass the estimate check
  for id in "${ALL_MODEL_IDS[@]}"; do
    if model_fits_estimate "$id" "$VRAM_MB"; then
      INSTALL_IDS+=("$id")
    else
      log "Skipping (estimate: insufficient VRAM): ${MODEL_NAME[$id]}"
    fi
  done
fi

if [ "${#INSTALL_IDS[@]}" -eq 0 ]; then
  log "No models to install."
  exit 0
fi

log ""
log "Models to install: ${#INSTALL_IDS[@]}"
for id in "${INSTALL_IDS[@]}"; do
  log "  - ${MODEL_NAME[$id]}"
done

# Install
FAILED=()
SUCCEEDED=()
for id in "${INSTALL_IDS[@]}"; do
  if install_model "$id"; then
    SUCCEEDED+=("$id")
  else
    FAILED+=("$id")
  fi
done

# Summary
# Persist paths so benchmark.sh works on this machine
_write_paths_to_config

log_section "Installation Summary"
log "Succeeded: ${#SUCCEEDED[@]}"
for id in "${SUCCEEDED[@]}"; do
  log "  ✓ ${MODEL_NAME[$id]}"
done
if [ "${#FAILED[@]}" -gt 0 ]; then
  log "Failed: ${#FAILED[@]}"
  for id in "${FAILED[@]}"; do
    log "  ✗ ${MODEL_NAME[$id]}"
  done
fi
log ""
log "benchmark_config.sh has been updated with compatible models."
log "Restart the GUI (or click Refresh) to see them in the model list."
