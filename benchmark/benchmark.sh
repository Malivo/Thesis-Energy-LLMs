#!/bin/bash
# ============================================
# LLM Energy + Temperature Benchmark v5.4
# Goal: measure and compare energy consumption
#       between LLM models at inference time.
# ============================================

set -u
set +m

# -----------------------------
# Load config (written by benchmark_gui.py)
# All variables below are defaults; sourcing the
# config file overrides any that the user has set.
# -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/benchmark_config.sh"
if [ -f "$CONFIG_FILE" ]; then
  # shellcheck source=/dev/null
  source "$CONFIG_FILE"
fi

# -----------------------------
# Timing (defaults - overridden by config if present)
# -----------------------------
DURATION=${DURATION:-3600}
INTERVAL=${INTERVAL:-1}
IDLE_DURATION=${IDLE_DURATION:-60}
PROMPT_TIMEOUT=${PROMPT_TIMEOUT:-300}
# RUN_MODE controls how long each model runs:
#   "duration" - cycle through prompts until DURATION seconds have elapsed
#   "all_once" - run every prompt exactly once, then move to the next model
RUN_MODE=${RUN_MODE:-duration}
# Number of full repetitions (each stored in run_N/ subdirectory).
# Per-run plots are generated after each run; aggregate plots after all runs.
N_RUNS=${N_RUNS:-3}

# -----------------------------
# Disk space guard
# -----------------------------
MIN_FREE_MB=${MIN_FREE_MB:-500}

# -----------------------------
# Sudo keepalive
#
# When run from a terminal: sudo -v prompts interactively as normal.
# When launched from the GUI (no terminal): the GUI pre-authenticates
# before launching, so the timestamp is already valid. We just extend
# it with sudo -n (non-interactive, never prompts). If somehow the
# timestamp has expired the keepalive will fail silently, but sudo
# commands later in the script will also fail - which is the correct
# visible failure mode.
# -----------------------------
if [ -t 0 ]; then
  # Running interactively - prompt normally
  sudo -v
else
  # No terminal - GUI has pre-authenticated, just verify quietly
  sudo -n true 2>/dev/null || {
    echo "[$(date "+%H:%M:%S")] ERROR: sudo authentication required. Please launch via the GUI." >&2
    exit 1
  }
fi
while true; do sudo -n true; sleep 60; done 2>/dev/null &
SUDO_KEEPALIVE_PID=$!
trap 'kill $SUDO_KEEPALIVE_PID >/dev/null 2>&1' EXIT

# Detect whether CPU power measurement (RAPL) is available.
# Virtualized machines (Xen, KVM) typically do not expose RAPL domains.
HAS_RAPL=1
if ! sudo powerstat -R 1 1 >/dev/null 2>&1; then
  HAS_RAPL=0
  echo "[$(date "+%H:%M:%S")] WARNING: powerstat/RAPL not available on this machine."
  echo "  CPU power will be N/A. GPU measurements are unaffected."
fi

# -----------------------------
# Paths (defaults - overridden by config if present)
# -----------------------------
BASE_LOGDIR="${BASE_LOGDIR:-$HOME/benchmark_results}"
# LLAMA_BIN is kept for compatibility but inference uses LLAMA_COMPLETION.
# Both are derived from LLAMA_DIR so only one path needs updating per machine.
LLAMA_DIR="${LLAMA_DIR:-$HOME/llama.cpp}"
LLAMA_BIN="${LLAMA_BIN:-$LLAMA_DIR/build/bin/llama-cli}"
LLAMA_COMPLETION="${LLAMA_COMPLETION:-$LLAMA_DIR/build/bin/llama-completion}"

mkdir -p "$BASE_LOGDIR"

FREE_MB=$(df -m "$BASE_LOGDIR" | awk 'NR==2 {print $4}')
if [ "$FREE_MB" -lt "$MIN_FREE_MB" ]; then
  echo "[$(date "+%H:%M:%S")] ERROR: Only ${FREE_MB} MB free in $BASE_LOGDIR (minimum: ${MIN_FREE_MB} MB). Aborting."
  exit 1
fi
echo "Disk space OK: ${FREE_MB} MB free."

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RUN_DIR="$BASE_LOGDIR/$TIMESTAMP"
mkdir -p "$RUN_DIR"

LOGFILE="$RUN_DIR/benchmark_log.txt"

# -----------------------------
# Models (default list - overridden by config if present)
# -----------------------------
if [ -z "${MODELS+x}" ]; then
  MODELS=(
    "/home/joao/models/qwen/qwen2_5_7b.gguf"
    "/home/joao/models/qwen/qwen2_5_coder_7b.gguf"
    "/home/joao/models/mistral/mistral_7b.Q4_K_M.gguf"
    "/home/joao/models/gemma/gemma3_4b.Q4_K_M.gguf"
  )
fi

# Per-model extra flags (e.g. --jinja for gpt-oss harmony format).
# Keys are the model basename (without path or .gguf suffix).
# Populated from benchmark_config.sh or set here as defaults.
declare -A MODEL_EXTRA_FLAGS
MODEL_EXTRA_FLAGS["gpt-oss-20b-mxfp4"]="--jinja"
MODEL_EXTRA_FLAGS["gpt-oss-120b-mxfp4"]="--jinja"
# Add more overrides here if needed, e.g.:
# MODEL_EXTRA_FLAGS["my_special_model"]="--some-flag"

# -----------------------------
# Prompts (defaults - overridden by config if present)
# -----------------------------
PROMPT_DIR="${PROMPT_DIR:-/home/joao/prompts}"
WARMUP_PROMPT="${WARMUP_PROMPT:-Say hello.}"

# Load prompts from language subdirectories if they exist, otherwise
# fall back to the flat structure (all .txt files directly in PROMPT_DIR).
# Produces two parallel arrays: PROMPT_TEXTS and PROMPT_LANGS.
declare -a PROMPT_TEXTS=()
declare -a PROMPT_LANGS=()
declare -a PROMPT_DIFFS=()

# Number of prompts per difficulty tier per file.
# With 90 prompts per file: lines 1-30 = easy, 31-60 = medium, 61-90 = hard.
# Set to 0 to disable difficulty classification (all tagged "unclassified").
DIFFICULTY_BLOCK_SIZE=${DIFFICULTY_BLOCK_SIZE:-30}

_load_prompts() {
  local dir="$1" lang="$2"
  local file line line_num diff
  for file in "$dir"/*.txt; do
    [ -f "$file" ] || continue
    line_num=0
    while IFS= read -r line || [ -n "$line" ]; do
      # Skip blank/whitespace-only lines
      [[ "$line" =~ ^[[:space:]]*$ ]] && continue
      line="${line%$'\r'}"   # strip Windows carriage return if present
      line_num=$(( line_num + 1 ))

      # Classify difficulty by position within file
      if [ "$DIFFICULTY_BLOCK_SIZE" -gt 0 ]; then
        if [ "$line_num" -le "$DIFFICULTY_BLOCK_SIZE" ]; then
          diff="easy"
        elif [ "$line_num" -le $(( DIFFICULTY_BLOCK_SIZE * 2 )) ]; then
          diff="medium"
        else
          diff="hard"
        fi
      else
        diff="unclassified"
      fi

      PROMPT_TEXTS+=("$line")
      PROMPT_LANGS+=("$lang")
      PROMPT_DIFFS+=("$diff")
    done < "$file"
  done
}

# Detect whether PROMPT_DIR contains language subdirectories
_has_lang_dirs=0
for _d in "$PROMPT_DIR"/*/; do
  [ -d "$_d" ] && { _has_lang_dirs=1; break; }
done

if [ "$_has_lang_dirs" -eq 1 ]; then
  echo "Language subdirectories detected in $PROMPT_DIR"
  for _lang_dir in "$PROMPT_DIR"/*/; do
    [ -d "$_lang_dir" ] || continue
    _lang=$(basename "$_lang_dir")
    _load_prompts "$_lang_dir" "$_lang"
    echo "  Loaded language: $_lang (${#PROMPT_TEXTS[@]} prompts total so far)"
  done
else
  echo "No language subdirectories found - loading prompts from flat directory"
  _load_prompts "$PROMPT_DIR" "unknown"
fi

if [ "${#PROMPT_TEXTS[@]}" -eq 0 ]; then
  echo "[$(date "+%H:%M:%S")] ERROR: No prompts found in $PROMPT_DIR" >&2
  exit 1
fi
echo "Total prompts loaded: ${#PROMPT_TEXTS[@]}"

# -----------------------------
# Ctrl+C cleanup
# -----------------------------
trap 'echo "Interrupted"; \
sudo kill "$POWERSTAT_PID" 2>/dev/null || true; \
kill "$TEMP_PID" 2>/dev/null || true; \
kill "$MODEL_PID" 2>/dev/null || true; \
kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true; \
exit 1' SIGINT

# ============================================================
# LOGGER HELPERS
# ============================================================
start_loggers() {
  local power_raw="$1"
  local cpu_temp_file="$2"
  local gpu_temp_file="$3"

  if [ "$HAS_RAPL" -eq 1 ]; then
    sudo stdbuf -oL powerstat -R "$INTERVAL" 999999 | tee "$power_raw" >/dev/null &
    POWERSTAT_PID=$!
  else
    touch "$power_raw"
    POWERSTAT_PID=0
  fi

  (
    while true; do
      sleep "$INTERVAL"
      TS=$(date +%s)

      # CPU temperature: gracefully handle missing sensors.
      # Some machines (especially servers) have no lm-sensors modules
      # loaded or no thermal_zone exposed.
      CPU_PACKAGE=$(sensors 2>/dev/null | awk '/Package id 0:/ {gsub("\\+|°C","",$4); print $4}')
      CPU_AVG=$(sensors 2>/dev/null | awk '/Core [0-9]+:/ \
        {gsub("\\+|°C","",$3); s+=$3; n++} END{if(n>0) printf "%.1f",s/n}')
      if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
        SYS_RAW=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null)
        SYS_TEMP=$(awk "BEGIN{printf \"%.1f\",${SYS_RAW:-0}/1000}")
      else
        SYS_TEMP="0"
      fi
      CPU_PACKAGE="${CPU_PACKAGE:-0}"
      CPU_AVG="${CPU_AVG:-0}"
      echo "$TS,$CPU_PACKAGE,$CPU_AVG,$SYS_TEMP" >> "$cpu_temp_file"

      # Query each field separately to avoid locale comma-decimal issues.
      # Guard against nvidia-smi driver failures: write zeros on failure
      # so the log row is valid and the run can continue without corrupt data.
      if ! nvidia-smi -L >/dev/null 2>&1; then
        echo "$TS,0,0,0,0" >> "$gpu_temp_file"
      else
        GPU_TEMP_C=$(nvidia-smi --query-gpu=temperature.gpu \
          --format=csv,noheader,nounits 2>/dev/null | tr -d ' \r')
        GPU_PWR_W=$(nvidia-smi --query-gpu=power.draw \
          --format=csv,noheader,nounits 2>/dev/null | tr -d ' \r' | tr ',' '.')
        GPU_UTIL_P=$(nvidia-smi --query-gpu=utilization.gpu \
          --format=csv,noheader,nounits 2>/dev/null | tr -d ' \r')
        GPU_MEM_M=$(nvidia-smi --query-gpu=memory.used \
          --format=csv,noheader,nounits 2>/dev/null | tr -d ' \r')
        if [[ "$GPU_PWR_W" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
          echo "$TS,$GPU_TEMP_C,$GPU_PWR_W,$GPU_UTIL_P,$GPU_MEM_M" >> "$gpu_temp_file"
        else
          echo "[$(date '+%H:%M:%S')] WARNING: nvidia-smi returned non-numeric power, skipping sample." >&2
        fi
      fi
    done
  ) &
  TEMP_PID=$!
}

stop_loggers() {
  kill "$TEMP_PID"           2>/dev/null || true
  if [ "$POWERSTAT_PID" -gt 0 ]; then
    sudo kill "$POWERSTAT_PID" 2>/dev/null || true
  fi
  sleep 2   # allow final writes to flush before summary awk reads the files
}

# ============================================================
# SAMPLE EXTRACTION HELPERS
#
# cpu_samples_in_window uses FS="[[:space:]]+" and matches
# /^[0-9]{2}:[0-9]{2}:[0-9]{2}/ - restored to the user's
# working v5.3 version which produced correct CPU readings.
# $1=HH:MM:SS, $13=watts (hardcoded field, matches powerstat -R).
# ============================================================
cpu_samples_in_window() {
  local power_raw="$1" start="$2" end="$3" today="$4"
  awk \
    -v s="$start" -v e="$end" -v d="$today" '
    BEGIN { FS = "[[:space:]]+" }
    /^[0-9]{2}:[0-9]{2}:[0-9]{2}/ {
      if (NF >= 13) {
        timestr = $1
        watts = $13
        gsub(/[, ]+$/, "", watts)
        cmd = "date -d \"" d " " timestr "\" +%s"
        cmd | getline epoch; close(cmd)
        if (epoch+0 >= s+0-1 && epoch+0 <= e+0+1 && watts ~ /^[0-9]*\.?[0-9]+$/)
          print watts
      }
    }
  ' "$power_raw"
}

# Extract gpu_power_W (column 3) for rows in [start, end).
gpu_samples_in_window() {
  local gpu_temp_file="$1" start="$2" end="$3"
  awk -F',' -v s="$start" -v e="$end" '
    NR==1 { next }
    $1+0 >= s+0-1 && $1+0 <= e+0+1 { print $3 }
  ' "$gpu_temp_file"
}

# Given a newline-separated list of watt values and a duration,
# prints: avg  max  energy_Wh
# Prints "N/A N/A N/A" if input is empty.
summarise_watts() {
  local samples="$1" dur="$2"
  if [ -z "$samples" ]; then
    echo "N/A N/A N/A"
    return
  fi
  echo "$samples" | awk -v dur="$dur" '
    { if($1>m) m=$1; sum+=$1; n++ }
    END { printf "%.3f %.3f %.6f\n", sum/n, m, (sum/n)*dur/3600 }
  '
}

# ============================================================
# IDLE BASELINE
# ============================================================
measure_idle_baseline() {
  echo "=== Measuring idle baseline (${IDLE_DURATION}s) ===" | tee -a "$LOGFILE"

  local idle_dir="$RUN_DIR/idle"
  mkdir -p "$idle_dir/temperature"

  local power_raw="$idle_dir/powerstat_raw.txt"
  local cpu_temp="$idle_dir/temperature/cpu_temperature.csv"
  local gpu_temp="$idle_dir/temperature/gpu_temperature.csv"

  : > "$power_raw"
  echo "timestamp_s,cpu_package_C,cpu_core_avg_C,system_C"            > "$cpu_temp"
  echo "timestamp_s,gpu_temp_C,gpu_power_W,gpu_util_pct,mem_used_MiB" > "$gpu_temp"

  start_loggers "$power_raw" "$cpu_temp" "$gpu_temp"
  sleep 2

  local t0; t0=$(date +%s)
  local elapsed=0
  while [ "$elapsed" -lt "$IDLE_DURATION" ]; do
    echo "  Idle baseline: ${elapsed}s / ${IDLE_DURATION}s..." | tee -a "$LOGFILE"
    sleep 10
    elapsed=$(( $(date +%s) - t0 ))
  done
  local t1; t1=$(date +%s)

  stop_loggers

  local today; today=$(date +%Y-%m-%d)
  local cpu_s; cpu_s=$(cpu_samples_in_window "$power_raw" "$t0" "$t1" "$today")
  local gpu_s; gpu_s=$(gpu_samples_in_window "$gpu_temp"  "$t0" "$t1")

  read -r IDLE_CPU_AVG_W _ _ <<< "$(summarise_watts "$cpu_s" "$IDLE_DURATION")"
  read -r IDLE_GPU_AVG_W _ _ <<< "$(summarise_watts "$gpu_s" "$IDLE_DURATION")"

  [ "$IDLE_CPU_AVG_W" = "N/A" ] && IDLE_CPU_AVG_W="0"
  [ "$IDLE_GPU_AVG_W" = "N/A" ] && IDLE_GPU_AVG_W="0"

  {
    echo "IDLE BASELINE"
    echo "CPU idle avg : $IDLE_CPU_AVG_W W"
    echo "GPU idle avg : $IDLE_GPU_AVG_W W"
    echo ""
  } | tee -a "$LOGFILE"
}

# ============================================================
# ENTRY POINT
# ============================================================
echo "=== LLM Energy + Temperature Benchmark v5.4 ===" | tee -a "$LOGFILE"
echo "Runs: $N_RUNS  |  Mode: $RUN_MODE  |  Duration: ${DURATION}s" | tee -a "$LOGFILE"
date | tee -a "$LOGFILE"
echo | tee -a "$LOGFILE"

# Idle baseline runs once for the whole session, before any model is loaded.
IDLE_CPU_AVG_W="0"
IDLE_GPU_AVG_W="0"
measure_idle_baseline

# Verify powerstat parsing works before committing to a full run
echo "=== Verification ===" | tee -a "$LOGFILE"
if [ "$HAS_RAPL" -eq 1 ]; then
  TEST_SAMPLE=$(cpu_samples_in_window \
    "$RUN_DIR/idle/powerstat_raw.txt" \
    "$(date -d '2 minutes ago' +%s)" \
    "$(date +%s)" \
    "$(date +%Y-%m-%d)" 2>/dev/null | head -1)
  if [ -n "$TEST_SAMPLE" ] && [ "$TEST_SAMPLE" != "0" ]; then
    echo "Powerstat parsing OK: sample=${TEST_SAMPLE} W" | tee -a "$LOGFILE"
  else
    echo "[$(date "+%H:%M:%S")] WARNING: Powerstat parsing returned no samples - CPU measurements will be N/A." \
      | tee -a "$LOGFILE"
  fi
else
  echo "RAPL not available - skipping powerstat verification." | tee -a "$LOGFILE"
fi
echo "" | tee -a "$LOGFILE"

# ── GPU telemetry sanity check ──────────────────────────────────────────────
# Verify nvidia-smi returns valid power readings before committing to a run.
# A driver failure here would produce zero GPU energy for every model.
GPU_CHECK=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | tr ',' '.' | tr -d ' ')
if [ -z "$GPU_CHECK" ] || [ "$GPU_CHECK" = "0" ]; then
  echo "[$(date "+%H:%M:%S")] ERROR: nvidia-smi returned no valid GPU power reading." | tee -a "$LOGFILE"
  echo "  GPU telemetry is not working. Possible causes:" | tee -a "$LOGFILE"
  echo "  - NVIDIA driver not loaded (try: sudo modprobe nvidia)" | tee -a "$LOGFILE"
  echo "  - Kernel/driver mismatch after a system update" | tee -a "$LOGFILE"
  echo "  - nvidia-smi not installed" | tee -a "$LOGFILE"
  echo "  Aborting to avoid recording invalid measurements." | tee -a "$LOGFILE"
  exit 1
else
  echo "GPU telemetry OK: current power draw = ${GPU_CHECK} W" | tee -a "$LOGFILE"
fi
echo "" | tee -a "$LOGFILE"

# ============================================================
# RUN LOOP
# ============================================================
for RUN_N in $(seq 1 "$N_RUNS"); do
  RUN_DIR_N="$RUN_DIR/run_$RUN_N"
  mkdir -p "$RUN_DIR_N"

  echo "" | tee -a "$LOGFILE"
  echo "=== Run $RUN_N of $N_RUNS ===" | tee -a "$LOGFILE"
  echo "" | tee -a "$LOGFILE"

# ============================================================
# MAIN MODEL LOOP
# ============================================================
for MODEL in "${MODELS[@]}"; do
  MODEL_NAME=$(basename "$MODEL" .gguf)
  MODEL_DIR="$RUN_DIR_N/$MODEL_NAME"
  mkdir -p "$MODEL_DIR"

  # llama.cpp clamps --n-gpu-layers to however many layers
  # actually fit in VRAM, so 999 always maximises GPU offload.
  GPU_LAYERS=999

  # Temperature files
  TEMP_DIR="$MODEL_DIR/temperature"
  mkdir -p "$TEMP_DIR"
  CPU_TEMP_FILE="$TEMP_DIR/cpu_temperature.csv"
  GPU_TEMP_FILE="$TEMP_DIR/gpu_temperature.csv"
  echo "timestamp_s,cpu_package_C,cpu_core_avg_C,system_C"            > "$CPU_TEMP_FILE"
  echo "timestamp_s,gpu_temp_C,gpu_power_W,gpu_util_pct,mem_used_MiB" > "$GPU_TEMP_FILE"

  # Reset logger PIDs - prevents stale PIDs from a previous model
  # iteration being used if stop_loggers is called before start_loggers
  POWERSTAT_PID=0
  TEMP_PID=0

  # Power + output files
  POWER_RAW="$MODEL_DIR/powerstat_raw.txt"
  : > "$POWER_RAW"
  MODEL_OUTPUT="$MODEL_DIR/model_output.txt"
  : > "$MODEL_OUTPUT"

  PROMPT_ENERGY="$MODEL_DIR/prompt_energy.csv"
  echo "prompt_index;language;difficulty;cpu_avg_W;cpu_max_W;gpu_avg_W;gpu_max_W;\
duration_s;cpu_energy_Wh;gpu_energy_Wh;\
cpu_net_avg_W;gpu_net_avg_W;cpu_net_Wh;gpu_net_Wh;\
tokens;tokens_per_s;Wh_per_token;throttled" > "$PROMPT_ENERGY"

  TIMEOUT_LOG="$MODEL_DIR/timeout_log.csv"
  echo "timestamp;prompt_index;language;difficulty" > "$TIMEOUT_LOG"

  echo "=== Running model: $MODEL_NAME (GPU layers: $GPU_LAYERS) ===" \
    | tee -a "$LOGFILE"

  # Warmup BEFORE start_loggers - cold-start energy not recorded
  # Look up any extra flags for this model (e.g. --jinja for gpt-oss)
  _MODEL_EXTRA="${MODEL_EXTRA_FLAGS[$MODEL_NAME]:-}"

  echo "  Warming up $MODEL_NAME..." | tee -a "$LOGFILE"
  # shellcheck disable=SC2086
  # shellcheck disable=SC2086
  timeout "$PROMPT_TIMEOUT" "$LLAMA_COMPLETION" \
      -m "$MODEL" --n-gpu-layers "$GPU_LAYERS" \
      -p "$WARMUP_PROMPT" \
      --log-disable \
      --single-turn \
      $_MODEL_EXTRA \
      >/dev/null 2>&1 || true
  sleep 2

  start_loggers "$POWER_RAW" "$CPU_TEMP_FILE" "$GPU_TEMP_FILE"
  sleep 2

  MODEL_PID=0
  START_TIME=$(date +%s)
  PROMPT_INDEX=0
  export TODAY
  TODAY=$(date +%Y-%m-%d)

  # Snapshot all paths and baseline values before the subshell.
  # Scalars are inherited by subshells; associative arrays are NOT,
  # so MODEL_EXTRA_FLAGS must be resolved to a plain string here.
  _POWER_RAW="$POWER_RAW"
  _GPU_TEMP_FILE="$GPU_TEMP_FILE"
  _MODEL_OUTPUT="$MODEL_OUTPUT"
  _PROMPT_ENERGY="$PROMPT_ENERGY"
  _MODEL_DIR="$MODEL_DIR"
  _IDLE_CPU="$IDLE_CPU_AVG_W"
  _IDLE_GPU="$IDLE_GPU_AVG_W"
  _N_PROMPTS="${#PROMPT_TEXTS[@]}"
  _EXTRA_FLAGS="${MODEL_EXTRA_FLAGS[$MODEL_NAME]:-}"
  _TIMEOUT_LOG="$TIMEOUT_LOG"
  # DIFF is read per-prompt inside the loop from PROMPT_DIFFS array
  # (arrays are inherited by subshells in bash)

  # ------------------------------------------------------------------
  # Inference loop
  # ------------------------------------------------------------------
  (
    # Loop condition depends on RUN_MODE:
    #   duration - keep going until DURATION seconds elapsed
    #   all_once - stop after every prompt has run once (PROMPT_INDEX reaches N)
    _done() {
      if [ "$RUN_MODE" = "all_once" ]; then
        [ "$PROMPT_INDEX" -ge "$_N_PROMPTS" ]
      else
        [ $(( $(date +%s) - START_TIME )) -ge $DURATION ]
      fi
    }
    while ! _done; do
      PROMPT="${PROMPT_TEXTS[$PROMPT_INDEX]}"
      LANG="${PROMPT_LANGS[$PROMPT_INDEX]}"
      DIFF="${PROMPT_DIFFS[$PROMPT_INDEX]:-unclassified}"

      INFER_START=$(date +%s)

      LLAMA_STDOUT=$(mktemp)
      LLAMA_STDERR=$(mktemp)
      # shellcheck disable=SC2086
      # stdout: model response text
      # stderr: timing lines (common_perf_print) used for token parsing
      # --log-disable is intentionally absent: it suppresses common_perf_print
      # which contains the token counts and speeds we need for Wh/token.
      if ! timeout "$PROMPT_TIMEOUT" "$LLAMA_COMPLETION" \
            -m "$MODEL" --n-gpu-layers "$GPU_LAYERS" \
            -p "$PROMPT" \
            --single-turn \
            $_EXTRA_FLAGS \
            >> "$LLAMA_STDOUT" 2>"$LLAMA_STDERR"; then
        echo "[$(date "+%H:%M:%S")] WARN: inference failed or timed out on prompt $PROMPT_INDEX [$LANG/$DIFF]" >&2
        # Record the timeout so it appears in analysis
        echo "$(date "+%Y-%m-%dT%H:%M:%S");$PROMPT_INDEX;$LANG;$DIFF" >> "$_TIMEOUT_LOG"
        rm -f "$LLAMA_STDERR" "$LLAMA_STDOUT"
        if [ "$RUN_MODE" = "all_once" ]; then
          ((PROMPT_INDEX=PROMPT_INDEX+1))
        else
          ((PROMPT_INDEX=(PROMPT_INDEX+1)%_N_PROMPTS))
        fi
        continue
      fi

      INFER_END=$(date +%s)
      DUR=$(( INFER_END - INFER_START ))

      # Append model output (strip UI chrome: banner, prompt echo, > lines)
      # llama-completion stdout is clean - append directly to model output
      cat "$LLAMA_STDOUT" >> "$_MODEL_OUTPUT" || true

      # ------------------------------------------------------------------
      # Token throughput parsing
      #
      # llama-completion (b9124+) confirmed output format:
      #   common_perf_print: prompt eval time = X ms / N tokens (... S tokens per second)
      #   common_perf_print:       eval time  = X ms / N runs   (... S tokens per second)
      #   common_perf_print:      total time  = X ms / N tokens
      # Normalise locale decimal separator (comma → dot) before parsing.
      #
      # Debug files saved per run:
      #   llama_stderr.log       - all prompts labelled
      #   llama_stderr_debug.txt - first prompt only
      # ------------------------------------------------------------------
      STDERR_LOG="$_MODEL_DIR/llama_stderr.log"
      echo "=== prompt $PROMPT_INDEX [$LANG/$DIFF] ===" >> "$STDERR_LOG"
      cat "$LLAMA_STDERR" >> "$STDERR_LOG"
      cat "$LLAMA_STDOUT" >> "$STDERR_LOG"
      if [ "$PROMPT_INDEX" -eq 0 ]; then
        cp "$LLAMA_STDERR" "$_MODEL_DIR/llama_stderr_debug.txt"
        cp "$LLAMA_STDOUT" "$_MODEL_DIR/llama_stdout_debug.txt"
      fi

      # Normalise locale decimal separator once
      STDERR_NORM=$(tr ',' '.' < "$LLAMA_STDERR")

      # Generated tokens from "eval time" line: "eval time = X ms / N runs"
      # This is the generation count only, excluding prompt tokens.
      # Used for Wh/token so we measure cost per output token, not total.
      TOKENS=$(echo "$STDERR_NORM" | \
        grep 'eval time' | grep -v 'prompt' | \
        grep -oP '/\s*\K[0-9]+(?=\s+runs)' | tail -1)

      # Fallback: total tokens if eval runs not found
      if [ -z "$TOKENS" ] || [ "$TOKENS" = "0" ]; then
        TOKENS=$(echo "$STDERR_NORM" | \
          grep -oP 'total time\s*=\s*[0-9.]+ ms\s*/\s*\K[0-9]+(?=\s+tokens)' | tail -1)
      fi

      # Generation speed from "eval time" line (excludes prompt processing)
      TOKENS_PS=$(echo "$STDERR_NORM" | \
        grep 'eval time' | grep -v 'prompt' | \
        grep -oP '[0-9]+\.?[0-9]*(?=\s+tokens per second)' | tail -1)

      rm -f "$LLAMA_STDERR" "$LLAMA_STDOUT"
      [ -z "$TOKENS_PS" ] && TOKENS_PS="0.0"

      # If total token count still missing, estimate from tokens/s x duration
      if [ -z "$TOKENS" ] || [ "$TOKENS" = "0" ]; then
        if [ "$TOKENS_PS" != "0.0" ] && [ "$DUR" -gt 0 ]; then
          TOKENS=$(awk -v tps="$TOKENS_PS" -v dur="$DUR" \
            'BEGIN{printf "%d", tps * dur}')
        else
          TOKENS="0"
        fi
      fi

      # CPU power window
      CPU_S=$(cpu_samples_in_window "$_POWER_RAW" "$INFER_START" "$INFER_END" "$TODAY")
      read -r AVG_W MAX_W ENERGY <<< "$(summarise_watts "$CPU_S" "$DUR")"

      # GPU power window
      GPU_S=$(gpu_samples_in_window "$_GPU_TEMP_FILE" "$INFER_START" "$INFER_END")
      read -r GPU_AVG_W GPU_MAX_W GPU_ENERGY <<< "$(summarise_watts "$GPU_S" "$DUR")"

      # Net energy = gross − idle baseline (clamped at 0)
      compute_net() {
        local raw="$1" idle="$2" dur="$3"
        if [ "$raw" = "N/A" ] || [ "$idle" = "N/A" ]; then
          echo "N/A N/A"
        else
          awk -v r="$raw" -v i="$idle" -v d="$dur" \
            'BEGIN{ n=r-i; if(n<0)n=0; printf "%.3f %.6f\n", n, n*d/3600 }'
        fi
      }
      read -r CPU_NET_W CPU_NET_WH <<< "$(compute_net "$AVG_W"     "$_IDLE_CPU" "$DUR")"
      read -r GPU_NET_W GPU_NET_WH <<< "$(compute_net "$GPU_AVG_W" "$_IDLE_GPU" "$DUR")"

      # Wh per token (net GPU energy / tokens generated)
      WH_PER_TOKEN="N/A"
      if [ "$GPU_NET_WH" != "N/A" ] && [ "$TOKENS" -gt 0 ] 2>/dev/null; then
        WH_PER_TOKEN=$(awk -v e="$GPU_NET_WH" -v t="$TOKENS" \
                         'BEGIN{printf "%.8f", e/t}')
      fi

      # Thermal throttling: high temp (>80°C) with low utilisation (<50%)
      THROTTLED=0
      AVG_UTIL=$(awk -F',' -v s="$INFER_START" -v e="$INFER_END" '
        NR==1 { next }
        $1+0 >= s+0 && $1+0 < e+0 { sum+=$4; n++ }
        END { if(n>0) printf "%.0f", sum/n; else print "0" }
      ' "$_GPU_TEMP_FILE")

      MAX_TEMP=$(awk -F',' -v s="$INFER_START" -v e="$INFER_END" '
        NR==1 { next }
        $1+0 >= s+0 && $1+0 < e+0 { if($2>m) m=$2 }
        END { printf "%.0f", m+0 }
      ' "$_GPU_TEMP_FILE")

      if [ -n "$AVG_UTIL" ] && [ -n "$MAX_TEMP" ]; then
        if [ "$MAX_TEMP" -gt 80 ] && [ "$AVG_UTIL" -lt 50 ]; then
          THROTTLED=1
        fi
      fi

      echo "$PROMPT_INDEX;$LANG;$DIFF;$AVG_W;$MAX_W;$GPU_AVG_W;$GPU_MAX_W;\
$DUR;$ENERGY;$GPU_ENERGY;\
$CPU_NET_W;$GPU_NET_W;$CPU_NET_WH;$GPU_NET_WH;\
$TOKENS;$TOKENS_PS;$WH_PER_TOKEN;$THROTTLED" >> "$_PROMPT_ENERGY"

      if [ "$RUN_MODE" = "all_once" ]; then
        ((PROMPT_INDEX=PROMPT_INDEX+1))
      else
        ((PROMPT_INDEX=(PROMPT_INDEX+1)%_N_PROMPTS))
      fi
    done
  ) &
  MODEL_PID=$!

  if [ "$RUN_MODE" = "all_once" ]; then
    # Wait for the subshell to finish all prompts (no time limit)
    echo "  Waiting for all prompts to complete..." | tee -a "$LOGFILE"
    wait "$MODEL_PID" 2>/dev/null || true
  else
    # Time-based mode - print progress every 30s then kill after DURATION
    _wait_start=$(date +%s)
    while [ $(( $(date +%s) - START_TIME )) -lt $DURATION ]; do
      _elapsed_run=$(( $(date +%s) - _wait_start ))
      if (( _elapsed_run % 30 == 0 )) && [ "$_elapsed_run" -gt 0 ]; then
        echo "  $MODEL_NAME: ${_elapsed_run}s / ${DURATION}s elapsed..." | tee -a "$LOGFILE"
      fi
      sleep 1
    done
    kill "$MODEL_PID" 2>/dev/null || true
  fi
  stop_loggers

  # ------------------------------------------------------------------
  # Per-model final summary - same FS and regex as cpu_samples_in_window
  # FS="[[:space:]]+" / /^[0-9]{2}:[0-9]{2}:[0-9]{2}/ / $13=watts
  # ------------------------------------------------------------------
  CPU_TOTAL_ENERGY=$(awk '
    BEGIN { FS = "[[:space:]]+" }
    /^[0-9]{2}:[0-9]{2}:[0-9]{2}/ {
      if (NF >= 13) { watts = $13; gsub(/[, ]+$/, "", watts); sum += watts }
    }
    END { printf "%.4f", sum * '"$INTERVAL"' / 3600 }
  ' "$POWER_RAW")

  CPU_AVG_PWR=$(awk '
    BEGIN { FS = "[[:space:]]+" }
    /^[0-9]{2}:[0-9]{2}:[0-9]{2}/ {
      if (NF >= 13) { watts = $13; gsub(/[, ]+$/, "", watts); sum += watts; c++ }
    }
    END { if (c > 0) printf "%.2f", sum / c; else print "0.00" }
  ' "$POWER_RAW")

  CPU_MAX_PWR=$(awk '
    BEGIN { FS = "[[:space:]]+" }
    /^[0-9]{2}:[0-9]{2}:[0-9]{2}/ {
      if (NF >= 13) { watts = $13; gsub(/[, ]+$/, "", watts); if (watts > m) m = watts }
    }
    END { printf "%.2f", m+0 }
  ' "$POWER_RAW")

  CPU_SAMPLES_TOTAL=$(awk '
    BEGIN { FS = "[[:space:]]+" }
    /^[0-9]{2}:[0-9]{2}:[0-9]{2}/ { n++ }
    END { print n+0 }
  ' "$POWER_RAW")

  GPU_TOTAL_ENERGY=$(awk -F',' \
    'NR>1 {sum+=$3} END {printf "%.4f", sum*'"$INTERVAL"'/3600}' "$GPU_TEMP_FILE")
  GPU_AVG_PWR=$(awk -F',' \
    'NR>1 {sum+=$3; n++} END {if(n>0) printf "%.2f", sum/n; else print "0.00"}' "$GPU_TEMP_FILE")
  GPU_MAX_PWR=$(awk -F',' \
    'NR>1 {if($3>m) m=$3} END {printf "%.2f", m+0}' "$GPU_TEMP_FILE")
  GPU_SAMPLES_TOTAL=$(awk 'NR>1 {n++} END {print n+0}' "$GPU_TEMP_FILE")

  {
    echo ""
    echo "CPU SUMMARY ($MODEL_NAME)"
    echo "Average CPU Power : $CPU_AVG_PWR W  (idle baseline: $IDLE_CPU_AVG_W W)"
    echo "Max CPU Power     : $CPU_MAX_PWR W"
    echo "Total CPU Energy  : $CPU_TOTAL_ENERGY Wh"
    echo ""
    echo "GPU SUMMARY ($MODEL_NAME)"
    echo "Average GPU Power : $GPU_AVG_PWR W  (idle baseline: $IDLE_GPU_AVG_W W)"
    echo "Max GPU Power     : $GPU_MAX_PWR W"
    echo "Total GPU Energy  : $GPU_TOTAL_ENERGY Wh"
    echo ""
    echo "Samples collected:"
    echo "CPU samples : $CPU_SAMPLES_TOTAL"
    echo "GPU samples : $GPU_SAMPLES_TOTAL"
  } | tee -a "$LOGFILE"

  {
    echo "idle_cpu_avg_W=$IDLE_CPU_AVG_W"
    echo "idle_gpu_avg_W=$IDLE_GPU_AVG_W"
  } > "$MODEL_DIR/idle_baseline.txt"

done   # end model loop

  # -------------------------------------------------
  # Per-run analysis (plots for this run only)
  # -------------------------------------------------
  VENV_DIR="$RUN_DIR/venv"

  if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip
    REQ_FILE="$(dirname "$0")/requirements.txt"
    if [ -f "$REQ_FILE" ]; then
      "$VENV_DIR/bin/pip" install -r "$REQ_FILE"
    else
      "$VENV_DIR/bin/pip" install pandas matplotlib numpy
    fi
  fi

  echo "Analysing run $RUN_N..." | tee -a "$LOGFILE"
  if "$VENV_DIR/bin/python" "$(dirname "$0")/analyze_results.py" "$RUN_DIR_N"; then
    echo "Run $RUN_N plots saved to: $RUN_DIR_N/comparison" | tee -a "$LOGFILE"
  else
    echo "[$(date "+%H:%M:%S")] WARNING: Analysis failed for run $RUN_N." | tee -a "$LOGFILE"
  fi

done   # end run loop

# -------------------------------------------------
# Aggregate analysis across all N_RUNS runs
# -------------------------------------------------
echo "" | tee -a "$LOGFILE"
echo "=== Aggregate analysis ($N_RUNS runs) ===" | tee -a "$LOGFILE"
if "$VENV_DIR/bin/python" "$(dirname "$0")/aggregate_analysis.py" "$RUN_DIR"; then
  echo "Aggregate plots saved to: $RUN_DIR/aggregate" | tee -a "$LOGFILE"
else
  echo "[$(date "+%H:%M:%S")] WARNING: Aggregate analysis failed." | tee -a "$LOGFILE"
fi

