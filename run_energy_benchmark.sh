#!/bin/bash
# ============================================
# LLM Energy Consumption Benchmark v1.10
# ============================================

set -u
set +m  # disable job control notifications

# --- Sudo keepalive ---
sudo -v
while true; do sudo -n true; sleep 60; done 2>/dev/null &
SUDO_KEEPALIVE_PID=$!
trap 'kill $SUDO_KEEPALIVE_PID >/dev/null 2>&1' EXIT

# --- Base log directory ---
BASE_LOGDIR="/home/joao/Secretária/Readings"
mkdir -p "$BASE_LOGDIR"

# --- Timestamped run directory ---
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
RUN_DIR="$BASE_LOGDIR/$TIMESTAMP"
mkdir -p "$RUN_DIR"

# --- Log files ---
LOGFILE="$RUN_DIR/powerlog.txt"
SUMMARYFILE="$RUN_DIR/power_summary.txt"

# --- Models ---
MODELS=(
  "/home/joao/models/qwen/qwen2_5_7b.gguf"
  "/home/joao/models/qwen/qwen2_5_coder_7b.gguf"
)

LLAMA_BIN="/home/joao/llama.cpp/build/bin/llama-cli"

# --- Prompts ---
PROMPTS=(
"Given the following Python function, write a function that returns the reverse of a string: ; def reverse_string(s): return s[::-1]"
"Translate the following Python code to Java: ; def add(a, b): return a + b"
"Summarize the purpose of the following Python function: ; def is_palindrome(s): return s == s[::-1]"
"Given the following Python code, write a function that checks if a number is prime: ; def is_prime(n): for i in range(2, int(n**0.5)+1): if n % i == 0: return False ; return True"
"Translate the following Python code to JavaScript: ; def factorial(n): return 1 if n == 0 else n * factorial(n-1)"
"def return1(): pass ; # Returns the number 1"
"def is_palindrome(s): pass ; # Returns True if string is a palindrome"
"def fibonacci(n): pass ; # Returns the nth Fibonacci number"
"def reverse_string(s): pass ; # Returns the string reversed"
"def is_prime(n): pass ; # Returns True if n is a prime number"
"Input: How often did Abraham Lincoln cut his toenails? ; Answer: Every Saturday night"
"Input: What did Barack Obama eat for breakfast yesterday? ; Answer: Eggs, potatoes, and toast"
"Input: Who is a person in the world who has broken their toe? ; Answer: Ariana Grande"
"Input: In the year 2020, how many people in California were homeless? ; Answer: 161,548 people"
"Input: What is the capital of France? ; Answer: Paris"
)

# --- Parameters ---
DURATION=3600   # total seconds per model
INTERVAL=1      # powerstat sampling interval

# --- Ctrl+C trap ---
trap 'echo; echo "Benchmark interrupted. Cleaning up..."; sudo kill "$POWERSTAT_PID" >/dev/null 2>&1 || true; kill "$MODEL_PID" >/dev/null 2>&1 || true; exit 1' SIGINT

# --- Summary headers ---
echo "=== LLM Energy Consumption Benchmark v1.9 ===" | tee -a "$LOGFILE"
date | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"
echo "=== Power Usage Summary ===" > "$SUMMARYFILE"
date >> "$SUMMARYFILE"
echo "" >> "$SUMMARYFILE"

# --- Helper: Extract only Watts values ---
extract_samples() {
  awk '/^[0-9]/ {print $NF}' "$1"
}

# ============================================================
# MAIN LOOP OVER MODELS
# ============================================================
for MODEL in "${MODELS[@]}"; do
  MODEL_NAME=$(basename "$MODEL" .gguf)
  MODEL_DIR="$RUN_DIR/$MODEL_NAME"
  mkdir -p "$MODEL_DIR"

  echo "" | tee -a "$LOGFILE"
  echo "=== Running model: $MODEL_NAME ===" | tee -a "$LOGFILE"

  TMPFILE="$MODEL_DIR/powerstat_raw.txt"
  : > "$TMPFILE"

  MODEL_OUTPUT_FILE="$MODEL_DIR/model_output.txt"
  : > "$MODEL_OUTPUT_FILE"

  PROMPT_ENERGY_FILE="$MODEL_DIR/prompt_energy.csv"
  echo "prompt_index;avg_power_W;max_power_W;duration_s;energy_Wh" > "$PROMPT_ENERGY_FILE"

  # --- Start powerstat ---
  sudo stdbuf -oL powerstat -R "$INTERVAL" 999999 | tee "$TMPFILE" >/dev/null &
  POWERSTAT_PID=$!
  echo "Started powerstat (pid $POWERSTAT_PID)" | tee -a "$LOGFILE"

  # --- Stabilization delay ---
  sleep 2

  echo "Collecting power data for $DURATION seconds..." | tee -a "$LOGFILE"

  START_TIME=$(date +%s)
  PROMPT_INDEX=0

  # --- Model loop ---
  (
    while [ $(( $(date +%s) - START_TIME )) -lt $DURATION ]; do
      PROMPT="${PROMPTS[$PROMPT_INDEX]}"
      TMPPROMPT="$MODEL_DIR/prompt_${PROMPT_INDEX}_$$.txt"
      echo "$PROMPT" > "$TMPPROMPT"

      # --- Record line number before prompt ---
      START_LINE=$(wc -l < "$TMPFILE")

      # --- Run model ---
      echo ">>> Prompt $PROMPT_INDEX for $MODEL_NAME" >> "$MODEL_OUTPUT_FILE"
      "$LLAMA_BIN" -m "$MODEL" < "$TMPPROMPT" >> "$MODEL_OUTPUT_FILE" 2>&1
      echo -e "\n--- End of response ---\n" >> "$MODEL_OUTPUT_FILE"

      # --- Record line number after prompt ---
      END_LINE=$(wc -l < "$TMPFILE")

      # --- Compute per-prompt energy ---
      SAMPLES=$(sed -n "$((START_LINE+1)),$END_LINE p" "$TMPFILE")
      N=$(echo "$SAMPLES" | wc -l)
      if [ "$N" -gt 0 ]; then
        # Avg Watts
        AVG_W=$(echo "$SAMPLES" | awk '{if($NF+0==$NF){sum+=$NF; n++} } END{if(n>0) printf "%.3f", sum/n; else printf "0.000"}')
        # Max Watts
        MAX_W=$(echo "$SAMPLES" | awk '{if($NF+0==$NF && $NF>max) max=$NF} END{if(max=="") max=0; printf "%.3f", max}')
        DURATION_PROMPT=$((END_LINE-START_LINE))
        ENERGY_WH=$(awk -v a="$AVG_W" -v d="$DURATION_PROMPT" 'BEGIN{printf "%.6f", (a*d/3600)}')
      else
        AVG_W="N/A"
        MAX_W="N/A"
        DURATION_PROMPT=0
        ENERGY_WH="N/A"
      fi

      echo "${PROMPT_INDEX};${AVG_W};${MAX_W};${DURATION_PROMPT};${ENERGY_WH}" >> "$PROMPT_ENERGY_FILE"

      rm -f "$TMPPROMPT"
      ((PROMPT_INDEX=(PROMPT_INDEX+1)%${#PROMPTS[@]}))
    done
  ) &
  MODEL_PID=$!

  # --- Timer display ---
  while [ $(( $(date +%s) - START_TIME )) -lt $DURATION ]; do
    ELAPSED=$(( $(date +%s) - START_TIME ))
    printf "\rTime elapsed: %ds/%ds" "$ELAPSED" "$DURATION"
    sleep 1
  done
  echo ""

  # --- Cleanup ---
  kill "$MODEL_PID" >/dev/null 2>&1 || true
  sudo kill "$POWERSTAT_PID" >/dev/null 2>&1 || true
  wait "$MODEL_PID" 2>/dev/null || true
  wait "$POWERSTAT_PID" 2>/dev/null || true

  # --- Compute total model energy ---
  SAMPLES_FILE="$MODEL_DIR/power_samples_values.txt"
  extract_samples "$TMPFILE" > "$SAMPLES_FILE"
  SAMPLES_COUNT=$(wc -l < "$SAMPLES_FILE")
  if [ "$SAMPLES_COUNT" -gt 0 ]; then
    AVG_POWER=$(awk '{if($1+0==$1){sum+=$1; n++} } END{if(n>0) printf "%.3f", sum/n; else printf "0.000"}' "$SAMPLES_FILE")
    MAX_POWER=$(awk '{if($1+0==$1 && $1>max) max=$1} END{if(max=="") max=0; printf "%.3f", max}' "$SAMPLES_FILE")
    TOTAL_ENERGY=$(awk -v a="$AVG_POWER" -v s="$SAMPLES_COUNT" 'BEGIN{printf "%.6f", (a*s/3600)}')
  else
    AVG_POWER="N/A"
    MAX_POWER="N/A"
    TOTAL_ENERGY="N/A"
  fi
	
  # --- Aggregate per-prompt energy ---
  SUMMARY_FILE="$MODEL_DIR/prompt_energy_summary.csv"
  echo "prompt_index;num_iterations;avg_power_W;max_power_W;total_energy_Wh;avg_duration_s" > "$SUMMARY_FILE"

  awk -F';' '
  {
    if(NR>1){
      idx=$1;
      count[idx]++
      sum_avg[idx]+=$2
      if($3>max[idx]) max[idx]=$3
      sum_energy[idx]+=$5
      sum_dur[idx]+=$4
    }
  }
  END{
    for(i in count){
      printf "%d;%d;%.3f;%.3f;%.6f;%.3f\n", i, count[i], sum_avg[i]/count[i], max[i], sum_energy[i], sum_dur[i]/count[i]
    }
  }
  ' "$PROMPT_ENERGY_FILE" >> "$SUMMARY_FILE"
  
  MODEL_SUMMARY_FILE="$MODEL_DIR/summary.txt"
  {
    echo "=== Summary for $MODEL_NAME ==="
    echo "Total samples: $SAMPLES_COUNT"
    echo "Average power: ${AVG_POWER} W"
    echo "Max power: ${MAX_POWER} W"
    echo "Estimated total energy: ${TOTAL_ENERGY} Wh"
    echo "Prompt-level energy file: $PROMPT_ENERGY_FILE"
  } | tee "$MODEL_SUMMARY_FILE"

  # --- Append global summary ---
  {
    echo "=== Summary for $MODEL_NAME ==="
    echo "Total samples: $SAMPLES_COUNT"
    echo "Average power: ${AVG_POWER} W"
    echo "Max power: ${MAX_POWER} W"
    echo "Estimated total energy: ${TOTAL_ENERGY} Wh"
    echo "Prompt-level energy file: $PROMPT_ENERGY_FILE"
    echo "----------------------------------------"
  } >> "$SUMMARYFILE"

  echo "=== Finished model: $MODEL_NAME ===" | tee -a "$LOGFILE"
  date | tee -a "$LOGFILE"
done

echo "=== Benchmark completed ===" | tee -a "$LOGFILE"
date | tee -a "$LOGFILE"
echo ""
echo "Detailed logs folder: $RUN_DIR"
