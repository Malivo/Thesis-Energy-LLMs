#!/bin/bash
# ============================================================
# rerun_analysis.sh - Regenerate all plots from existing data
#
# Re-runs analyze_results.py on each run_N/ subdirectory, then
# runs aggregate_analysis.py on the session directory.
# No experiments are re-run; only the CSV data is re-analyzed.
#
# Usage:
#   ./rerun_analysis.sh /path/to/2026-05-29_17-07-16
#   ./rerun_analysis.sh   (uses most recent session in BASE_LOGDIR)
# ============================================================

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source config for BASE_LOGDIR if available
CONFIG_FILE="$SCRIPT_DIR/benchmark_config.sh"
if [ -f "$CONFIG_FILE" ]; then
  source "$CONFIG_FILE"
fi
BASE_LOGDIR="${BASE_LOGDIR:-$HOME/benchmark_results}"

# Determine session directory
if [ -n "${1:-}" ]; then
  SESSION_DIR="$1"
else
  # Find the most recent session directory
  SESSION_DIR=$(find "$BASE_LOGDIR" -maxdepth 1 -type d -name "20*" | sort -r | head -1)
  if [ -z "$SESSION_DIR" ]; then
    echo "ERROR: No session directories found in $BASE_LOGDIR"
    exit 1
  fi
  echo "No path given, using most recent session: $SESSION_DIR"
fi

if [ ! -d "$SESSION_DIR" ]; then
  echo "ERROR: Directory not found: $SESSION_DIR"
  exit 1
fi

echo "=== Re-running analysis ==="
echo "Session: $SESSION_DIR"
echo ""

# Find or create the venv
VENV_DIR="$SESSION_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating Python virtual environment..."
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --upgrade pip
  REQ_FILE="$SCRIPT_DIR/requirements.txt"
  if [ -f "$REQ_FILE" ]; then
    "$VENV_DIR/bin/pip" install -r "$REQ_FILE"
  else
    "$VENV_DIR/bin/pip" install pandas matplotlib numpy
  fi
fi

PYTHON="$VENV_DIR/bin/python"
ANALYZE="$SCRIPT_DIR/analyze_results.py"
AGGREGATE="$SCRIPT_DIR/aggregate_analysis.py"

# Check scripts exist
if [ ! -f "$ANALYZE" ]; then
  echo "ERROR: analyze_results.py not found at $ANALYZE"
  exit 1
fi

# Count run directories
RUN_DIRS=$(find "$SESSION_DIR" -maxdepth 1 -type d -name "run_*" | sort)
N_RUNS=$(echo "$RUN_DIRS" | grep -c "run_")

if [ "$N_RUNS" -eq 0 ]; then
  # No run_N subdirs -- single-run session, analyze the session dir directly
  echo "Single-run session detected."
  echo ""
  echo "--- Analyzing: $SESSION_DIR ---"
  if "$PYTHON" "$ANALYZE" "$SESSION_DIR"; then
    echo "[OK] Analysis complete."
  else
    echo "[FAIL] Analysis failed."
  fi
else
  # Multi-run session
  echo "Found $N_RUNS runs."
  echo ""

  PASS=0
  FAIL=0
  for RUN_DIR in $RUN_DIRS; do
    RUN_NAME=$(basename "$RUN_DIR")
    echo "--- Analyzing: $RUN_NAME ---"
    if "$PYTHON" "$ANALYZE" "$RUN_DIR"; then
      echo "[OK] $RUN_NAME done."
      PASS=$((PASS + 1))
    else
      echo "[FAIL] $RUN_NAME failed."
      FAIL=$((FAIL + 1))
    fi
    echo ""
  done

  echo "Per-run analysis: $PASS passed, $FAIL failed."

  # Aggregate analysis
  if [ -f "$AGGREGATE" ]; then
    echo ""
    echo "--- Aggregate analysis ---"
    if "$PYTHON" "$AGGREGATE" "$SESSION_DIR"; then
      echo "[OK] Aggregate analysis complete."
    else
      echo "[FAIL] Aggregate analysis failed."
    fi
  else
    echo "NOTE: aggregate_analysis.py not found, skipping aggregate step."
  fi
fi

echo ""
echo "=== Done ==="
echo "Results in: $SESSION_DIR"
