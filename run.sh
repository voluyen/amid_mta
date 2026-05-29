#! /bin/bash
set -eo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${BASE_PATH}"   # ensure sub-scripts with BASE_PATH=. resolve correctly
LOG_DIR="${BASE_PATH}/run_logs"
mkdir -p "${LOG_DIR}"

# ============================================================
# 1. Install Python dependencies (uv sync → creates .venv)
# ============================================================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing Python dependencies..."

bash "${BASE_PATH}/install.sh"

# Activate the uv-managed venv created by install.sh
source "${BASE_PATH}/.venv/bin/activate"
echo "[INFO] Activated venv: ${BASE_PATH}/.venv (python=$(which python))"

# Download spaCy English model
python -m spacy download en_core_web_sm

# Ensure NCCL not verbose
export NCCL_DEBUG=""

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All dependencies installed."

# ============================================================
# Run train_qwen_0.5B and train_qwen_0.5B_mta in parallel.
# ============================================================

FAILED=0

run_wave () {
    local wave_name="$1"; shift
    local -a names=()
    local -a pids=()

    echo ""
    echo "========================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${wave_name}: launching $# job(s)..."
    echo "========================================================"

    # Args alternate: name script name script ...
    while [ $# -gt 0 ]; do
        local name="$1"
        local script="$2"
        shift 2
        bash "${BASE_PATH}/${script}" > "${LOG_DIR}/${name}.log" 2>&1 &
        local pid=$!
        names+=("$name")
        pids+=("$pid")
        echo "[INFO] ${name} → PID=${pid}  log=${LOG_DIR}/${name}.log"
    done

    set +e
    local i
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}"
        local code=$?
        if [ "$code" -ne 0 ]; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: ${names[$i]} (exit=$code) — see ${LOG_DIR}/${names[$i]}.log"
            FAILED=1
        else
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:   ${names[$i]}"
        fi
    done
    set -e
}

# Fail-fast: abort the run if any wave reported a failure.
check_wave () {
    if [ $FAILED -ne 0 ]; then
        echo ""
        echo "========================================================"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ABORTING: a job in the previous wave failed."
        echo "Check logs in: ${LOG_DIR}/"
        echo "========================================================"
        exit 1
    fi
}

# ── Wave 1: 2 script Qwen 0.5B chạy song song ────────────────
run_wave "Wave 1 (Qwen 0.5B baseline + MTA)" \
    "train_qwen_0.5B"      "scripts/amid_1gpu/train_qwen_0.5B.sh" \
    "train_qwen_0.5B_mta"  "scripts/amid_1gpu/train_qwen_0.5B_mta.sh"
check_wave

echo ""
echo "========================================================"
if [ $FAILED -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] All experiments completed successfully."
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] One or more experiments FAILED. Check logs in: ${LOG_DIR}/"
    exit 1
fi
echo "Logs saved to: ${LOG_DIR}/"
echo "========================================================"
