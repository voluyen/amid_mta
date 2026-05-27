#! /bin/bash

# Run all 3 MTA experiments in PARALLEL
# GPU 0 → GPT2-120M  (teacher: GPT2-1.5B)   — Full fine-tuning
# GPU 1 → Qwen1.5-0.5B (teacher: Qwen1.5-1.8B) — Full fine-tuning
# GPU 2 → OPT-1.3B   (teacher: OPT-6.7B)    — LoRA (rank=256, alpha=8)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/run_logs"
mkdir -p "${LOG_DIR}"

echo "========================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching 3 experiments in parallel..."
echo "  GPU 0 → gpt2_base_mta"
echo "  GPU 1 → qwen_0.5B_mta"
echo "  GPU 2 → opt_1.3b_mta"
echo "========================================================"

# Launch all 3 in background, each writing to its own log file
bash "${SCRIPT_DIR}/train_gpt2_base_mta.sh" \
    > "${LOG_DIR}/gpt2_base_mta.log" 2>&1 &
PID_GPT2=$!

bash "${SCRIPT_DIR}/train_qwen_0.5B_mta.sh" \
    > "${LOG_DIR}/qwen_0.5B_mta.log" 2>&1 &
PID_QWEN=$!

bash "${SCRIPT_DIR}/train_opt_1.3b_mta.sh" \
    > "${LOG_DIR}/opt_1.3b_mta.log" 2>&1 &
PID_OPT=$!

echo "[INFO] PIDs: gpt2=$PID_GPT2  qwen=$PID_QWEN  opt=$PID_OPT"
echo "[INFO] Logs: ${LOG_DIR}/"
echo ""

# ── Wait for each job and collect exit codes ──────────────────
FAILED=0

wait $PID_GPT2; CODE_GPT2=$?
if [ $CODE_GPT2 -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: gpt2_base_mta (exit=$CODE_GPT2) — see ${LOG_DIR}/gpt2_base_mta.log"
    FAILED=1
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:   gpt2_base_mta"
fi

wait $PID_QWEN; CODE_QWEN=$?
if [ $CODE_QWEN -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: qwen_0.5B_mta  (exit=$CODE_QWEN) — see ${LOG_DIR}/qwen_0.5B_mta.log"
    FAILED=1
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:   qwen_0.5B_mta"
fi

wait $PID_OPT; CODE_OPT=$?
if [ $CODE_OPT -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: opt_1.3b_mta   (exit=$CODE_OPT)  — see ${LOG_DIR}/opt_1.3b_mta.log"
    FAILED=1
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:   opt_1.3b_mta"
fi

echo ""
echo "========================================================"
if [ $FAILED -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] All 3 experiments completed successfully."
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] One or more experiments FAILED. Check logs in: ${LOG_DIR}/"
    exit 1
fi
echo "Logs saved to: ${LOG_DIR}/"
echo "========================================================"
