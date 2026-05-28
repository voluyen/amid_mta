#! /bin/bash
set -eo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
# 2. Wave 1: 3 main training experiments in PARALLEL
#    GPU 0 → GPT2-120M       (teacher: GPT2-1.5B)
#    GPU 0 → Qwen1.5-0.5B    (teacher: Qwen1.5-1.8B)
#    GPU 1 → OPT-1.3B        (teacher: OPT-6.7B)
# ============================================================

echo ""
echo "========================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Wave 1: launching 3 main experiments in parallel..."
echo "========================================================"

bash "${BASE_PATH}/scripts/amid_1gpu/train_gpt2_base_mta.sh" \
    > "${LOG_DIR}/gpt2_base_mta.log" 2>&1 &
PID_GPT2=$!

bash "${BASE_PATH}/scripts/amid_1gpu/train_qwen_0.5B_mta.sh" \
    > "${LOG_DIR}/qwen_0.5B_mta.log" 2>&1 &
PID_QWEN=$!

bash "${BASE_PATH}/scripts/amid_1gpu/train_opt_1.3b_mta.sh" \
    > "${LOG_DIR}/opt_1.3b_mta.log" 2>&1 &
PID_OPT=$!

echo "[INFO] PIDs: gpt2=$PID_GPT2  qwen=$PID_QWEN  opt=$PID_OPT"
echo "[INFO] Logs: ${LOG_DIR}/"
echo ""

# ── Wait for each job and collect exit codes ──────────────────
FAILED=0

set +e
wait $PID_GPT2; CODE_GPT2=$?
wait $PID_QWEN; CODE_QWEN=$?
wait $PID_OPT;  CODE_OPT=$?
set -e

[ $CODE_GPT2 -ne 0 ] && { echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: gpt2_base_mta (exit=$CODE_GPT2)"; FAILED=1; } || echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:   gpt2_base_mta"
[ $CODE_QWEN -ne 0 ] && { echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: qwen_0.5B_mta  (exit=$CODE_QWEN)";  FAILED=1; } || echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:   qwen_0.5B_mta"
[ $CODE_OPT  -ne 0 ] && { echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: opt_1.3b_mta   (exit=$CODE_OPT)";   FAILED=1; } || echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:   opt_1.3b_mta"

# ============================================================
# 3. Wave 2: 3 ablation experiments in PARALLEL (on gpt2-base)
#    GPU 0 → word_level    (all 3 layers word-level)
#    GPU 1 → phrase_level  (all 3 layers phrase-level)
#    GPU 2 → wo_weight     (uniform mean pooling, no token weights)
# ============================================================
echo ""
echo "========================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Wave 2: launching 3 ablation experiments in parallel..."
echo "========================================================"

bash "${BASE_PATH}/scripts/amid_1gpu/ablation_word_level.sh" \
    > "${LOG_DIR}/ablation_word_level.log" 2>&1 &
PID_AB_W=$!

bash "${BASE_PATH}/scripts/amid_1gpu/ablation_phrase_level.sh" \
    > "${LOG_DIR}/ablation_phrase_level.log" 2>&1 &
PID_AB_P=$!

bash "${BASE_PATH}/scripts/amid_1gpu/ablation_wo_weight.sh" \
    > "${LOG_DIR}/ablation_wo_weight.log" 2>&1 &
PID_AB_WO=$!

echo "[INFO] PIDs: word=$PID_AB_W  phrase=$PID_AB_P  wo_weight=$PID_AB_WO"
echo ""

set +e
wait $PID_AB_W;  CODE_AB_W=$?
wait $PID_AB_P;  CODE_AB_P=$?
wait $PID_AB_WO; CODE_AB_WO=$?
set -e

[ $CODE_AB_W  -ne 0 ] && { echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: ablation_word_level    (exit=$CODE_AB_W)";  FAILED=1; } || echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:   ablation_word_level"
[ $CODE_AB_P  -ne 0 ] && { echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: ablation_phrase_level  (exit=$CODE_AB_P)";  FAILED=1; } || echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:   ablation_phrase_level"
[ $CODE_AB_WO -ne 0 ] && { echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: ablation_wo_weight     (exit=$CODE_AB_WO)"; FAILED=1; } || echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done:   ablation_wo_weight"

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
