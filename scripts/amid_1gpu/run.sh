#! /bin/bash

# Run all 3 MTA experiments sequentially
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/run_logs"
mkdir -p ${LOG_DIR}

run_exp() {
    local script=$1
    local name=$2
    echo "========================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${name}"
    echo "========================================================"

    bash "${SCRIPT_DIR}/${script}" 2>&1 | tee "${LOG_DIR}/${name}.log"
    local exit_code=${PIPESTATUS[0]}

    if [ ${exit_code} -ne 0 ]; then
        echo "========================================================"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: ${name} (exit code: ${exit_code})"
        echo "========================================================"
        exit ${exit_code}
    fi

    echo "========================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done: ${name}"
    echo "========================================================"
}

# 1. GPT2-base (student) → GPT2-XLarge (teacher)
run_exp "train_gpt2_base_mta.sh" "gpt2_base_mta"

# 2. Qwen1.5-0.5B (student) → Qwen1.5-1.8B (teacher)
run_exp "train_qwen_0.5B_mta.sh" "qwen_0.5B_mta"

# 3. OPT-1.3B (student) → OPT-6.7B (teacher)
run_exp "train_opt_1.3b_mta.sh" "opt_1.3b_mta"

echo "========================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All experiments completed successfully."
echo "========================================================"
