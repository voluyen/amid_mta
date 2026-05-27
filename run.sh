#! /bin/bash
set -eo pipefail

BASE_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="amid"
LOG_DIR="${BASE_PATH}/run_logs"
mkdir -p "${LOG_DIR}"

# ============================================================
# 1. Setup conda environment
# ============================================================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Setting up conda environment '${ENV_NAME}'..."

# Load conda into current shell
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -qE "^${ENV_NAME}[[:space:]]"; then
    echo "[INFO] Conda env '${ENV_NAME}' already exists, skipping creation."
else
    echo "[INFO] Creating conda env '${ENV_NAME}' from environment.yml..."
    conda env create -f "${BASE_PATH}/environment.yml"
fi

conda activate "${ENV_NAME}"
echo "[INFO] Activated conda env: $(conda info --envs | grep '*' | awk '{print $1}')"

# ============================================================
# 2. Install Python dependencies
# ============================================================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Installing Python dependencies..."

bash "${BASE_PATH}/install.sh"

# install.sh chạy 'uv sync' → tạo .venv riêng, cần activate để dùng đúng packages
source "${BASE_PATH}/.venv/bin/activate"

# spaCy + English model (required by span_finetune.py)
pip install --quiet spacy
python -m spacy download en_core_web_sm

# Đảm bảo NCCL không verbose (install.sh set trong subshell, không truyền lên)
export NCCL_DEBUG=""

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All dependencies installed."

# ============================================================
# 3. Run training experiments
# ============================================================
run_exp() {
    local script_path="${BASE_PATH}/$1"
    local name="$2"
    echo ""
    echo "========================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: ${name}"
    echo "========================================================"

    bash "${script_path}" 2>&1 | tee "${LOG_DIR}/${name}.log"
    local exit_code=${PIPESTATUS[0]}

    if [ ${exit_code} -ne 0 ]; then
        echo "========================================================"
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED: ${name} (exit code: ${exit_code})"
        echo "Log saved to: ${LOG_DIR}/${name}.log"
        echo "========================================================"
        exit ${exit_code}
    fi

    echo "========================================================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done: ${name}"
    echo "========================================================"
}

# 1. GPT2-base (student) → GPT2-XLarge (teacher)
run_exp "scripts/amid_1gpu/train_gpt2_base_mta.sh" "gpt2_base_mta"

# 2. Qwen1.5-0.5B (student) → Qwen1.5-1.8B (teacher)
run_exp "scripts/amid_1gpu/train_qwen_0.5B_mta.sh" "qwen_0.5B_mta"

# 3. OPT-1.3B (student) → OPT-6.7B (teacher)
run_exp "scripts/amid_1gpu/train_opt_1.3b_mta.sh" "opt_1.3b_mta"

echo ""
echo "========================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All experiments completed successfully."
echo "Logs saved to: ${LOG_DIR}/"
echo "========================================================"
