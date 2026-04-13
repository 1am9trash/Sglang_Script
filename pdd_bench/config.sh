#!/bin/bash
# ============================================================
#  PDD (Prefill-Decode Disaggregation) — 集中設定檔
#  所有腳本皆 source 此檔，修改參數只需改這裡
# ============================================================

# ==================== 部署模式 ====================
# "single" = 單機拆兩組 GPU（1P1D on 1 node）
# "multi"  = 雙機各用全部 GPU（1P1D on 2 nodes）
PDD_MODE="single"

# -------------------- Docker --------------------
DOCKER_IMAGE="rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260412"
CONTAINER_NAME="thomas_pdd_bench"
DOCKER_SHM_SIZE="32g"
DOCKER_VOLUMES=(
    "$HOME/thomas:/thomas"
    "/data:/data"
)
HOST_IB_LIB="/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so"
CONTAINER_IB_LIB="/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so"

# -------------------- 模型 --------------------
MODEL_PATH="/data/GLM-5-FP8/"
# MODEL_PATH="/models/DeepSeek-V3.2-Exp/"

# -------------------- SGLang 原始碼 --------------------
SGLANG_PYTHON_PATH=""

# -------------------- 網路 --------------------
PREFILL_PORT=30025
DECODE_PORT=30026
ROUTER_PORT=30028

# ============================================================
#  依 PDD_MODE 自動設定的參數
# ============================================================
if [ "$PDD_MODE" = "single" ]; then
    # --- Single-node: 一台機器，GPU 拆兩組 ---
    PREFILL_HOST="127.0.0.1"
    DECODE_HOST="127.0.0.1"
    PREFILL_GPUS="0,1,2,3"
    DECODE_GPUS="4,5,6,7"
    TP_SIZE=4
    PREFILL_IB_DEVS="ionic_0,ionic_1,ionic_2,ionic_3"
    DECODE_IB_DEVS="ionic_4,ionic_5,ionic_6,ionic_7"
else
    # --- Multi-node: 兩台機器，各用全部 GPU ---
    PREFILL_HOST="10.235.58.248"       # ← 改成 prefill 機器的 IP
    DECODE_HOST="10.235.58.247"        # ← 改成 decode 機器的 IP
    PREFILL_GPUS=""                    # 不限制，用全部 GPU
    DECODE_GPUS=""
    TP_SIZE=8
    PREFILL_IB_DEVS="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"
    DECODE_IB_DEVS="ionic_0,ionic_1,ionic_2,ionic_3,ionic_4,ionic_5,ionic_6,ionic_7"
fi

# -------------------- Server 共用參數 --------------------
KV_CACHE_DTYPE="fp8_e4m3"
MEM_FRACTION_STATIC=0.85
MAX_PREFILL_TOKENS=131072
CUDA_GRAPH_MAX_BS=128
MAX_RUNNING_REQUESTS=128
CONTEXT_LENGTH=131072
DISABLE_RADIX_CACHE=true
MODEL_LOADER_EXTRA_CONFIG='{"enable_multithread_load": true, "num_threads": 8}'

# GLM-5 專用參數（DeepSeek 不需要可註解掉）
NSA_PREFILL_BACKEND="tilelang"
NSA_DECODE_BACKEND="tilelang"
TOOL_CALL_PARSER="glm47"
REASONING_PARSER="glm45"

# -------------------- ROCm 環境變數 --------------------
SGLANG_ROCM_FUSED_DECODE_MLA=0
ROCM_QUICK_REDUCE_QUANTIZATION="INT4"
SAFETENSORS_FAST_GPU=1

# -------------------- NCCL --------------------
NCCL_IB_RETRY_CNT=15
NCCL_IB_TIMEOUT=22

# -------------------- Benchmark --------------------
BENCH_INPUT_LENS=(1024)
BENCH_OUTPUT_LENS=(1024)
BENCH_CONCURRENCIES=(1 2 4 8)
BENCH_DURATION=8                    # num_prompts = concurrency × duration
BENCH_REQUEST_RATE="INF"
BENCH_OUTPUT_DIR="./bench_results"

# -------------------- Accuracy --------------------
ACC_EVAL_NAME="gsm8k"
ACC_NUM_EXAMPLES=2000

# -------------------- 日誌 --------------------
LOG_DIR="./logs"
