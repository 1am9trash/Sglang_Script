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

# -------------------- 模型 --------------------
# "glm5" 或 "deepseek"，切換後所有參數自動對應
MODEL_NAME="glm5"

# -------------------- 網路 --------------------
PREFILL_PORT=30025
DECODE_PORT=30026
ROUTER_PORT=30028

# ============================================================
#  依 PDD_MODE 自動設定的參數
# ============================================================
ALL_IB_DEVS=$(ibv_devinfo 2>/dev/null | awk '/hca_id:/{print $2}' | paste -sd, -)
PREFILL_IB_DEVS="$ALL_IB_DEVS"
DECODE_IB_DEVS="$ALL_IB_DEVS"

if [ "$PDD_MODE" = "single" ]; then
    PREFILL_HOST="127.0.0.1"
    DECODE_HOST="127.0.0.1"
    PREFILL_GPUS="0,1,2,3"
    DECODE_GPUS="4,5,6,7"
    TP_SIZE=4
else
    PREFILL_HOST="10.235.58.248"       # ← 改成 prefill 機器的 IP
    DECODE_HOST="10.235.58.247"        # ← 改成 decode 機器的 IP
    PREFILL_GPUS=""
    DECODE_GPUS=""
    TP_SIZE=8
fi

# ============================================================
#  依 MODEL_NAME 自動設定的參數
# ============================================================
NSA_PREFILL_BACKEND="tilelang"
NSA_DECODE_BACKEND="tilelang"
KV_CACHE_DTYPE="fp8_e4m3"
MEM_FRACTION_STATIC=0.85
DISABLE_RADIX_CACHE=true
MAX_PREFILL_TOKENS=131072
CHUNKED_PREFILL_SIZE=131072
CUDA_GRAPH_MAX_BS=128
MAX_RUNNING_REQUESTS=128
CONTEXT_LENGTH=131072
NUM_CONTINUOUS_DECODE_STEPS=4

if [ "$MODEL_NAME" = "glm5" ]; then
    MODEL_PATH="/data/models/GLM-5-fp8/"
    SGLANG_ROCM_FUSED_DECODE_MLA=0
    ROCM_QUICK_REDUCE_QUANTIZATION="INT4"
    SAFETENSORS_FAST_GPU=1
    TOOL_CALL_PARSER="glm47"
    REASONING_PARSER="glm45"
    MODEL_LOADER_EXTRA_CONFIG='{"enable_multithread_load": true, "num_threads": 8}'
elif [ "$MODEL_NAME" = "deepseek" ]; then
    MODEL_PATH="/data/models/DeepSeek-V3.2/"
fi

# -------------------- NCCL --------------------
NCCL_IB_RETRY_CNT=15
NCCL_IB_TIMEOUT=22

# -------------------- Health Check --------------------
HEALTH_CHECK_MAX_WAIT=240          # × 5 秒 = 20 分鐘

# -------------------- Benchmark --------------------
BENCH_INPUT_LENS=(1024)
BENCH_OUTPUT_LENS=(1024)
BENCH_CONCURRENCIES=(1 2 4 8)
BENCH_DURATION=8
BENCH_REQUEST_RATE="INF"
BENCH_OUTPUT_DIR="./bench_results"

# -------------------- Accuracy --------------------
ACC_NUM_EXAMPLES=200

# -------------------- 日誌 --------------------
LOG_DIR="./logs"
