#!/bin/bash
# ============================================================
#  PDD (Prefill-Decode Disaggregation) — 集中設定檔
#  所有腳本皆 source 此檔，修改參數只需改這裡
# ============================================================

# ==================== 部署模式 ====================
# "single" = 單機拆兩組 GPU（1P1D on 1 node）
# "multi"  = 雙機各用全部 GPU（1P1D on 2 nodes）
PDD_MODE="multi"

# ==================== KV Transfer Backend ====================
# "mooncake" = Mooncake transfer engine (default)
# "mori"     = MoRI-IO transfer engine
DISAGG_TRANSFER_BACKEND="mori"

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
if [ -z "$ALL_IB_DEVS" ]; then
    ALL_IB_DEVS=$(ls /sys/class/infiniband/ 2>/dev/null | paste -sd, -)
fi
PREFILL_IB_DEVS="$ALL_IB_DEVS"
DECODE_IB_DEVS="$ALL_IB_DEVS"

if [ "$PDD_MODE" = "single" ]; then
    PREFILL_HOST="127.0.0.1"
    DECODE_HOST="127.0.0.1"
    ROUTER_HOST="127.0.0.1"
    PREFILL_GPUS="0,1,2,3"
    DECODE_GPUS="4,5,6,7"
    TP_SIZE=4
    # Leave empty to let SGLang auto-detect the local host IP for MORI.
    # Set this explicitly (for example, a NIC IP) if auto-detection is wrong.
    SGLANG_HOST_IP=""
    NODE_NIC=""
else
    PREFILL_HOST="10.235.58.246"        # ← 改成 prefill 機器的 IP (必須在 NODE_NIC 上)
    DECODE_HOST="10.235.58.247"         # ← 改成 decode 機器的 IP (必須在 NODE_NIC 上)
    ROUTER_HOST="$DECODE_HOST"          # ← router 跑在哪台就填哪台
    PREFILL_GPUS=""
    DECODE_GPUS=""
    TP_SIZE=8
    # 數據面 NIC (GLOO/NCCL/MORI socket + SGLANG_HOST_IP auto-derive)
    # 兩台機器若是同型號，NIC 名通常相同; 若不同，可在該機器上 export NODE_NIC=xxx 覆蓋
    NODE_NIC="${NODE_NIC:-enp196s0}"
    SGLANG_HOST_IP=""                   # 由 launcher 依 role 自動填成本機 IP
    # MORI-IO RDMA 設備清單 (兩台機器必須完全一致，含順序)
    # 可列多張聚合頻寬 (例如 ionic_0,ionic_1,...,ionic_8)，目前只用一張做功能驗證
    MORI_RDMA_DEVICES="${MORI_RDMA_DEVICES:-ionic_0}"
    MORI_DISABLE_TOPO="${MORI_DISABLE_TOPO:-1}"
fi

# ============================================================
#  依 MODEL_NAME 自動設定的參數
# ============================================================
NSA_PREFILL_BACKEND="tilelang"
NSA_DECODE_BACKEND="tilelang"
KV_CACHE_DTYPE=""
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
    KV_CACHE_DTYPE="fp8_e4m3"
    SGLANG_ROCM_FUSED_DECODE_MLA=0
    ROCM_QUICK_REDUCE_QUANTIZATION="INT4"
    SAFETENSORS_FAST_GPU=1
    TOOL_CALL_PARSER="glm47"
    REASONING_PARSER="glm45"
elif [ "$MODEL_NAME" = "glm5_fp4" ]; then
    MODEL_PATH="/data/models/GLM-5-MXFP4/"
    KV_CACHE_DTYPE="fp8_e4m3"
    SGLANG_ROCM_FUSED_DECODE_MLA=0
    ROCM_QUICK_REDUCE_QUANTIZATION="INT4"
    SAFETENSORS_FAST_GPU=1
    TOOL_CALL_PARSER="glm47"
    REASONING_PARSER="glm45"
elif [ "$MODEL_NAME" = "deepseek" ]; then
    MODEL_PATH="/data/models/DeepSeek-V3.2/"
fi

# -------------------- NCCL --------------------
NCCL_IB_RETRY_CNT=15
NCCL_IB_TIMEOUT=22

# -------------------- Health Check --------------------
HEALTH_CHECK_MAX_WAIT=240          # × 5 秒 = 20 分鐘

# -------------------- Benchmark --------------------
BENCH_INPUT_LENS=(1024 8192)
BENCH_OUTPUT_LENS=(1024 1024)
BENCH_CONCURRENCIES=(4 8 16 32 64)
BENCH_PROMPT_MULTIPLIER=10
BENCH_RANDOM_RANGE_RATIO=0.8
BENCH_OUTPUT_DIR="./bench_results"

# -------------------- Accuracy --------------------
ACC_NUM_EXAMPLES=1200
ACC_PARALLEL=200

# -------------------- 日誌 --------------------
LOG_DIR="./logs"
