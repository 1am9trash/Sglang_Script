#!/bin/bash
# ============================================================
#  Step 1: 啟動 Decode server
#  在 container 內執行。需先於 prefill / router 之前啟動。
#  用法:  bash 1_decode.sh
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/decode_${TIMESTAMP}.log"

export PYTHONPATH="${SGLANG_PYTHON_PATH}:${PYTHONPATH}"
export LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:${LD_LIBRARY_PATH}
[ -n "$DECODE_GPUS" ] && export HIP_VISIBLE_DEVICES="$DECODE_GPUS"
export SGLANG_USE_AITER=1
export SGLANG_ROCM_FUSED_DECODE_MLA="$SGLANG_ROCM_FUSED_DECODE_MLA"
export ROCM_QUICK_REDUCE_QUANTIZATION="$ROCM_QUICK_REDUCE_QUANTIZATION"
export SAFETENSORS_FAST_GPU="$SAFETENSORS_FAST_GPU"
export NCCL_IB_RETRY_CNT="$NCCL_IB_RETRY_CNT"
export NCCL_IB_TIMEOUT="$NCCL_IB_TIMEOUT"

echo "[decode] Mode: $PDD_MODE"
echo "[decode] GPU:  ${DECODE_GPUS:-all}"
echo "[decode] IB:   $DECODE_IB_DEVS"
echo "[decode] Host: $DECODE_HOST:$DECODE_PORT"
echo "[decode] Log:  $LOG_FILE"

echo "[decode] Launching decode server..."

# 組裝可選參數
OPTIONAL_ARGS=""
[ -n "$NSA_PREFILL_BACKEND" ] && OPTIONAL_ARGS="$OPTIONAL_ARGS --nsa-prefill-backend $NSA_PREFILL_BACKEND"
[ -n "$NSA_DECODE_BACKEND" ]  && OPTIONAL_ARGS="$OPTIONAL_ARGS --nsa-decode-backend $NSA_DECODE_BACKEND"
[ -n "$TOOL_CALL_PARSER" ]    && OPTIONAL_ARGS="$OPTIONAL_ARGS --tool-call-parser $TOOL_CALL_PARSER"
[ -n "$REASONING_PARSER" ]    && OPTIONAL_ARGS="$OPTIONAL_ARGS --reasoning-parser $REASONING_PARSER"
[ "$DISABLE_RADIX_CACHE" = "true" ] && OPTIONAL_ARGS="$OPTIONAL_ARGS --disable-radix-cache"

python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --disaggregation-mode decode \
    --disaggregation-ib-device "$DECODE_IB_DEVS" \
    --host "$DECODE_HOST" \
    --port "$DECODE_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --trust-remote-code \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --context-length "$CONTEXT_LENGTH" \
    --model-loader-extra-config "$MODEL_LOADER_EXTRA_CONFIG" \
    $OPTIONAL_ARGS \
    2>&1 | tee "$LOG_FILE"
