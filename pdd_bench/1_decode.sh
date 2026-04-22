#!/bin/bash
# ============================================================
#  Step 1: 啟動 Decode server
#  用法:  bash 1_decode.sh
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/decode_${TIMESTAMP}.log"

export LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:${LD_LIBRARY_PATH}
[ -n "$DECODE_GPUS" ] && export HIP_VISIBLE_DEVICES="$DECODE_GPUS"
export SGLANG_USE_AITER=1
export NCCL_IB_RETRY_CNT="$NCCL_IB_RETRY_CNT"
export NCCL_IB_TIMEOUT="$NCCL_IB_TIMEOUT"

if [ "$PDD_MODE" = "multi" ]; then
    export GLOO_SOCKET_IFNAME="$NODE_NIC"
    export NCCL_SOCKET_IFNAME="$NODE_NIC"
    export MORI_SOCKET_IFNAME="$NODE_NIC"
    export SGLANG_HOST_IP="${SGLANG_HOST_IP:-$DECODE_HOST}"
    [ -n "$MORI_RDMA_DEVICES" ] && export MORI_RDMA_DEVICES
    [ -n "$MORI_DISABLE_TOPO" ] && export MORI_DISABLE_TOPO
elif [ -n "$SGLANG_HOST_IP" ]; then
    export SGLANG_HOST_IP
fi
[ -n "$SGLANG_ROCM_FUSED_DECODE_MLA" ] && export SGLANG_ROCM_FUSED_DECODE_MLA="$SGLANG_ROCM_FUSED_DECODE_MLA"
[ -n "$ROCM_QUICK_REDUCE_QUANTIZATION" ] && export ROCM_QUICK_REDUCE_QUANTIZATION="$ROCM_QUICK_REDUCE_QUANTIZATION"
[ -n "$SAFETENSORS_FAST_GPU" ] && export SAFETENSORS_FAST_GPU="$SAFETENSORS_FAST_GPU"

echo "[decode] Mode:         $PDD_MODE"
echo "[decode] GPU:          ${DECODE_GPUS:-all}"
echo "[decode] IB:           $DECODE_IB_DEVS"
echo "[decode] Host:         $DECODE_HOST:$DECODE_PORT"
echo "[decode] NODE_NIC:     ${NODE_NIC:-<unset>}"
echo "[decode] HOST_IP:      ${SGLANG_HOST_IP:-<auto>}"
echo "[decode] GLOO_IF:      ${GLOO_SOCKET_IFNAME:-<unset>}"
echo "[decode] MORI_IF:      ${MORI_SOCKET_IFNAME:-<unset>}"
echo "[decode] MORI_RDMA:    ${MORI_RDMA_DEVICES:-<unset>}"
echo "[decode] MORI_NOTOPO:  ${MORI_DISABLE_TOPO:-<unset>}"
echo "[decode] Log:          $LOG_FILE"

# 組裝可選參數
OPTIONAL_ARGS=""
[ -n "$NSA_PREFILL_BACKEND" ] && OPTIONAL_ARGS="$OPTIONAL_ARGS --nsa-prefill-backend $NSA_PREFILL_BACKEND"
[ -n "$NSA_DECODE_BACKEND" ]  && OPTIONAL_ARGS="$OPTIONAL_ARGS --nsa-decode-backend $NSA_DECODE_BACKEND"
[ -n "$TOOL_CALL_PARSER" ]    && OPTIONAL_ARGS="$OPTIONAL_ARGS --tool-call-parser $TOOL_CALL_PARSER"
[ -n "$REASONING_PARSER" ]    && OPTIONAL_ARGS="$OPTIONAL_ARGS --reasoning-parser $REASONING_PARSER"
[ "$DISABLE_RADIX_CACHE" = "true" ] && OPTIONAL_ARGS="$OPTIONAL_ARGS --disable-radix-cache"
[ -n "$KV_CACHE_DTYPE" ]             && OPTIONAL_ARGS="$OPTIONAL_ARGS --kv-cache-dtype $KV_CACHE_DTYPE"
[ -n "$MODEL_LOADER_EXTRA_CONFIG" ]  && OPTIONAL_ARGS="$OPTIONAL_ARGS --model-loader-extra-config '$MODEL_LOADER_EXTRA_CONFIG'"

EXTRA_ARGS=""
if [ "${1:-}" = "--no-cuda-graph" ]; then
    EXTRA_ARGS="--disable-cuda-graph"
    echo "[decode] *** CUDA Graph DISABLED (profile mode) ***"
fi

echo "[decode] Model: $MODEL_NAME"
echo "[decode] Transfer backend: $DISAGG_TRANSFER_BACKEND"
echo "[decode] Launching decode server..."
python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend "$DISAGG_TRANSFER_BACKEND" \
    --disaggregation-ib-device "$DECODE_IB_DEVS" \
    --host "$DECODE_HOST" \
    --port "$DECODE_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --trust-remote-code \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --num-continuous-decode-steps "$NUM_CONTINUOUS_DECODE_STEPS" \
    --context-length "$CONTEXT_LENGTH" \
    $OPTIONAL_ARGS \
    $EXTRA_ARGS \
    2>&1 | tee "$LOG_FILE"
