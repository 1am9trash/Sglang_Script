#!/bin/bash
# ============================================================
#  Step 2: 啟動 Prefill server
#  用法:  bash 2_prefill.sh
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/prefill_${TIMESTAMP}.log"

export LD_LIBRARY_PATH=/opt/rocm/lib:/usr/local/lib:${LD_LIBRARY_PATH}
[ -n "$PREFILL_GPUS" ] && export HIP_VISIBLE_DEVICES="$PREFILL_GPUS"
export SGLANG_USE_AITER=1
export NCCL_IB_RETRY_CNT="$NCCL_IB_RETRY_CNT"
export NCCL_IB_TIMEOUT="$NCCL_IB_TIMEOUT"

echo "[prefill] Mode: $PDD_MODE"
echo "[prefill] GPU:  ${PREFILL_GPUS:-all}"
echo "[prefill] IB:   $PREFILL_IB_DEVS"
echo "[prefill] Host: $PREFILL_HOST:$PREFILL_PORT"
echo "[prefill] Log:  $LOG_FILE"

# 等待 decode server 就緒
echo "[prefill] Waiting for decode server at ${DECODE_HOST}:${DECODE_PORT} ..."
WAIT=0
while true; do
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://${DECODE_HOST}:${DECODE_PORT}/health" 2>/dev/null || echo "000")
    [ "$STATUS" = "200" ] && break
    sleep 5
    WAIT=$((WAIT + 1))
    if [ $WAIT -ge $HEALTH_CHECK_MAX_WAIT ]; then
        echo "[prefill] ERROR: Decode server not ready after $((WAIT * 5))s. Start 1_decode.sh first."
        exit 1
    fi
    [ $((WAIT % 12)) -eq 0 ] && echo "[prefill]   ... still waiting ($((WAIT * 5))s elapsed, HTTP $STATUS)"
done
echo "[prefill] Decode server is ready!"

# aiter kernel warmup
MAX_JOBS=192 python3 /sgl-workspace/aiter/op_tests/test_rmsnorm2d.py

# 組裝可選參數
OPTIONAL_ARGS=""
[ -n "$NSA_PREFILL_BACKEND" ] && OPTIONAL_ARGS="$OPTIONAL_ARGS --nsa-prefill-backend $NSA_PREFILL_BACKEND"
[ -n "$NSA_DECODE_BACKEND" ]  && OPTIONAL_ARGS="$OPTIONAL_ARGS --nsa-decode-backend $NSA_DECODE_BACKEND"
[ -n "$TOOL_CALL_PARSER" ]    && OPTIONAL_ARGS="$OPTIONAL_ARGS --tool-call-parser $TOOL_CALL_PARSER"
[ -n "$REASONING_PARSER" ]    && OPTIONAL_ARGS="$OPTIONAL_ARGS --reasoning-parser $REASONING_PARSER"
[ "$DISABLE_RADIX_CACHE" = "true" ] && OPTIONAL_ARGS="$OPTIONAL_ARGS --disable-radix-cache"

echo "[prefill] Launching prefill server..."
python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --disaggregation-mode prefill \
    --disaggregation-ib-device "$PREFILL_IB_DEVS" \
    --host "$PREFILL_HOST" \
    --port "$PREFILL_PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --trust-remote-code \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --max-prefill-tokens "$MAX_PREFILL_TOKENS" \
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS" \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --num-continuous-decode-steps "$NUM_CONTINUOUS_DECODE_STEPS" \
    --context-length "$CONTEXT_LENGTH" \
    --model-loader-extra-config "$MODEL_LOADER_EXTRA_CONFIG" \
    $OPTIONAL_ARGS \
    2>&1 | tee "$LOG_FILE"
