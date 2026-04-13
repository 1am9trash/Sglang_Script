#!/bin/bash
# ============================================================
#  Step 3: 啟動 Router (load balancer)
#  在 container 內執行。需在 prefill + decode 都就緒後啟動。
#  用法:  bash 3_router.sh
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/router_${TIMESTAMP}.log"

PREFILL_URL="http://${PREFILL_HOST}:${PREFILL_PORT}"
DECODE_URL="http://${DECODE_HOST}:${DECODE_PORT}"

# 同時等待 prefill + decode 都就緒
echo "[router] Waiting for both servers to be ready ..."
echo "[router]   Prefill: ${PREFILL_URL}"
echo "[router]   Decode:  ${DECODE_URL}"
PREFILL_READY=false
DECODE_READY=false
WAIT=0
while [ "$PREFILL_READY" = false ] || [ "$DECODE_READY" = false ]; do
    if [ "$PREFILL_READY" = false ]; then
        STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${PREFILL_URL}/health" 2>/dev/null || echo "000")
        [ "$STATUS" = "200" ] && PREFILL_READY=true && echo "[router] Prefill server is ready!"
    fi
    if [ "$DECODE_READY" = false ]; then
        STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${DECODE_URL}/health" 2>/dev/null || echo "000")
        [ "$STATUS" = "200" ] && DECODE_READY=true && echo "[router] Decode server is ready!"
    fi
    if [ "$PREFILL_READY" = false ] || [ "$DECODE_READY" = false ]; then
        sleep 5
        WAIT=$((WAIT + 1))
        if [ $WAIT -ge $HEALTH_CHECK_MAX_WAIT ]; then
            echo "[router] ERROR: Timeout after $((WAIT * 5))s. Prefill=$PREFILL_READY, Decode=$DECODE_READY"
            exit 1
        fi
        [ $((WAIT % 12)) -eq 0 ] && echo "[router]   ... waiting ($((WAIT * 5))s) Prefill=$PREFILL_READY Decode=$DECODE_READY"
    fi
done

echo "[router] Launching router on port $ROUTER_PORT ..."
echo "[router] Prefill: $PREFILL_URL"
echo "[router] Decode:  $DECODE_URL"
echo "[router] Log:     $LOG_FILE"

python3 -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill "$PREFILL_URL" \
    --decode "$DECODE_URL" \
    --host 0.0.0.0 \
    --port "$ROUTER_PORT" \
    2>&1 | tee "$LOG_FILE"
