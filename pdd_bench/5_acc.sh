#!/bin/bash
# ============================================================
#  Step 5: 精確度測試（sanity check + GSM8K eval）
#  在 container 內執行。需在 router 就緒後執行。
#  用法:  bash 5_acc.sh
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

export PYTHONPATH="${SGLANG_PYTHON_PATH}:${PYTHONPATH}"

ROUTER_URL="http://${PREFILL_HOST}:${ROUTER_PORT}"
PREFILL_URL="http://${PREFILL_HOST}:${PREFILL_PORT}"

echo "=============================================="
echo "  Accuracy Tests"
echo "  Router:  $ROUTER_URL"
echo "  Prefill: $PREFILL_URL"
echo "=============================================="
echo ""

# --- Test 1: 直接對 prefill server 做 sanity check（不經過 PD 路徑）---
echo "=== [1/3] Direct prefill test (no KV transfer) ==="
curl -s --max-time 60 "http://${PREFILL_HOST}:${PREFILL_PORT}/generate" \
    -d '{"text":"What is 2+3?","sampling_params":{"max_new_tokens":32}}' \
    -H "Content-Type: application/json"
echo ""
echo ""

# --- Test 2: 透過 router 走完整 PD 路徑 ---
echo "=== [2/3] Router PD test (full prefill→decode path) ==="
curl -s --max-time 120 "${ROUTER_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"What is 2+3?"}],"max_tokens":32}'
echo ""
echo ""

# --- Test 3: GSM8K 精確度評估 ---
echo "=== [3/3] ${ACC_EVAL_NAME} accuracy eval (n=${ACC_NUM_EXAMPLES}) ==="
python3 -m sglang.test.run_eval \
    --port "$ROUTER_PORT" \
    --eval-name "$ACC_EVAL_NAME" \
    --num-examples "$ACC_NUM_EXAMPLES"
echo ""
echo "=== Accuracy tests complete ==="
