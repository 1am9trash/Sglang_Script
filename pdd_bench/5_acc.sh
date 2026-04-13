#!/bin/bash
# ============================================================
#  Step 5: 精確度測試（sanity check + GSM8K eval）
#  在 container 內執行。需在 router 就緒後執行。
#  用法:  bash 5_acc.sh
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

ROUTER_URL="http://${PREFILL_HOST}:${ROUTER_PORT}"
PREFILL_URL="http://${PREFILL_HOST}:${PREFILL_PORT}"

echo "=============================================="
echo "  Accuracy Tests"
echo "  Router:  $ROUTER_URL"
echo "  Prefill: $PREFILL_URL"
echo "=============================================="
echo ""

# --- Test 1: 透過 router 走完整 PD 路徑 ---
echo "=== [1/2] Router PD test (full prefill→decode path) ==="
curl -s --max-time 120 "${ROUTER_URL}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"default","messages":[{"role":"user","content":"What is 2+3?"}],"max_tokens":32}'
echo ""
echo ""

# --- Test 2: GSM8K 精確度 + 並發吞吐測試 ---
echo "=== [2/2] ${ACC_EVAL_NAME} bench (n=${ACC_NUM_EXAMPLES}, parallel=${ACC_NUM_EXAMPLES}) ==="
python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
    --num-questions "$ACC_NUM_EXAMPLES" \
    --parallel "$ACC_NUM_EXAMPLES" \
    --port "$ROUTER_PORT"
echo ""
echo "=== Accuracy tests complete ==="
