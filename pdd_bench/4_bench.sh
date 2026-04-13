#!/bin/bash
# ============================================================
#  Step 4: 效能 Benchmark（throughput / latency sweep）
#  在 container 內執行。需在 router 就緒後執行。
#  用法:  bash 4_bench.sh
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

export PYTHONPATH="${SGLANG_PYTHON_PATH}:${PYTHONPATH}"

ROUTER_URL="http://${PREFILL_HOST}:${ROUTER_PORT}"
mkdir -p "$BENCH_OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${BENCH_OUTPUT_DIR}/summary_${TIMESTAMP}.txt"
CSV="${BENCH_OUTPUT_DIR}/data_${TIMESTAMP}.csv"

# 檢查 router 是否就緒
echo "[bench] Checking router at ${ROUTER_URL} ..."
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${ROUTER_URL}/health" 2>/dev/null || echo "000")
if [ "$STATUS" != "200" ]; then
    echo "[bench] ERROR: Router not reachable (HTTP $STATUS). Run 3_router.sh first."
    exit 1
fi
echo "[bench] Router is ready."

# 列印設定
cat <<INFO | tee "$SUMMARY"
================================================================
  PDD Benchmark Sweep
  Date:          $(date)
  Model:         $MODEL_PATH
  Router:        $ROUTER_URL
  Input lens:    ${BENCH_INPUT_LENS[*]}
  Output lens:   ${BENCH_OUTPUT_LENS[*]}
  Concurrencies: ${BENCH_CONCURRENCIES[*]}
  Duration mult: $BENCH_DURATION
  Request rate:  $BENCH_REQUEST_RATE
================================================================

INFO

# CSV header
echo "isl,osl,concurrency,num_prompts,request_rate,req_per_s,input_tok_s,output_tok_s,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,mean_e2e_ms,median_e2e_ms,p99_e2e_ms" > "$CSV"

printf "%-8s %-8s %-6s %-8s %-10s %-10s %-10s %-10s %-10s %-10s\n" \
    "ISL" "OSL" "Conc" "Prompts" "Req/s" "InTok/s" "OutTok/s" "TTFT_med" "TPOT_med" "E2E_med" | tee -a "$SUMMARY"
printf '%0.s-' {1..100} | tee -a "$SUMMARY"
echo "" | tee -a "$SUMMARY"

TOTAL=$(( ${#BENCH_INPUT_LENS[@]} * ${#BENCH_CONCURRENCIES[@]} ))
N=0

for idx in "${!BENCH_INPUT_LENS[@]}"; do
    ISL="${BENCH_INPUT_LENS[$idx]}"
    OSL="${BENCH_OUTPUT_LENS[$idx]:-${BENCH_OUTPUT_LENS[-1]}}"

    for CONC in "${BENCH_CONCURRENCIES[@]}"; do
        N=$((N + 1))

        if [[ "$BENCH_REQUEST_RATE" == "INF" ]]; then
            NUM_PROMPTS=$((CONC * BENCH_DURATION))
        else
            NUM_PROMPTS=$((BENCH_REQUEST_RATE * BENCH_DURATION))
        fi

        RESULT_JSON="${BENCH_OUTPUT_DIR}/bench_i${ISL}_o${OSL}_c${CONC}_${TIMESTAMP}.json"

        echo "" | tee -a "$SUMMARY"
        echo "[${N}/${TOTAL}] ISL=${ISL} OSL=${OSL} Conc=${CONC} Prompts=${NUM_PROMPTS}" | tee -a "$SUMMARY"

        python3 -m sglang.bench_serving \
            --model "$MODEL_PATH" \
            --dataset-name random \
            --host "$PREFILL_HOST" \
            --port "$ROUTER_PORT" \
            --num-prompts "$NUM_PROMPTS" \
            --random-input-len "$ISL" \
            --random-output-len "$OSL" \
            --random-range-ratio 1 \
            --request-rate "$BENCH_REQUEST_RATE" \
            --max-concurrency "$CONC" \
            --output-file "$RESULT_JSON" \
            2>&1 | tee "${RESULT_JSON%.json}.log"

        if [ $? -eq 0 ] && [ -f "$RESULT_JSON" ]; then
            python3 -c "
import json
d = json.load(open('$RESULT_JSON'))
req_s   = d.get('request_throughput', 0)
in_tok  = d.get('input_throughput', 0)
out_tok = d.get('output_throughput', 0)
mttft   = d.get('mean_ttft_ms', 0)
mdttft  = d.get('median_ttft_ms', 0)
p99ttft = d.get('p99_ttft_ms', 0)
mtpot   = d.get('mean_tpot_ms', 0)
mdtpot  = d.get('median_tpot_ms', 0)
p99tpot = d.get('p99_tpot_ms', 0)
me2e    = d.get('mean_e2e_latency_ms', 0)
mde2e   = d.get('median_e2e_latency_ms', 0)
p99e2e  = d.get('p99_e2e_latency_ms', 0)
# CSV
print(f'$ISL,$OSL,$CONC,$NUM_PROMPTS,$BENCH_REQUEST_RATE,{req_s:.2f},{in_tok:.0f},{out_tok:.0f},{mttft:.1f},{mdttft:.1f},{p99ttft:.1f},{mtpot:.2f},{mdtpot:.2f},{p99tpot:.2f},{me2e:.1f},{mde2e:.1f},{p99e2e:.1f}')
" >> "$CSV" 2>/dev/null

            python3 -c "
import json
d = json.load(open('$RESULT_JSON'))
print(f'  {d.get(\"request_throughput\",0):<10.2f}{d.get(\"input_throughput\",0):<10.0f}{d.get(\"output_throughput\",0):<10.0f}{d.get(\"median_ttft_ms\",0):<10.1f}{d.get(\"median_tpot_ms\",0):<10.2f}{d.get(\"median_e2e_latency_ms\",0):<10.1f}')
" | tee -a "$SUMMARY" 2>/dev/null
        else
            printf "  %-8s FAILED\n" "$CONC" | tee -a "$SUMMARY"
        fi
    done
done

echo "" | tee -a "$SUMMARY"
echo "================================================================" | tee -a "$SUMMARY"
echo "  Sweep complete: $(date)" | tee -a "$SUMMARY"
echo "  Summary: $SUMMARY" | tee -a "$SUMMARY"
echo "  CSV:     $CSV" | tee -a "$SUMMARY"
echo "================================================================" | tee -a "$SUMMARY"
