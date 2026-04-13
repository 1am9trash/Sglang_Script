# SGLang Prefill-Decode Disaggregation 測試腳本

## 適用範圍
- Model: deepseek v3.2, GLM-5-fp8
- Workload: single-node 1P1D, multi-node 1P1D

## 檔案結構

| 腳本 | 說明 |
|------|------|
| `config.sh` | 設定檔，使用時只需要修改此檔 |
| `0_docker.sh` | 建立 / 進入 Docker |
| `1_decode.sh` | run decode server |
| `2_prefill.sh` | run prefill server |
| `3_router.sh` | run router |
| `4_bench.sh` | benchmark |
| `5_acc.sh` | sanity check, GSM8K |

## 設定

編輯 `config.sh`，需要修改 PDD mode 、使用 model 、機器的 IP 跟 model 位置。

```bash
PDD_MODE="single"           # "single" = single-node 1P1D / "multi" = 2-node 1P1D
MODEL_NAME="glm5"           # "glm5" 或 "deepseek"
MODEL_PATH="/path/model"
```

Multi-node 需額外填寫 IP：

```bash
PREFILL_HOST="10.235.58.xxx"
DECODE_HOST="10.235.58.xxx"
ROUTER_HOST="$DECODE_HOST"  # 預設在 router 在 decode node 執行，可改成 prefill node
```

## Single-node 操作

一台機器，GPU 拆兩組（0-3 prefill / 4-7 decode），TP=4。

```bash
# host 上建立 container
bash 0_docker.sh

# 以下都在 container 內執行，開三個 terminal
bash 1_decode.sh       # terminal 1
bash 2_prefill.sh      # terminal 2
bash 3_router.sh       # terminal 3

# 就緒後
bash 5_acc.sh          # accuracy
bash 4_bench.sh        # benchmark
```

## Multi-node 操作

兩台機器各用全部 8 張 GPU，TP=8。

```sh
Decode 機器           Prefill 機器
─────────────         ──────────────
bash 0_docker.sh      bash 0_docker.sh
bash 1_decode.sh      bash 2_prefill.sh
bash 3_router.sh
bash 5_acc.sh
bash 4_bench.sh
```
