#!/bin/bash
# ============================================================
#  Step 0: 建立 Docker container
#  用法:  bash 0_docker.sh          — 建立並進入 container
#         bash 0_docker.sh attach   — 重新進入已存在的 container
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

ACTION="${1:-create}"

if [ "$ACTION" = "attach" ]; then
    echo "[0_docker] Attaching to existing container: $CONTAINER_NAME"
    sudo docker exec -it "$CONTAINER_NAME" bash
    exit 0
fi

# Build volume args
VOL_ARGS=""
for v in "${DOCKER_VOLUMES[@]}"; do
    VOL_ARGS="$VOL_ARGS -v $v"
done

echo "[0_docker] Creating container: $CONTAINER_NAME"
echo "  Image:  $DOCKER_IMAGE"
echo "  SHM:    $DOCKER_SHM_SIZE"

sudo docker run -it --name "$CONTAINER_NAME" \
    --network=host \
    --pid=host \
    --ipc=host \
    --ulimit memlock=-1 \
    --cap-add=IPC_LOCK \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --privileged \
    --shm-size="$DOCKER_SHM_SIZE" \
    --group-add video \
    --group-add rdma \
    -e SGLANG_USE_AITER=1 \
    $VOL_ARGS \
    -v /dev/infiniband:/dev/infiniband \
    "$DOCKER_IMAGE" \
    bash
