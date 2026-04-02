#!/bin/bash
# Run DL v2 training on 2 DGX Spark nodes in parallel.
# Node 1 (local): configs 0-10, Node 2 (192.168.200.13): configs 10-20
#
# v2 improvements: Optuna HPO, per-config loss, CQR, ensemble
#
# Usage:
#   ./docker/run_dual_node_v2.sh

set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="noryangjin-gpu-v2"
NODE2="192.168.200.13"
NODE2_USER="nvidia"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_CMD="ssh -i $SSH_KEY $NODE2_USER@$NODE2"
RSYNC_SSH="ssh -i $SSH_KEY"
NODE2_DIR="/home/nvidia/noryangjin-gpu"

echo "=== Dual-Node DL v2 Training ==="
echo "Node 1 (local): configs 0-10"
echo "Node 2 ($NODE2): configs 10-20"
echo "Improvements: Optuna HPO, per-config loss, CQR, ensemble"
echo ""

# Build image locally
echo "Building Docker image on Node 1..."
docker build -t "$IMAGE_NAME" -f "$PROJECT_DIR/docker/Dockerfile" "$PROJECT_DIR"

# Sync project to Node 2
echo "Syncing project to Node 2..."
rsync -az --delete -e "$RSYNC_SSH" \
  "$PROJECT_DIR/" "$NODE2_USER@$NODE2:$NODE2_DIR/"

echo "Building Docker image on Node 2..."
$SSH_CMD "cd $NODE2_DIR && docker build -t $IMAGE_NAME -f docker/Dockerfile ."

# Run on Node 1: first 10 configs
echo ""
echo "Starting Node 1 (configs 0-10)..."
docker rm -f noryangjin-v2-n1 2>/dev/null || true
docker run --gpus all --ipc=host -d \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e CONFIG_SLICE="0:10" \
  --name noryangjin-v2-n1 \
  -v "$PROJECT_DIR/data:/workspace/data" \
  "$IMAGE_NAME" \
  python scripts/train_dl_v2.py

# Run on Node 2: last 10 configs
echo "Starting Node 2 (configs 10-20)..."
$SSH_CMD "\
  docker rm -f noryangjin-v2-n2 2>/dev/null || true; \
  docker run --gpus all --ipc=host -d \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -e NVIDIA_DISABLE_REQUIRE=1 \
    -e CONFIG_SLICE='10:20' \
    --name noryangjin-v2-n2 \
    -v '$NODE2_DIR/data:/workspace/data' \
    '$IMAGE_NAME' \
    python scripts/train_dl_v2.py"

echo ""
echo "Both nodes running. Monitor with:"
echo "  Node 1: docker logs -f noryangjin-v2-n1"
echo "  Node 2: $SSH_CMD docker logs -f noryangjin-v2-n2"
echo ""
echo "Wait for both to finish, then merge results from:"
echo "  Node 1: data/poc_results/dl_v2_results.json"
echo "  Node 2: $SSH_CMD cat $NODE2_DIR/data/poc_results/dl_v2_results.json"
