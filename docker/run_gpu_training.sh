#!/bin/bash
# GPU training launcher for DGX Spark (GB10 Blackwell)
#
# Usage:
#   ./docker/run_gpu_training.sh [tft|gru|all]
#
# Requires: Docker with NVIDIA runtime

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="noryangjin-gpu-train"
CONTAINER_NAME="noryangjin-train"
MODEL="${1:-tft}"

echo "=== Noryangjin Fish Price GPU Training ==="
echo "Project: $PROJECT_DIR"
echo "Model: $MODEL"
echo ""

# Build image
echo "Building Docker image..."
docker build -t "$IMAGE_NAME" -f "$PROJECT_DIR/docker/Dockerfile" "$PROJECT_DIR"

# Stop existing container if running
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# Run training
case "$MODEL" in
  tft)
    echo "Training TFT (Temporal Fusion Transformer)..."
    docker run --gpus all --name "$CONTAINER_NAME" \
      -v "$PROJECT_DIR/data:/workspace/data" \
      "$IMAGE_NAME" \
      python scripts/train_tft.py
    ;;
  gru)
    echo "Training VMD-GRU..."
    docker run --gpus all --name "$CONTAINER_NAME" \
      -v "$PROJECT_DIR/data:/workspace/data" \
      "$IMAGE_NAME" \
      python scripts/train_gru.py
    ;;
  all)
    echo "Training all GPU models..."
    docker run --gpus all --name "$CONTAINER_NAME" \
      -v "$PROJECT_DIR/data:/workspace/data" \
      "$IMAGE_NAME" \
      bash -c "python scripts/train_tft.py && python scripts/train_gru.py"
    ;;
  *)
    echo "Unknown model: $MODEL"
    echo "Usage: $0 [tft|gru|all]"
    exit 1
    ;;
esac

echo ""
echo "=== Training complete ==="
echo "Results saved to data/poc_results/"
