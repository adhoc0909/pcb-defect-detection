#!/usr/bin/env bash
set -euo pipefail

# Install SPDConv Ultralytics fork.
# Use a separate env to avoid conflicts with baseline ultralytics.

REPO_URL="https://github.com/Cateners/yolov8-spd.git"
TARGET_DIR="${1:-./third_party/yolov8-spd}"

echo "[1/4] Prepare directory: ${TARGET_DIR}"
mkdir -p "$(dirname "${TARGET_DIR}")"

if [ -d "${TARGET_DIR}/.git" ]; then
  echo "[2/4] Repo already exists. Pulling latest..."
  git -C "${TARGET_DIR}" pull
else
  echo "[2/4] Cloning ${REPO_URL} -> ${TARGET_DIR}"
  git clone "${REPO_URL}" "${TARGET_DIR}"
fi

# Uninstall pip ultralytics if present (the fork also provides `ultralytics` package)
echo "[3/4] Uninstalling existing ultralytics (if any)..."
python -m pip uninstall -y ultralytics || true

# Install fork in editable mode
echo "[4/4] Installing SPDConv fork (editable)..."
python -m pip install -e "${TARGET_DIR}"

echo "✅ SPDConv fork installed. You should be able to: python -c 'from ultralytics import YOLO; print(YOLO)'"
