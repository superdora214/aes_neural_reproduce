#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.cuda.get_device_name(0))"
mkdir -p runs
RUN_DIR="runs/full_r2_$(date +%Y%m%d_%H%M%S)"
mkdir "$RUN_DIR"
cp train.py "$RUN_DIR/train_source.py"
echo "[OUTDIR] $RUN_DIR"
python -u train.py \
  --rounds 2 --bytes 0,1 \
  --train-size 10000000 --eval-size 1000000 --test-size 1000000 \
  --epochs 20 --batch-size 1000 --depth 5 \
  --period 9 --lr-high 0.002 --lr-low 0.0001 --l2 0.00001 \
  --seed 20261024 --device cuda --save-every-epoch \
  --outdir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/train.log"
echo "[DONE] $RUN_DIR"
