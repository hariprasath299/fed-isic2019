#!/bin/sh
# Pooled finetune LR sweep, rerun after the float16-AMP NaN fix.
# Sequential: one 4 GB GPU, ~31 min per arm at bf16.
set -e
cd "C:/Users/harry/skin care"
PY=.venv/Scripts/python.exe
for spec in "1e4 1e-4" "5e4 5e-4" "1e3 1e-3"; do
  set -- $spec
  tag=$1; lr=$2
  echo "=== starting pooled_lrsweep_$tag (lr=$lr) at $(date) ==="
  "$PY" scripts/run.py \
    --arm pooled --mode finetune --epochs 5 --eval-every 1 \
    --lr "$lr" --optimizer adam --batch-size 64 --gamma 2.0 --alpha pooled \
    --amp --num-workers 0 --seed 0 --device cuda \
    --run-name "pooled_lrsweep_$tag" 2>&1 | tee "logs/pooled_lrsweep_$tag.log"
  echo "=== finished pooled_lrsweep_$tag at $(date) ==="
done
echo "=== SWEEP COMPLETE at $(date) ==="
