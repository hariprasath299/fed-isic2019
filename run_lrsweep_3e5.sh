#!/bin/sh
# Extra sweep point at 3e-5. 1e-4 won the {1e-4, 5e-4, 1e-3} grid but sits at
# the grid's edge, so the optimum may lie below it. Protocol identical to the
# other three points: 5 epochs, eval every epoch, seed 0, same everything else.
set -e
cd "C:/Users/harry/skin care"
"$PWD/.venv/Scripts/python.exe" scripts/run.py \
  --arm pooled --mode finetune --epochs 5 --eval-every 1 \
  --lr 3e-5 --optimizer adam --batch-size 64 --gamma 2.0 --alpha pooled \
  --amp --num-workers 0 --seed 0 --device cuda \
  --run-name pooled_lrsweep_3e5 2>&1 | tee logs/pooled_lrsweep_3e5.log
echo "=== 3e-5 SWEEP POINT COMPLETE at $(date) ==="
