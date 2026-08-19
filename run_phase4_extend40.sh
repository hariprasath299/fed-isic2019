#!/bin/sh
# Extend Phase 4 arms A+B from 20 to 40 epochs. Both curves were still climbing
# at ep 20 (SPEC §5: "not still climbing at the end, else raise rounds").
# Same run names, so the CSVs and curves continue rather than fork.
set -e
cd "C:/Users/harry/skin care"
PY=.venv/Scripts/python.exe
LR=1e-4   # sweep winner, unchanged

echo "=== [A] pooled finetune, resume 20 -> 40 === $(date)"
"$PY" scripts/run.py --arm pooled --mode finetune --epochs 40 --eval-every 1 \
  --lr "$LR" --optimizer adam --batch-size 64 --gamma 2.0 --alpha pooled \
  --amp --num-workers 0 --seed 0 --device cuda --resume \
  --run-name pooled_finetune_s0 2>&1 | tee -a logs/pooled_finetune_s0.log

echo "=== [B] local finetune, resume 20 -> 40 === $(date)"
"$PY" scripts/run.py --arm local --mode finetune --epochs 40 --eval-every 1 \
  --lr "$LR" --optimizer adam --batch-size 64 --gamma 2.0 --alpha pooled \
  --amp --num-workers 0 --seed 0 --device cuda --resume \
  --run-name local_finetune_s0 2>&1 | tee -a logs/local_finetune_s0.log

echo "=== EXTENSION TO 40 EPOCHS COMPLETE at $(date) ==="
