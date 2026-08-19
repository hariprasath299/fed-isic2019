#!/bin/sh
# Pooled + local finetune arms at the locked LR, for the seeds given.
# Usage: ./run_seeds12.sh <lr> <seed> [seed...]
# Seeds run in separate slots on purpose: each is self-contained, so seed 2
# blocks nothing and can take the next free GPU window.
# Protocol matches seed 0: 40 epochs, eval every epoch, batch 64, Adam, focal
# gamma 2.0 with pooled alphas, augmentation on, bf16 AMP.
# One difference from s0, deliberate: s0 reached 40 by resuming at ep 20 and so
# carries one Adam-moment reset there. These run 40 continuous epochs, which is
# the cleaner trajectory. Reproducing the reset would copy an artefact.
set -e
cd "C:/Users/harry/skin care"
PY=.venv/Scripts/python.exe
LR="$1"
shift || true
[ -n "$LR" ] && [ "$#" -gt 0 ] || { echo "usage: $0 <lr> <seed> [seed...]"; exit 2; }
echo "=== seeds $* at lr=$LR, 40 epochs === $(date)"

for SEED in "$@"; do
  echo "=== [pooled s$SEED] === $(date)"
  "$PY" scripts/run.py --arm pooled --mode finetune --epochs 40 --eval-every 1 \
    --lr "$LR" --optimizer adam --batch-size 64 --gamma 2.0 --alpha pooled \
    --amp --num-workers 0 --seed "$SEED" --device cuda --resume \
    --run-name "pooled_finetune_s$SEED" 2>&1 | tee -a "logs/pooled_finetune_s$SEED.log"

  echo "=== [local s$SEED] === $(date)"
  "$PY" scripts/run.py --arm local --mode finetune --epochs 40 --eval-every 1 \
    --lr "$LR" --optimizer adam --batch-size 64 --gamma 2.0 --alpha pooled \
    --amp --num-workers 0 --seed "$SEED" --device cuda --resume \
    --run-name "local_finetune_s$SEED" 2>&1 | tee -a "logs/local_finetune_s$SEED.log"
done
echo "=== SEEDS $* COMPLETE at $(date) ==="
