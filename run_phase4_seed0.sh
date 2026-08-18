#!/bin/sh
# Phase 4 (SPEC §5), arms A+B only: pooled + local finetune at the winning
# sweep LR, seed 0. The fed arm (~11 h) is deferred and run separately.
# Sequential on one GPU. Both arms are --resume-able: run.py checkpoints at
# each eval, so a crash or reboot costs at most one epoch, never the run.
set -e
cd "C:/Users/harry/skin care"
PY=.venv/Scripts/python.exe

LR=$("$PY" - <<'PYEOF'
import csv, glob, re, sys
best = None
for f in sorted(glob.glob("results/pooled_lrsweep_*.csv")):
    if "_per_class" in f:
        continue
    rows = list(csv.DictReader(open(f)))
    if not rows:
        continue
    tag = re.search(r"lrsweep_([0-9a-z]+)\.csv", f).group(1)
    lr = {"1e4": "1e-4", "5e4": "5e-4", "1e3": "1e-3"}[tag]
    score = float(rows[-1]["bal_acc_pooled"])
    print(f"#   lr={lr}  final_bal_acc={score:.4f}  epochs={len(rows)}", file=sys.stderr)
    if best is None or score > best[0]:
        best = (score, lr)
if best is None:
    sys.exit("no sweep CSVs found")

# Guard: never spend 5.6 h on a degenerate sweep. A collapsed run scores about
# 1/n_classes and predicts a single class -- exactly what the float16 NaN bug
# produced. Epoch 1 of a healthy run already scores ~0.56.
score, lr = best
if score < 0.45:
    sys.exit(f"ABORT: winning sweep bal_acc {score:.4f} < 0.45 -- looks degenerate, not launching")
tag = {"1e-4": "1e4", "5e-4": "5e4", "1e-3": "1e3"}[lr]
pc = f"results/pooled_lrsweep_{tag}_per_class.csv"
try:
    nz = sum(1 for r in csv.DictReader(open(pc)) if float(r["recall"]) > 0)
except FileNotFoundError:
    sys.exit(f"ABORT: {pc} missing -- winning run did not finish cleanly")
if nz < 4:
    sys.exit(f"ABORT: winner predicts only {nz}/8 classes -- degenerate, not launching")
print(f"#   winner lr={lr} bal_acc={score:.4f} nonzero_class_recall={nz}/8", file=sys.stderr)
print(lr)
PYEOF
)
echo "=== winning LR: $LR ==="

echo "=== [A] pooled finetune, seed 0, 20 epochs === $(date)"
"$PY" scripts/run.py --arm pooled --mode finetune --epochs 20 --eval-every 1 \
  --lr "$LR" --optimizer adam --batch-size 64 --gamma 2.0 --alpha pooled \
  --amp --num-workers 0 --seed 0 --device cuda --resume \
  --run-name pooled_finetune_s0 2>&1 | tee -a logs/pooled_finetune_s0.log

echo "=== [B] local finetune, seed 0, 20 epochs x 6 silos === $(date)"
"$PY" scripts/run.py --arm local --mode finetune --epochs 20 --eval-every 1 \
  --lr "$LR" --optimizer adam --batch-size 64 --gamma 2.0 --alpha pooled \
  --amp --num-workers 0 --seed 0 --device cuda --resume \
  --run-name local_finetune_s0 2>&1 | tee -a logs/local_finetune_s0.log

echo "=== PHASE 4 ARMS A+B COMPLETE at $(date) ==="
