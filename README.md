# fed-isic2019

Three-arm benchmark harness for **Fed-ISIC2019** (6 dermatology centres, 8-class
skin-lesion classification, severe class imbalance and a 28x client-size skew).

The question the repo answers: **how much of the gap between local-only training
and centralised pooling does federated learning recover?** Metric: balanced
accuracy (per client and pooled). Strategies: FedAvg, FedProx, FedAdam, FedYogi,
FedAdagrad — hand-rolled in plain PyTorch, no FL framework.

Read `SPEC.md` for the full plan, verified reference facts, and acceptance
criteria. This README is just the how-to-run.

## Install

```bash
pip install -r requirements.txt        # or: pip install -e .
```

## Test the harness (no data needed, <1 min, CPU)

```bash
python -m pytest tests/ -q             # 11 tests: averaging/federation invariants
python scripts/make_fake_features.py   # synthetic 6-silo feature cache
python scripts/run.py --arm fed --mode probe --strategy fedavg \
    --rounds 3 --local-steps 5 --features-dir data/features_fake \
    --out /tmp/smoke --num-workers 0 --device cpu
```

## Real pipeline

```bash
# Phase 0 — verify the HF mirror (schema, counts, class distribution, samples)
python scripts/phase0_verify_data.py --out inspection

# Phase 1 — cache frozen EfficientNet-B0 features (one GPU pass)
python scripts/phase1_cache_features.py --out data/features --device cuda

# Phase 3 — three arms as linear probes on cached features (fast)
python scripts/run.py --arm pooled --mode probe --epochs 20
python scripts/run.py --arm local  --mode probe --epochs 20
python scripts/run.py --arm fed    --mode probe --strategy fedavg --rounds 50

# Phase 4/5 — full fine-tuning (GPU; add --amp)
python scripts/run.py --arm pooled --mode finetune --epochs 20 --amp
python scripts/run.py --arm local  --mode finetune --epochs 20 --amp
python scripts/run.py --arm fed    --mode finetune --strategy fedavg \
    --rounds 50 --local-steps 100 --amp
python scripts/run.py --arm fed    --mode finetune --strategy fedprox \
    --prox-mu 0.1 --rounds 50 --local-steps 100 --amp
python scripts/run.py --arm fed    --mode finetune --strategy fedadam \
    --server-lr 1e-2 --rounds 50 --local-steps 100 --amp
```

Repeat the runs that go in the final table with `--seed 0 --seed 1 --seed 2`
(one run per seed).

## Outputs

Each run `{name}` writes to `--out` (default `results/`):

- `{name}.csv` — one row per evaluation: balanced accuracy per client + pooled
- `{name}.pt` — atomic resume checkpoint (`--resume` continues pooled/fed runs)
- `{name}_final.pt` — final weights
- `{name}_per_class.csv` — per-class recall on the pooled test set
- `{name}_config.json` — exact arguments of the run

## Colab notes

- Keep the HF cache and outputs on the **local** disk (`/content/...`), never
  train while reading from a mounted Drive. Copy result CSVs/checkpoints to
  Drive after (or periodically) — they're small.
- `--amp` for fine-tuning on T4/A100; probes don't need it.
- A disconnect costs at most one round: rerun the same command with `--resume`.

## Layout

```
fedisic/
  data.py          HF mirror loading, per-centre silos, transforms, feature cache
  losses.py        weighted focal loss + inverse-frequency alphas
  models.py        EfficientNet-B0 (finetune), frozen backbone, linear probe
  evaluate.py      balanced accuracy per client/pooled, per-class recall, bootstrap CIs
  utils.py         seeding, append-only CSV logger, atomic checkpoints
  fed/
    averaging.py   weighted state-dict averaging (dtype-safe, validated weights)
    strategies.py  local step (FedAvg/FedProx) + FedOpt server optimizers
    simulate.py    the serial round loop with eval/checkpoint hooks and resume
scripts/
  phase0_verify_data.py    schema + count verification (run this first)
  phase1_cache_features.py frozen-feature caching
  run.py                   universal runner: --arm {pooled,local,fed}
  make_fake_features.py    synthetic features for offline smoke tests
tests/
  test_harness.py          11 CPU unit tests for the FL machinery
```
