# SPEC — Fed-ISIC2019 three-arm federated learning study

This document is the contract for finishing the project. The scaffold in this
repo is real, tested code — read "What already works" before changing anything,
and "Verify on real data" before trusting anything.

## 1. Mission

Quantify what federated learning buys on Fed-ISIC2019 (6 dermatology centres,
8-class skin-lesion classification, ~49% majority class down to <1% rarest,
28x train-size skew between the largest and smallest centre).

Three arms, identical model/loss/eval:

| arm | training data | role |
|---|---|---|
| pooled | all 18,597 train images | upper bound |
| local | each centre's own silo (6 models) | lower bound |
| fed | FedAvg / FedProx / FedAdam / FedYogi / FedAdagrad | the question |

Headline number per centre and pooled: **balanced accuracy**, and the
**fraction of the pooled-minus-local gap that federation closes**:
`(fed - local) / (pooled - local)`.

Success target from the project plan: federated balanced accuracy > 0.60
pooled, with FedAvg expected in the **0.59–0.66** band and pooled fine-tuning
around **0.65+**. If pooled lands far below 0.65, debug pooled first — the
federated numbers are uninterpretable under a broken baseline.

## 2. Verified reference facts (fetched from FLamby source, Aug 2026)

From `owkin/FLamby`, `flamby/datasets/fed_isic2019/` — the benchmark of record:

- Model: **EfficientNet-B0**, ImageNet-pretrained, final FC replaced with 8-way
  head (they use `efficientnet_pytorch`; we use torchvision's — same arch).
- Loss: **weighted focal loss**, gamma = 2.0, fixed alphas
  `[5.5813, 2.0472, 7.0204, 26.1194, 9.5369, 101.0707, 92.5224, 38.3443]`
  (kept in `fedisic/losses.py` as `FLAMBY_ALPHA`).
- Optimiser: **Adam, lr 5e-4**, batch size **64**, `NUM_EPOCHS_POOLED = 20`.
- Metric: `sklearn.metrics.balanced_accuracy_score` on argmax predictions.
- Images: FLamby preprocesses with colour constancy + resize to **shorter edge
  224** (aspect kept), then the benchmark crops **200x200** (random crop +
  flips/rotate(50)/brightness-contrast/shear/coarse-dropout for train,
  centre crop for test). Our torchvision pipeline mirrors this
  (`fedisic/data.py: train_transforms / eval_transforms`).
- Per-centre (train/test) counts:
  `(9930/2483) (3163/791) (2691/672) (1807/452) (655/164) (351/88)`
  → 18,597 train / 4,650 test / 23,247 total.
- Data source for this repo: HF mirror **`flwrlabs/fed-isic2019`** (23.2k rows,
  matches the totals). Exact column schema is auto-detected, not yet verified —
  see §4.

## 3. What already works (do not break)

`python -m pytest tests/ -q` → **11/11 passing**. The tests pin down:

1. Averaging two identical clients returns the identical model.
2. A one-client federation is bit-equal to centralised training (same steps).
3. Client weights are validated: non-negative, sum to 1.
4. Integer state-dict buffers (BatchNorm `num_batches_tracked`) keep their
   dtype through averaging — naive float averaging crashes a round later.
5. FedProx with mu=0 is exactly FedAvg; with large mu it stays measurably
   closer to the global weights than FedAvg does.
6. FedAdagrad server step matches a hand-computed value; FedOpt applies the
   adaptive update only to trainable keys and plain-averages buffers.
7. Focal loss with gamma=0, alpha=1 reduces exactly to cross-entropy;
   inverse-frequency alphas give absent classes weight 0.

Smoke-tested end-to-end on synthetic features (`scripts/make_fake_features.py`):
all three arms, all five strategies, CSV schema, atomic checkpoints, per-class
recall dump, and `--resume` (restores model + server-optimizer state + round).

Design decisions baked in (with reasons — keep them):

- **Rounds are fixed local steps, not epochs** (`--local-steps`, default 100).
  With 28x size skew, "one epoch each" gives the biggest silo 28x more updates
  per round on top of its 53% averaging weight.
- **Per-client optimizer re-initialised each round** (standard FedAvg);
  pooled/local arms keep one optimizer across epochs (`local_train(opt=...)`).
- **Focal alphas default to `--alpha pooled`**: computed from the actually
  loaded pooled label counts, so they survive any label-id permutation in the
  mirror. `--alpha flamby` uses the published constants and is only valid if
  phase 0 confirms label order. `--alpha local` is the privacy-realistic
  per-silo variant — worth one comparison run.
- **Serial simulation** on one GPU; nothing but weights crosses clients.
- Every eval appends a CSV row and writes an atomic checkpoint; a Colab
  disconnect costs at most one round.

## 4. Verify on real data before any training (Phase 0)

`python scripts/phase0_verify_data.py --out inspection` must exit 0. It checks
or reveals, in order of what could silently invalidate everything:

1. **Splits**: mirror must have `train` and `test` (FLamby's fixed split). If
   the split is encoded differently (e.g. a `fold` column), adapt
   `load_hf_dataset()`.
2. **Column names**: auto-detected among
   image `(image|img)`, label `(label|target|labels|diagnosis)`,
   center `(center|centre|client|datacenter|center_id)`. The script prints the
   detected schema and full features spec — fix `fedisic/data.py` candidates if
   detection is wrong.
3. **Centre id ↔ FLamby centre correspondence**: per-centre counts must match
   §2 exactly. If counts match but ids are permuted vs FLamby's ordering, all
   analysis still works (ids are just names), only cross-paper comparisons of
   individual centres need care.
4. **Label-id order**: printed mapping. Only affects `--alpha flamby`.
5. **Image preprocessing**: the script saves one sample per centre and prints
   sizes. Shorter edge exactly 224 → mirror ships FLamby-preprocessed images
   (expected). If images are raw ISIC (large/variable), the pipeline still
   works (`Resize(224)` is already in the eval transform), but note the
   deviation: colour constancy would be missing. Flag it, don't silently eat a
   few points of balanced accuracy.

## 5. Phase plan with acceptance criteria

Phases 0–2 cost minutes and catch ~everything. Do not skip ahead.

| phase | command(s) | accept when |
|---|---|---|
| 0 data | `scripts/phase0_verify_data.py` | exit 0; sample images look like dermoscopy; class table saved |
| 1 features | `scripts/phase1_cache_features.py --device cuda` | `train.npz` features (18597, 1280), `test.npz` (4650, 1280) |
| 2 harness | `python -m pytest tests/ -q` on the target machine | 11/11 |
| 3 probe arms | `run.py --arm {pooled,local,fed} --mode probe` | ordering local ≤ fed ≤ pooled on pooled bal-acc; no arm degenerate (per-class recall not all-zero outside the majority class) |
| 4 finetune arms | same with `--mode finetune --amp`, 3 seeds | pooled ≈ 0.65+; FedAvg in 0.59–0.66; curves in CSVs not still climbing at the end (else raise rounds) |
| 5 strategies | fed with `fedprox/fedadam/fedyogi/fedadagrad`, 3 seeds | best-of-sweep per strategy reported with mean ± std over seeds |
| 6 analysis | new script (see §7) | final table + who-benefits + rare-class recall + bootstrap CIs |

Probe-phase absolute numbers are not the point (frozen ImageNet features on
dermoscopy will be mediocre); the point is that the machinery produces the
right ordering and sane per-class behaviour before GPU-days are spent.

## 6. Hyperparameter policy

- Sweep pooled LR only: `{1e-4, 5e-4, 1e-3}` (FLamby's 5e-4 is the favourite).
  Carry the winner to every arm — do not re-tune per arm; note the caveat that
  this mildly favours pooled.
- FedProx: `--prox-mu {0.01, 0.1, 1.0}`, short runs (~15 rounds) to pick, then
  full runs at the winner.
- FedOpt: `--server-lr {1e-3, 1e-2, 1e-1}` the same way; `tau=1e-3`,
  `beta1=0.9`, `beta2=0.99` fixed.
- Defaults elsewhere: batch 64, 50 rounds x 100 local steps, Adam clients,
  eval every round, seeds `{0, 1, 2}` for anything in the final table.
- Budget sanity: one fed fine-tune round = 6 x 100 steps x batch 64 ≈ 38k
  images ≈ 2 pooled epochs of compute. 50 rounds is a real T4 session — use
  `--amp`, keep `--eval-every 1` (test set is only 4,650 images), rely on
  `--resume` across disconnects.

## 7. Remaining work (in order)

1. **Run phases 0–3** as above; fix `fedisic/data.py` if phase 0 demands it.
2. **`scripts/aggregate_results.py`** (new): read `results/*.csv` +
   `*_per_class.csv`, and emit
   - the headline table: rows = centres 0–5 + pooled; columns = local, fedavg,
     best-fed-variant, pooled (mean ± std over seeds), plus gap-closed %;
   - bootstrap 95% CIs from `fedisic.evaluate.bootstrap_balanced_accuracy`
     using the saved `*_final.pt` weights, at minimum for centres 4 and 5
     (test n = 164 and 88 — point estimates there are not measurements);
   - rare-class recall comparison (classes with <2% prevalence) across arms.
3. **Plots** (matplotlib, no seaborn): training curves from the CSVs
   (bal_acc_pooled vs round per strategy); who-benefits scatter — x = silo
   train size (log), y = fed minus local balanced accuracy per centre.
4. **Optional analyses** (the "make it yours" angles, pick per interest):
   - Feature-vs-classifier decomposition: probe-arm vs finetune-arm gap-closed
     tells whether federation's value is in the backbone or the head.
   - Cross-silo generalisation matrix from the `local_xeval` CSV rows
     (already logged): 6x6 heatmap of local model i evaluated on silo j.
5. **Writeup numbers discipline**: every number in the report traces to a CSV
   in `results/` — never a number typed from memory.

## 8. Engineering rules

- Data and HF cache live on local disk (Colab: `/content`), never a mounted
  Drive; sync small artefacts (CSVs, checkpoints) out, not in.
- Never overwrite a run: run names encode arm/mode/strategy/seed; reruns either
  `--resume` or get a new `--run-name`.
- Any change to `fedisic/fed/*` requires the 11 tests to pass unmodified, or a
  deliberate test update explained in the commit message.
- Keep stdout logs: every round prints per-client + pooled balanced accuracy.
- GPU etiquette: `--amp` for finetune; probes are fine anywhere, including CPU.
