# Phase 3 — probe arms (frozen EfficientNet-B0 features, seed 0)

Status: **PASSED** acceptance (SPEC §5, phase 3).
Runs: `pooled_probe_s0` (20 ep), `local_probe_s0` (6 silos x 20 ep),
`fed_probe_fedavg_s0` (50 rounds x 100 local steps). All numbers below trace to
a file in `results/`.

## Harness fix that made this run valid

`scripts/run.py:run_local` logged the per-epoch `bal_acc_own` row and the
6x6 cross-silo row to the *same* `CsvLogger`. `CsvLogger` fixes its header from
the first row it sees and writes with `extrasaction="ignore"`, so every
`bal_acc_c0..c5` / `bal_acc_pooled` column of the cross-silo rows was silently
dropped. Cross-silo rows now go to `{run}_xeval.csv`. `pytest tests/ -q` = 11/11
with `fedisic/fed/*` unmodified.

## Headline table — FINAL epoch/round (reporting policy below)

| centre | n_train | n_test | local | fedavg | pooled | fed - local |
|---|---|---|---|---|---|---|
| c0 | 9930 | 2483 | 0.5450 | 0.5020 | 0.5349 | -0.0430 |
| c1 | 3163 |  791 | 0.6253 | 0.7241 | 0.7000 | +0.0988 |
| c2 | 2691 |  672 | 0.5815 | 0.5220 | 0.6639 | -0.0595 |
| c3 | 1807 |  452 | 0.4831 | 0.4267 | 0.4555 | -0.0564 |
| c4 |  655 |  164 | 0.5882 | 0.4889 | 0.4745 | -0.0993 |
| c5 |  351 |   88 | 0.4099 | 0.7333 | 0.7427 | +0.3234 |

Per-centre entries are own-model-vs-fed by design: the `local` column is centre
i's own model on centre i's test set (`local_probe_s0.csv`), `fedavg`/`pooled`
are the single global model on that same test set.

### Pooled-row aggregations (both kept, per reporting policy 4)

| aggregation | local | fedavg | pooled | gap closed |
|---|---|---|---|---|
| **mean over centres** (comparable aggregate) | 0.5388 | 0.5661 | 0.5953 | **48.4%** |
| **routed union** (deployable local baseline) | 0.6002 | 0.5127 | 0.5638 | n/a |
| single model on pooled test set | — | 0.5127 | 0.5638 | — |

- *mean over centres*: unweighted mean of the six per-centre numbers above.
  Every arm is measured the same way, so this is the ordering metric.
- *routed union*: pooled test set scored as one set, each image predicted by its
  own centre's local model. This is the honest **deployable** local baseline,
  but it grants local an oracle centre-ID router that neither fed nor pooled
  has — which is exactly why it scores 0.6002, above pooled. Do not read it as
  federation losing; read it as routing being worth ~6 points on this data.
- Best single local model deployed everywhere: c0, 0.4975 (`local_probe_s0_xeval.csv`).

**Acceptance — ordering:** `local 0.5388 <= fed 0.5661 <= pooled 0.5953` holds on
the mean-over-centres metric. `fed 0.5127 <= pooled 0.5638` also holds on the
pooled test set. The local arm has no single-model pooled number by construction.

## Per-class recall, pooled test set (`*_per_class.csv`)

| arm | c0 | c1 | c2 | c3 | c4 | c5 | c6 | c7 | nonzero |
|---|---|---|---|---|---|---|---|---|---|
| local (routed)  | 0.580 | 0.700 | 0.567 | 0.548 | 0.567 | 0.491 | 0.865 | 0.483 | 8/8 |
| fedavg          | 0.501 | 0.682 | 0.464 | 0.310 | 0.641 | 0.545 | 0.769 | 0.189 | 8/8 |
| pooled          | 0.542 | 0.649 | 0.613 | 0.406 | 0.555 | 0.491 | 0.808 | 0.448 | 8/8 |

**Acceptance — no degenerate arm:** all 8 classes have nonzero recall in all
three arms; nothing collapsed onto the majority class. `run_local` writes no
per-class file, so `local_probe_s0_per_class.csv` was generated from the six
saved `_c{i}_final.pt` checkpoints under the routed-union definition.

## Best vs final (curve note — not used in the headline table)

| run | metric | final | best | round of best |
|---|---|---|---|---|
| pooled_probe_s0 | bal_acc_pooled | 0.5638 | 0.5668 | ep 15 |
| pooled_probe_s0 | mean-over-centres | 0.5953 | 0.6188 | ep 15 |
| fed_probe_fedavg_s0 | bal_acc_pooled | 0.5127 | 0.5273 | rd 25 |
| fed_probe_fedavg_s0 | mean-over-centres | 0.5661 | 0.5972 | rd 25 |
| local c0 | bal_acc_own | 0.5450 | 0.5450 | ep 20 |
| local c1 | bal_acc_own | 0.6253 | 0.7117 | ep 14 |
| local c2 | bal_acc_own | 0.5815 | 0.5875 | ep 19 |
| local c3 | bal_acc_own | 0.4831 | 0.4877 | ep 19 |
| local c4 | bal_acc_own | 0.5882 | 0.5968 | ep 12 |
| local c5 | bal_acc_own | 0.4099 | 0.4099 | ep 19 |

Pooled is flat-noisy from ~ep 13. FedAvg peaks at rd 25 and **drifts down 1.5
points** (0.5273 -> 0.5127) over rounds 26-50. Under reporting policy 3 this is
the >1-point drift trigger: if finetune FedAvg repeats it, pause after seed 0 and
propose cosine LR decay or a shorter schedule before seeds 1-2. Local c1 drifts
8.6 points off its ep-14 peak — small silo, 791 test images, overfitting a linear
head; expected, and it is why final-round reporting matters.

## Who benefits (already visible)

c5 (351 train images, the smallest silo) gains **+0.32** balanced accuracy from
federation; c1 gains +0.10. Every other centre loses. **c4 (655 train) loses
-0.099** despite being the second-smallest silo — it is the one centre that
breaks the size-monotone story, and it is tracked explicitly in Phase 4.

## Reporting policy (fixed here, applied in Phase 6)

1. The headline table uses **final-round / final-epoch** numbers. The budget is
   fixed in advance and there is no validation split, so selecting the best round
   on the test set would be test-set model selection. Best round + value are
   recorded separately as a curve note (table above).
2. The local baseline is reported in **both** aggregations everywhere — routed
   union (deployable) and mean-over-centres (comparable) — always labeled.
   Per-centre comparisons remain own-model-vs-fed.
3. Probe absolute numbers are not the deliverable (SPEC §5); the ordering and
   sane per-class behaviour are. Both hold.
