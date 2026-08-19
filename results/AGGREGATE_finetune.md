# Phase 6 aggregate - finetune arms

Recomputed from saved weights on cuda; 1000 bootstrap resamples, 95% percentile CIs.

## Runs included

| arm | run | seed | epochs/rounds |
|---|---|---|---|
| local | `local_finetune_s0` | 0 | 40 |
| pooled | `pooled_finetune_s0` | 0 | 40 |

## Headline table

Balanced accuracy, final epoch/round. `[lo, hi]` = 95% bootstrap CI (shown when a single seed makes a std undefined).

| row | test n | local (routed) | pooled | gap-closed |
|---|---|---|---|---|
| centre 0 | 2483 | 0.7823 [0.748, 0.811] | 0.7857 [0.752, 0.817] | n/a |
| centre 1 | 791 | 0.7470 [0.679, 0.797] | 0.7042 [0.588, 0.853] | n/a |
| centre 2 | 672 | 0.6877 [0.643, 0.838] | 0.6834 [0.633, 0.839] | n/a |
| centre 3 | 452 | 0.6082 [0.546, 0.669] | 0.6394 [0.575, 0.703] | n/a |
| centre 4 | 164 | 0.6848 [0.611, 0.757] | 0.6980 [0.626, 0.768] | n/a |
| centre 5 | 88 | 0.4724 [0.358, 0.802] | 0.7504 [0.484, 0.904] | n/a |
| **pooled union** | 4650 | 0.7646 [0.740, 0.788] | 0.7711 [0.745, 0.792] | n/a |
| **mean over centres** | 4650 | 0.6637 | 0.7102 | n/a |

Local is reported in both aggregations per the Phase 3 policy: routed union **0.7646**, mean over centres **0.6637**. The union is size-weighted, so the largest silo dominates it; the mean weights every centre equally.

## Rare-class recall (< 2% of pooled train)

Balanced accuracy averages these away; a model can look fine overall and still miss them entirely.

| arm | class 5 (1.0%) | class 6 (1.1%) | mean rare | mean common |
|---|---|---|---|---|
| local (routed) | 0.727 | 0.923 | 0.825 | 0.744 |
| pooled | 0.691 | 0.923 | 0.807 | 0.759 |

## Caveats

- **Single seed** for: local, pooled. No std over seeds is computable, so the bootstrap CI is the only uncertainty estimate here - and it captures test-set sampling only, not run-to-run variance from initialisation, augmentation, and client order.
- **No federated arm present**, so gap-closed % is undefined: it measures how much of the pooled-minus-local headroom federation recovers, and there is nothing here to recover it. The pooled and local columns are the two endpoints only.
- **Small test sets**: centre(s) c4, c5 have under 200 test images (c4=164, c5=88). Their CIs are wide and their point estimates should not be read as measurements.
- Per-centre balanced accuracy also moves epoch to epoch within the converged region, which the bootstrap does not capture: it resamples one fixed set of predictions. Treat a per-centre difference as real only if it clears both the CI and that epoch-to-epoch scatter.

