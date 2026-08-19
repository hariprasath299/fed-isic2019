# Phase 6 aggregate - probe arms

Recomputed from saved weights on cuda; 1000 bootstrap resamples, 95% percentile CIs.

## Runs included

| arm | run | seed | epochs/rounds |
|---|---|---|---|
| fedavg | `fed_probe_fedavg_s0` | 0 | 50 |
| local | `local_probe_s0` | 0 | 20 |
| pooled | `pooled_probe_s0` | 0 | 20 |

## Headline table

Balanced accuracy, final epoch/round. `[lo, hi]` = 95% bootstrap CI (shown when a single seed makes a std undefined).

| row | test n | local (routed) | fedavg | pooled | gap-closed |
|---|---|---|---|---|---|
| centre 0 | 2483 | 0.5450 [0.509, 0.582] | 0.5020 [0.466, 0.539] | 0.5349 [0.495, 0.573] | n/a |
| centre 1 | 791 | 0.6253 [0.542, 0.730] | 0.7241 [0.617, 0.863] | 0.7000 [0.592, 0.844] | 132.1% |
| centre 2 | 672 | 0.5815 [0.537, 0.716] | 0.5220 [0.461, 0.657] | 0.6639 [0.556, 0.715] | -72.2% |
| centre 3 | 452 | 0.4831 [0.423, 0.545] | 0.4267 [0.371, 0.486] | 0.4555 [0.392, 0.522] | n/a |
| centre 4 | 164 | 0.5882 [0.510, 0.662] | 0.4889 [0.427, 0.548] | 0.4745 [0.407, 0.547] | n/a |
| centre 5 | 88 | 0.4099 [0.327, 0.646] | 0.7333 [0.499, 0.849] | 0.7427 [0.514, 0.869] | 97.2% |
| **pooled union** | 4650 | 0.6002 [0.574, 0.626] | 0.5127 [0.485, 0.538] | 0.5638 [0.535, 0.590] | n/a |
| **mean over centres** | 4650 | 0.5388 | 0.5661 | 0.5953 | 48.4% |

gap-closed = (fed - local) / (pooled - local), using **fedavg**. It reads as the share of the local-to-pooled headroom that federation recovers at that centre. Values outside 0-100% are meaningful, not errors: **above 100%** means federation beat centralised training there, **negative** means it landed below that centre's own local model. It is **n/a** wherever pooled did not beat local, because the headroom it normalises by is then zero or negative and the ratio stops meaning anything.

Local is reported in both aggregations per the Phase 3 policy: routed union **0.6002**, mean over centres **0.5388**. The union is size-weighted, so the largest silo dominates it; the mean weights every centre equally.

## Rare-class recall (< 2% of pooled train)

Balanced accuracy averages these away; a model can look fine overall and still miss them entirely.

| arm | class 5 (1.0%) | class 6 (1.1%) | mean rare | mean common |
|---|---|---|---|---|
| local (routed) | 0.491 | 0.865 | 0.678 | 0.574 |
| fedavg | 0.545 | 0.769 | 0.657 | 0.464 |
| pooled | 0.491 | 0.808 | 0.649 | 0.535 |

## Caveats

- **Single seed** for: fedavg, local, pooled. No std over seeds is computable, so the bootstrap CI is the only uncertainty estimate here - and it captures test-set sampling only, not run-to-run variance from initialisation, augmentation, and client order.
- **Small test sets**: centre(s) c4, c5 have under 200 test images (c4=164, c5=88). Their CIs are wide and their point estimates should not be read as measurements.
- Per-centre balanced accuracy also moves epoch to epoch within the converged region, which the bootstrap does not capture: it resamples one fixed set of predictions. Treat a per-centre difference as real only if it clears both the CI and that epoch-to-epoch scatter.

