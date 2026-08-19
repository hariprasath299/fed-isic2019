# Phase 6 aggregate - probe arms

Recomputed from saved weights on cuda; 1000 bootstrap resamples, 95% percentile CIs.

## Runs included

| arm | run | seed | epochs/rounds |
|---|---|---|---|
| fedavg | `fed_probe_fedavg_s0` | 0 | 50 |
| local | `local_probe_s0` | 0 | 20 |
| pooled | `pooled_probe_s0` | 0 | 20 |

## Headline table

Balanced accuracy, final epoch/round, as `mean +/- seed std [bootstrap CI]`.

The two uncertainties measure different things and neither bounds the other. The **std** is run-to-run variation across seeds (initialisation, augmentation draw, client order) and is absent with one seed. The **CI** is test-set sampling, computed from the first seed's predictions - a different draw of test images, same run. A cell can be stable across reruns yet poorly measured because its centre has few test images, or the reverse.

These are **per-cell** uncertainties: each says how precisely that one number is measured. They are **not** a way to compare two columns - the arms share test images, so their intervals share noise and overlap carries no verdict. Comparisons live in the paired section below.

| row | test n | local (routed) | fedavg | pooled | gap-closed |
|---|---|---|---|---|---|
| centre 0 | 2483 | 0.5450 [0.509, 0.585] | 0.5020 [0.465, 0.540] | 0.5349 [0.500, 0.575] | n/a |
| centre 1 | 791 | 0.6253 [0.541, 0.732] | 0.7241 [0.625, 0.869] | 0.7000 [0.584, 0.850] | (132.1%) n.s. |
| centre 2 | 672 | 0.5815 [0.536, 0.718] | 0.5220 [0.468, 0.666] | 0.6639 [0.556, 0.718] | (-72.2%) n.s. |
| centre 3 | 452 | 0.4831 [0.423, 0.549] | 0.4267 [0.369, 0.489] | 0.4555 [0.392, 0.520] | n/a |
| centre 4 | 164 | 0.5882 [0.508, 0.663] | 0.4889 [0.428, 0.552] | 0.4745 [0.400, 0.550] | n/a |
| centre 5 | 88 | 0.4099 [0.337, 0.632] | 0.7333 [0.499, 0.843] | 0.7427 [0.503, 0.861] | (97.2%) n.s. |
| **pooled union** | 4650 | 0.6002 [0.574, 0.627] | 0.5127 [0.488, 0.541] | 0.5638 [0.540, 0.592] | n/a |
| **mean over centres** | 4650 | 0.5388 [0.517, 0.595] | 0.5661 [0.521, 0.608] | 0.5953 [0.538, 0.628] | (48.4%) n.s. |

gap-closed = (fed - local) / (pooled - local), using **fedavg**. It reads as the share of the local-to-pooled headroom that federation recovers at that centre. Values outside 0-100% are meaningful, not errors: **above 100%** means federation beat centralised training there, **negative** means it landed below that centre's own local model. It is **n/a** wherever pooled did not beat local, because the headroom it normalises by is then zero or negative and the ratio stops meaning anything. A value in parentheses marked **n.s.** is worse than n/a: pooled leads local numerically, but by an amount the paired test cannot distinguish from zero, so the quotient is noise divided by noise and its magnitude carries no information.

Local is reported in both aggregations per the Phase 3 policy: routed union **0.6002**, mean over centres **0.5388**. The union is size-weighted, so the largest silo dominates it; the mean weights every centre equally.

## Paired comparisons

Every arm is scored on the **same** test images, so the marginal CIs above share their noise and their overlap is not a test - two arms can differ significantly with overlapping marginals, and can fail to differ with disjoint ones. Each replicate below draws one index resample and scores both arms on it, so the shared component cancels. **A difference is significant iff its CI excludes 0.** Mean-over-centres resamples within each centre so that every centre keeps equal weight.

The estimator is the **difference of seed-means**: each replicate draws one stratified resample, scores every available seed of both arms on it, and differences the seed-means. Seed counts need not match - no seed is discarded and no pairing is invented between unrelated runs.

| comparison | row | delta | 95% CI | significant |
|---|---|---|---|---|
| pooled - local | centre 0 | -0.0101 | [-0.0391, +0.0172] | no |
| pooled - local | centre 1 | +0.0747 | [-0.0794, +0.2433] | no |
| pooled - local | centre 2 | +0.0824 | [-0.1147, +0.1331] | no |
| pooled - local | centre 3 | -0.0276 | [-0.1063, +0.0491] | no |
| pooled - local | centre 4 | -0.1137 | [-0.1950, -0.0308] | **yes** |
| pooled - local | centre 5 | +0.3328 | [-0.0659, +0.4632] | no |
| pooled - local | **pooled union** | -0.0365 | [-0.0629, -0.0097] | **yes** |
| pooled - local | **mean over centres** | +0.0564 | [-0.0361, +0.0908] | no |
| fedavg - local | centre 0 | -0.0430 | [-0.0747, -0.0131] | **yes** |
| fedavg - local | centre 1 | +0.0987 | [-0.0148, +0.2531] | no |
| fedavg - local | centre 2 | -0.0595 | [-0.1272, +0.0051] | no |
| fedavg - local | centre 3 | -0.0565 | [-0.1344, +0.0212] | no |
| fedavg - local | centre 4 | -0.0994 | [-0.1903, -0.0096] | **yes** |
| fedavg - local | centre 5 | +0.3234 | [-0.0736, +0.4431] | no |
| fedavg - local | **pooled union** | -0.0876 | [-0.1133, -0.0608] | **yes** |
| fedavg - local | **mean over centres** | +0.0273 | [-0.0499, +0.0644] | no |
| fedavg - pooled | centre 0 | -0.0329 | [-0.0556, -0.0134] | **yes** |
| fedavg - pooled | centre 1 | +0.0240 | [-0.0277, +0.0976] | no |
| fedavg - pooled | centre 2 | -0.1418 | [-0.1747, +0.0338] | no |
| fedavg - pooled | centre 3 | -0.0289 | [-0.0727, +0.0167] | no |
| fedavg - pooled | centre 4 | +0.0143 | [-0.0489, +0.0791] | no |
| fedavg - pooled | centre 5 | -0.0094 | [-0.0702, +0.0638] | no |
| fedavg - pooled | **pooled union** | -0.0511 | [-0.0685, -0.0349] | **yes** |
| fedavg - pooled | **mean over centres** | -0.0291 | [-0.0498, +0.0112] | no |

7 of 24 comparisons are significant at the 95% level.

### Per-seed deltas (robustness, reported separately)

Deltas computed seed by seed on the real test set, no resampling. Sign consistency asks whether every seed agrees with the direction of the seed-mean delta; it is a descriptor, not a second test, and is only meaningful once both arms have at least two seeds.

| comparison | row | per-seed deltas | sign-consistent |
|---|---|---|---|
| pooled - local | **pooled union** | s0 -0.0365 | n/a (needs 2+ seeds per arm) |
| pooled - local | **mean over centres** | s0 +0.0564 | n/a (needs 2+ seeds per arm) |
| fedavg - local | **pooled union** | s0 -0.0876 | n/a (needs 2+ seeds per arm) |
| fedavg - local | **mean over centres** | s0 +0.0273 | n/a (needs 2+ seeds per arm) |
| fedavg - pooled | **pooled union** | s0 -0.0511 | n/a (needs 2+ seeds per arm) |
| fedavg - pooled | **mean over centres** | s0 -0.0291 | n/a (needs 2+ seeds per arm) |

**Provisional.** Arm(s) fedavg, local, pooled have a single seed. Their run-to-run variance is unmeasured, so any finding involving them is provisional regardless of how narrow its CI is.

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

