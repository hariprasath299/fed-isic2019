# Phase 6 aggregate - finetune arms

Recomputed from saved weights on cuda; 1000 bootstrap resamples, 95% percentile CIs.

## Runs included

| arm | run | seed | epochs/rounds |
|---|---|---|---|
| local | `local_finetune_s0` | 0 | 40 |
| local | `local_finetune_s1` | 1 | 40 |
| local | `local_finetune_s2` | 2 | 40 |
| pooled | `pooled_finetune_s0` | 0 | 40 |
| pooled | `pooled_finetune_s1` | 1 | 40 |
| pooled | `pooled_finetune_s2` | 2 | 40 |

## Headline table

Balanced accuracy, final epoch/round, as `mean +/- seed std [bootstrap CI]`.

The two uncertainties measure different things and neither bounds the other. The **std** is run-to-run variation across seeds (initialisation, augmentation draw, client order) and is absent with one seed. The **CI** is test-set sampling, computed from the first seed's predictions - a different draw of test images, same run. A cell can be stable across reruns yet poorly measured because its centre has few test images, or the reverse.

Union-row CIs are **conditional on the test set's centre composition**: since the 2026-08-20 amendment every row shares one draw resampled *within* centre, so the union holds each centre's proportion fixed instead of resampling it. Small differences against pre-amendment AGGREGATE files are that scheme change, not new data.

These are **per-cell** uncertainties: each says how precisely that one number is measured. They are **not** a way to compare two columns - the arms share test images, so their intervals share noise and overlap carries no verdict. Comparisons live in the paired section below.

| row | test n | local (routed) | pooled | gap-closed |
|---|---|---|---|---|
| centre 0 | 2483 | 0.7849 +/- 0.0214 [0.759, 0.809] | 0.7940 +/- 0.0163 [0.769, 0.817] | n/a |
| centre 1 | 791 | 0.7384 +/- 0.0109 [0.688, 0.778] | 0.6840 +/- 0.0175 [0.611, 0.775] | n/a |
| centre 2 | 672 | 0.7359 +/- 0.0758 [0.695, 0.839] | 0.7723 +/- 0.0784 [0.726, 0.826] | n/a |
| centre 3 | 452 | 0.6340 +/- 0.0228 [0.573, 0.688] | 0.6283 +/- 0.0266 [0.581, 0.676] | n/a |
| centre 4 | 164 | 0.6651 +/- 0.0213 [0.596, 0.733] | 0.6880 +/- 0.0161 [0.631, 0.745] | n/a |
| centre 5 | 88 | 0.4609 +/- 0.0199 [0.360, 0.768] | 0.7610 +/- 0.0125 [0.529, 0.903] | n/a |
| **stratum: small** (c4, c5) | 252 | 0.5311 +/- 0.0169 [0.490, 0.751] | 0.7857 +/- 0.0110 [0.676, 0.822] | n/a |
| **stratum: mid** (c2, c3) | 1124 | 0.7161 +/- 0.0220 [0.679, 0.754] | 0.7007 +/- 0.0191 [0.667, 0.735] | n/a |
| **stratum: large** (c0, c1) | 3274 | 0.8032 +/- 0.0194 [0.781, 0.825] | 0.8051 +/- 0.0129 [0.783, 0.827] | n/a |
| **pooled union** | 4650 | 0.7753 +/- 0.0158 [0.756, 0.795] | 0.7736 +/- 0.0144 [0.754, 0.792] | n/a |
| **mean over centres** | 4650 | 0.6699 +/- 0.0218 [0.647, 0.733] | 0.7213 +/- 0.0161 [0.678, 0.753] | n/a |

**Stratum rows** are the pre-registered primary endpoints (2026-08-19), fixed by training-set size before any federated number existed. A stratum is scored over its centres' pooled test images - the same centre-subset statistic that produces the per-centre rows (singleton subset) and the union row (all centres), so the three agree by construction. Pooling is what buys test-set power: 252 images for `small` against 88 for c5 alone.

Local is reported in both aggregations per the Phase 3 policy: routed union **0.7753**, mean over centres **0.6699**. The union is size-weighted, so the largest silo dominates it; the mean weights every centre equally.

## Paired comparisons

Every arm is scored on the **same** test images, so the marginal CIs above share their noise and their overlap is not a test - two arms can differ significantly with overlapping marginals, and can fail to differ with disjoint ones. Each replicate below draws one index resample and scores both arms on it, so the shared component cancels. **A difference is significant iff its CI excludes 0.** Mean-over-centres resamples within each centre so that every centre keeps equal weight.

The estimator is the **difference of seed-means**: each replicate draws one stratified resample, scores every available seed of both arms on it, and differences the seed-means. Seed counts need not match - no seed is discarded and no pairing is invented between unrelated runs.

| comparison | row | delta | 95% CI | significant |
|---|---|---|---|---|
| pooled - local | centre 0 | +0.0090 | [-0.0121, +0.0292] | no |
| pooled - local | centre 1 | -0.0544 | [-0.1326, +0.0135] | no |
| pooled - local | centre 2 | +0.0365 | [-0.0359, +0.0586] | no |
| pooled - local | centre 3 | -0.0057 | [-0.0491, +0.0377] | no |
| pooled - local | centre 4 | +0.0229 | [-0.0330, +0.0755] | no |
| pooled - local | centre 5 | +0.3001 | [-0.0429, +0.4530] | no |
| pooled - local | **stratum: small** | +0.2547 | [-0.0314, +0.2872] | no |
| pooled - local | **stratum: mid** | -0.0154 | [-0.0477, +0.0145] | no |
| pooled - local | **stratum: large** | +0.0019 | [-0.0183, +0.0218] | no |
| pooled - local | **pooled union** | -0.0017 | [-0.0179, +0.0144] | no |
| pooled - local | **mean over centres** | +0.0514 | [-0.0163, +0.0765] | no |

0 of 11 comparisons are significant at the 95% level.

### Per-seed deltas (robustness, reported separately)

Deltas computed seed by seed on the real test set, no resampling. Sign consistency asks whether every seed agrees with the direction of the seed-mean delta; it is a descriptor, not a second test, and is only meaningful once both arms have at least two seeds.

| comparison | row | per-seed deltas | sign-consistent |
|---|---|---|---|
| pooled - local | **stratum: small** | s0 +0.2443, s1 +0.2614, s2 +0.2584 | **yes** |
| pooled - local | **stratum: mid** | s0 +0.0108, s1 -0.0469, s2 -0.0101 | no - seeds disagree in sign |
| pooled - local | **stratum: large** | s0 -0.0004, s1 +0.0103, s2 -0.0043 | no - seeds disagree in sign |
| pooled - local | **mean over centres** | s0 +0.0464, s1 +0.0622, s2 +0.0456 | **yes** |
| pooled - local | **pooled union** | s0 +0.0065, s1 -0.0073, s2 -0.0044 | no - seeds disagree in sign |

## Rare-class recall (< 2% of pooled train)

Balanced accuracy averages these away; a model can look fine overall and still miss them entirely.

| arm | class 5 (1.0%) | class 6 (1.1%) | mean rare | mean common |
|---|---|---|---|---|
| local (routed) | 0.733 | 0.955 | 0.844 | 0.752 |
| pooled | 0.727 | 0.936 | 0.832 | 0.754 |

## Caveats

- **No federated arm present**, so gap-closed % is undefined: it measures how much of the pooled-minus-local headroom federation recovers, and there is nothing here to recover it. The pooled and local columns are the two endpoints only.
- **Small test sets**: centre(s) c4, c5 have under 200 test images (c4=164, c5=88). Their CIs are wide and their point estimates should not be read as measurements.
- Per-centre balanced accuracy also moves epoch to epoch within the converged region, which the bootstrap does not capture: it resamples one fixed set of predictions. Treat a per-centre difference as real only if it clears both the CI and that epoch-to-epoch scatter.

