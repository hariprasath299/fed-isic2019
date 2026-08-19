# Phase 6 aggregate - finetune arms

Recomputed from saved weights on cuda; 1000 bootstrap resamples, 95% percentile CIs.

## Runs included

| arm | run | seed | epochs/rounds |
|---|---|---|---|
| local | `local_finetune_s0` | 0 | 40 |
| local | `local_finetune_s1` | 1 | 40 |
| pooled | `pooled_finetune_s0` | 0 | 40 |
| pooled | `pooled_finetune_s1` | 1 | 40 |

Skipped (no usable checkpoint):

- `pooled_finetune_s2` - missing checkpoint(s): pooled_finetune_s2_final.pt

## Headline table

Balanced accuracy, final epoch/round, as `mean +/- seed std [bootstrap CI]`.

The two uncertainties measure different things and neither bounds the other. The **std** is run-to-run variation across seeds (initialisation, augmentation draw, client order) and is absent with one seed. The **CI** is test-set sampling, computed from the first seed's predictions - a different draw of test images, same run. A cell can be stable across reruns yet poorly measured because its centre has few test images, or the reverse.

Union-row CIs are **conditional on the test set's centre composition**: since the 2026-08-20 amendment every row shares one draw resampled *within* centre, so the union holds each centre's proportion fixed instead of resampling it. Small differences against pre-amendment AGGREGATE files are that scheme change, not new data.

These are **per-cell** uncertainties: each says how precisely that one number is measured. They are **not** a way to compare two columns - the arms share test images, so their intervals share noise and overlap carries no verdict. Comparisons live in the paired section below.

| row | test n | local (routed) | pooled | gap-closed |
|---|---|---|---|---|
| centre 0 | 2483 | 0.7736 +/- 0.0122 [0.747, 0.801] | 0.7846 +/- 0.0017 [0.757, 0.811] | n/a |
| centre 1 | 791 | 0.7366 +/- 0.0148 [0.667, 0.786] | 0.6884 +/- 0.0223 [0.602, 0.792] | n/a |
| centre 2 | 672 | 0.6922 +/- 0.0063 [0.650, 0.843] | 0.7428 +/- 0.0840 [0.698, 0.820] | n/a |
| centre 3 | 452 | 0.6253 +/- 0.0242 [0.564, 0.683] | 0.6187 +/- 0.0293 [0.565, 0.669] | n/a |
| centre 4 | 164 | 0.6636 +/- 0.0300 [0.592, 0.735] | 0.6837 +/- 0.0202 [0.622, 0.742] | n/a |
| centre 5 | 88 | 0.4551 +/- 0.0244 [0.350, 0.766] | 0.7541 +/- 0.0053 [0.529, 0.903] | n/a |
| **stratum: small** (c4, c5) | 252 | 0.5284 +/- 0.0230 [0.486, 0.751] | 0.7812 +/- 0.0109 [0.669, 0.819] | n/a |
| **stratum: mid** (c2, c3) | 1124 | 0.7094 +/- 0.0264 [0.672, 0.748] | 0.6914 +/- 0.0144 [0.655, 0.727] | n/a |
| **stratum: large** (c0, c1) | 3274 | 0.7928 +/- 0.0098 [0.767, 0.817] | 0.7977 +/- 0.0023 [0.773, 0.822] | n/a |
| **pooled union** | 4650 | 0.7662 +/- 0.0024 [0.745, 0.788] | 0.7658 +/- 0.0074 [0.746, 0.785] | n/a |
| **mean over centres** | 4650 | 0.6577 +/- 0.0085 [0.633, 0.728] | 0.7120 +/- 0.0026 [0.669, 0.748] | n/a |

**Stratum rows** are the pre-registered primary endpoints (2026-08-19), fixed by training-set size before any federated number existed. A stratum is scored over its centres' pooled test images - the same centre-subset statistic that produces the per-centre rows (singleton subset) and the union row (all centres), so the three agree by construction. Pooling is what buys test-set power: 252 images for `small` against 88 for c5 alone.

Local is reported in both aggregations per the Phase 3 policy: routed union **0.7662**, mean over centres **0.6577**. The union is size-weighted, so the largest silo dominates it; the mean weights every centre equally.

## Paired comparisons

Every arm is scored on the **same** test images, so the marginal CIs above share their noise and their overlap is not a test - two arms can differ significantly with overlapping marginals, and can fail to differ with disjoint ones. Each replicate below draws one index resample and scores both arms on it, so the shared component cancels. **A difference is significant iff its CI excludes 0.** Mean-over-centres resamples within each centre so that every centre keeps equal weight.

The estimator is the **difference of seed-means**: each replicate draws one stratified resample, scores every available seed of both arms on it, and differences the seed-means. Seed counts need not match - no seed is discarded and no pairing is invented between unrelated runs.

| comparison | row | delta | 95% CI | significant |
|---|---|---|---|---|
| pooled - local | centre 0 | +0.0109 | [-0.0146, +0.0336] | no |
| pooled - local | centre 1 | -0.0482 | [-0.1348, +0.0372] | no |
| pooled - local | centre 2 | +0.0506 | [-0.0522, +0.0780] | no |
| pooled - local | centre 3 | -0.0066 | [-0.0608, +0.0480] | no |
| pooled - local | centre 4 | +0.0201 | [-0.0461, +0.0816] | no |
| pooled - local | centre 5 | +0.2990 | [-0.0707, +0.4442] | no |
| pooled - local | **stratum: small** | +0.2528 | [-0.0420, +0.2883] | no |
| pooled - local | **stratum: mid** | -0.0180 | [-0.0563, +0.0168] | no |
| pooled - local | **stratum: large** | +0.0050 | [-0.0177, +0.0262] | no |
| pooled - local | **pooled union** | -0.0004 | [-0.0193, +0.0175] | no |
| pooled - local | **mean over centres** | +0.0543 | [-0.0208, +0.0829] | no |

0 of 11 comparisons are significant at the 95% level.

### Per-seed deltas (robustness, reported separately)

Deltas computed seed by seed on the real test set, no resampling. Sign consistency asks whether every seed agrees with the direction of the seed-mean delta; it is a descriptor, not a second test, and is only meaningful once both arms have at least two seeds.

| comparison | row | per-seed deltas | sign-consistent |
|---|---|---|---|
| pooled - local | **stratum: small** | s0 +0.2443, s1 +0.2614 | **yes** |
| pooled - local | **stratum: mid** | s0 +0.0108, s1 -0.0469 | no - seeds disagree in sign |
| pooled - local | **stratum: large** | s0 -0.0004, s1 +0.0103 | no - seeds disagree in sign |
| pooled - local | **mean over centres** | s0 +0.0464, s1 +0.0622 | **yes** |
| pooled - local | **pooled union** | s0 +0.0065, s1 -0.0073 | no - seeds disagree in sign |

## Rare-class recall (< 2% of pooled train)

Balanced accuracy averages these away; a model can look fine overall and still miss them entirely.

| arm | class 5 (1.0%) | class 6 (1.1%) | mean rare | mean common |
|---|---|---|---|---|
| local (routed) | 0.700 | 0.952 | 0.826 | 0.746 |
| pooled | 0.691 | 0.923 | 0.807 | 0.752 |

## Caveats

- **No federated arm present**, so gap-closed % is undefined: it measures how much of the pooled-minus-local headroom federation recovers, and there is nothing here to recover it. The pooled and local columns are the two endpoints only.
- **Small test sets**: centre(s) c4, c5 have under 200 test images (c4=164, c5=88). Their CIs are wide and their point estimates should not be read as measurements.
- Per-centre balanced accuracy also moves epoch to epoch within the converged region, which the bootstrap does not capture: it resamples one fixed set of predictions. Treat a per-centre difference as real only if it clears both the CI and that epoch-to-epoch scatter.

