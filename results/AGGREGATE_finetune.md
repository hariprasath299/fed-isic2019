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

These are **per-cell** uncertainties: each says how precisely that one number is measured. They are **not** a way to compare two columns - the arms share test images, so their intervals share noise and overlap carries no verdict. Comparisons live in the paired section below.

| row | test n | local (routed) | pooled | gap-closed |
|---|---|---|---|---|
| centre 0 | 2483 | 0.7736 +/- 0.0122 [0.748, 0.811] | 0.7846 +/- 0.0017 [0.752, 0.817] | n/a |
| centre 1 | 791 | 0.7366 +/- 0.0148 [0.679, 0.797] | 0.6884 +/- 0.0223 [0.588, 0.853] | n/a |
| centre 2 | 672 | 0.6922 +/- 0.0063 [0.643, 0.838] | 0.7428 +/- 0.0840 [0.633, 0.839] | n/a |
| centre 3 | 452 | 0.6253 +/- 0.0242 [0.546, 0.669] | 0.6187 +/- 0.0293 [0.575, 0.703] | n/a |
| centre 4 | 164 | 0.6636 +/- 0.0300 [0.611, 0.757] | 0.6837 +/- 0.0202 [0.626, 0.768] | n/a |
| centre 5 | 88 | 0.4551 +/- 0.0244 [0.358, 0.802] | 0.7541 +/- 0.0053 [0.484, 0.904] | n/a |
| **pooled union** | 4650 | 0.7662 +/- 0.0024 [0.740, 0.788] | 0.7658 +/- 0.0074 [0.745, 0.792] | n/a |
| **mean over centres** | 4650 | 0.6577 +/- 0.0085 | 0.7120 +/- 0.0026 | n/a |

Local is reported in both aggregations per the Phase 3 policy: routed union **0.7662**, mean over centres **0.6577**. The union is size-weighted, so the largest silo dominates it; the mean weights every centre equally.

## Paired comparisons

Every arm is scored on the **same** test images, so the marginal CIs above share their noise and their overlap is not a test - two arms can differ significantly with overlapping marginals, and can fail to differ with disjoint ones. Each replicate below draws one index resample and scores both arms on it, so the shared component cancels. **A difference is significant iff its CI excludes 0.** Mean-over-centres resamples within each centre so that every centre keeps equal weight.

Where an arm has several seeds the first is used, since pairing requires one prediction vector per arm.

| comparison | row | delta | 95% CI | significant |
|---|---|---|---|---|
| pooled - local | centre 0 | +0.0035 | [-0.0276, +0.0374] | no |
| pooled - local | centre 1 | -0.0429 | [-0.1589, +0.0839] | no |
| pooled - local | centre 2 | -0.0043 | [-0.0559, +0.0438] | no |
| pooled - local | centre 3 | +0.0312 | [-0.0468, +0.1066] | no |
| pooled - local | centre 4 | +0.0132 | [-0.0557, +0.0820] | no |
| pooled - local | centre 5 | +0.2780 | [-0.1201, +0.4292] | no |
| pooled - local | **pooled union** | +0.0065 | [-0.0201, +0.0304] | no |
| pooled - local | **mean over centres** | +0.0464 | [-0.0288, +0.0866] | no |

0 of 8 comparisons are significant at the 95% level.

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

