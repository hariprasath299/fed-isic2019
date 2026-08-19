# Phase 4 summary — finetune arms A + B (seed 0)

EfficientNet-B0 fine-tuned end to end, LR 1e-4 (sweep winner), 40 epochs,
seed 0. Arms: **pooled** (upper bound) and **local** (lower bound). The
federated arm is deferred, so Phase 4 is not formally complete.

All numbers here are recomputed from the saved `*_final.pt` weights by
`scripts/aggregate_results.py`; the full report is `AGGREGATE_finetune.md`.

## Headline

| row | test n | local (routed) | pooled |
|---|---|---|---|
| centre 0 | 2483 | 0.7823 [0.748, 0.811] | 0.7857 [0.752, 0.817] |
| centre 1 | 791 | 0.7470 [0.679, 0.797] | 0.7042 [0.588, 0.853] |
| centre 2 | 672 | 0.6877 [0.643, 0.838] | 0.6834 [0.633, 0.839] |
| centre 3 | 452 | 0.6082 [0.546, 0.669] | 0.6394 [0.575, 0.703] |
| centre 4 | 164 | 0.6848 [0.611, 0.757] | 0.6980 [0.626, 0.768] |
| centre 5 | 88 | 0.4724 [0.358, 0.802] | 0.7504 [0.484, 0.904] |
| **pooled union** | 4650 | 0.7646 [0.740, 0.788] | 0.7711 [0.745, 0.792] |
| **mean over centres** | 4650 | 0.6637 | 0.7102 |

Pooled clears SPEC §5's `pooled ≈ 0.65+` target and improves **+0.205** over
the frozen-feature probe (0.5638 → 0.7711). Fine-tuning the backbone, not the
head, is where the accuracy is.

## The comparative result: nothing separates the two arms

**0 of 8** paired comparisons are significant at the 95% level.

| row | pooled − local | 95% CI | significant |
|---|---|---|---|
| centre 0 | +0.0035 | [−0.0276, +0.0374] | no |
| centre 1 | −0.0429 | [−0.1589, +0.0839] | no |
| centre 2 | −0.0043 | [−0.0559, +0.0438] | no |
| centre 3 | +0.0312 | [−0.0468, +0.1066] | no |
| centre 4 | +0.0132 | [−0.0557, +0.0820] | no |
| centre 5 | +0.2780 | [−0.1201, +0.4292] | no |
| **pooled union** | +0.0065 | [−0.0201, +0.0304] | no |
| **mean over centres** | +0.0464 | [−0.0288, +0.0866] | no |

On this evidence, at 40 epochs and one seed, **centralised training is not
measurably better than per-silo training on Fed-ISIC2019**, at any centre or
in either aggregation.

### Two earlier claims this retracts

Both came from reading overlap between *marginal* CIs, which is not a test:
the arms are scored on identical test images, so their bootstrap noise is
shared. Overlapping marginals can hide a real difference, and disjoint ones
can appear where none exists. Only the paired interval answers the question.

1. *"Pooled and local are separable at the union level."* They are not. The
   union delta is +0.0065 with CI [−0.0201, +0.0304], comfortably spanning 0.
2. *"c5's +0.278 is the one difference that survives."* It does not survive.
   Its CI is [−0.1201, +0.4292] — the widest in the table, on 88 test images.
   It remains the largest point estimate and the most plausible place for a
   real effect, but it is not established here.

The probe arms show the error runs both ways: `fedavg − local` at centre 0 is
significant under pairing (−0.0430, CI [−0.0761, −0.0114]) while its marginal
CIs, 0.5450 [0.509, 0.582] against 0.5020 [0.466, 0.539], overlap heavily. The
old reading would have discarded a real effect.

## Consequence for the federated arm

Gap-closed % is `(fed − local) / (pooled − local)`. In these arms the
denominator is **+0.0065 at the union with a CI containing 0**. Dividing by a
quantity indistinguishable from zero yields a figure that swings to ±hundreds
of percent on noise. `aggregate_results.py` now marks such cells `n.s.` rather
than printing them as results.

So when the fed arm lands, **gap-closed will not be a reportable metric for
the finetune arms at the union level.** The honest alternatives:

- report the paired deltas `fed − local` and `fed − pooled` directly, which
  need no denominator and are already produced;
- use mean-over-centres, where the headroom is larger (+0.0464) though still
  not significant — check its paired CI before quoting a ratio;
- pool centres into size strata, which is what actually buys test-set power:
  a stratum statistic is computed over more test images than any single small
  centre, so its paired CI is narrower.

Adding seeds does **not** help here, and the two uncertainties must not be
conflated. A bootstrap CI measures **test-set sampling** — how much the number
would move on a different draw of test images. Seeds measure **run-to-run
variance** — how much it would move on a different initialisation,
augmentation draw, and client order. Extra seeds stabilise the point estimate
and let a std be quoted; they leave the test set, and therefore the
test-sampling interval, exactly as it was. Both are needed, and neither
substitutes for the other.

## Convergence

Extending 20 → 40 epochs gained pooled **+0.0021** (0.7690 → 0.7711). The
"still climbing at +0.93 pts/epoch" reading at epoch 20 was noise: ep 1–20 has
sd 0.0505, so a 5-epoch slope cannot separate trend from scatter there.
ep 21–40 has sd 0.0092 and slope −0.04 pts/epoch. **Pooled converges near
0.76–0.77 by roughly epoch 20.**

Local gained more (mean 0.6553 → 0.6637) and every silo's best epoch landed at
29–39, so the extra epochs were worth more to the small silos than to pooled.

SPEC's "not still climbing at the end" check needs a window sized against the
curve's own noise, or it keeps recommending epochs that buy nothing.

## Standing caveats

- **One seed.** The bootstrap CIs cover test-set sampling only — not
  initialisation, augmentation order, or client order. Run-to-run variance is
  unmeasured, so every interval here is narrower than the true uncertainty.
- **Small test sets.** c4 (164) and c5 (88) dominate the width of the
  mean-over-centres statistic. c5 alone spans 0.44.
- **Epoch-to-epoch scatter** within the converged region reaches 0.20 on c2
  and 0.14 on c1. The bootstrap does not capture it — it resamples one fixed
  set of predictions from one epoch. A per-centre difference should clear both
  the paired CI and that scatter before it is called real.
- **Resume discontinuity.** Neither arm saves optimizer state, so Adam's
  moments reset once at epoch 20 for both arms equally. The 40-epoch curves
  are not identical to one continuous 40-epoch run.
- ~~**LR at the grid edge.**~~ Retired 2026-08-19: a 3e-5 point was added and
  scored 0.6165, bracketing 1e-4 as an interior maximum. See the LR decision
  log below.

---

# Addendum — pre-registration for the federated arm (2026-08-19)

Written **before any federated finetune number exists**, so the analysis
cannot be shaped by the result. Anything not fixed here is exploratory and
must be labelled as such in the writeup.

## Primary endpoints

Paired bootstrap deltas, 95% CI, significance = CI excludes 0:

- `fed − local`
- `fed − pooled`

evaluated at:

1. **mean over centres** — every centre weighted equally, stratified
   resampling within centre;
2. **per size stratum**, strata fixed now by training-set size:

| stratum | centres | train n | test n |
|---|---|---|---|
| small | c4, c5 | 655 + 351 = 1006 | 164 + 88 = **252** |
| mid | c2, c3 | 2691 + 1807 = 4498 | 672 + 452 = **1124** |
| large | c0, c1 | 9930 + 3163 = 13093 | 2483 + 791 = **3274** |

The stratum statistic is balanced accuracy over the stratum's **pooled test
images**, with the paired bootstrap **resampling within centre**. Pooling the
images is what buys the power — 252 images for the small stratum instead of 88
for c5 alone — while resampling within centre keeps each centre's contribution
to the resample proportional to its own test set rather than letting one
centre's draw stand in for the stratum.

Per-centre deltas remain reported, as secondary and descriptive. c5 alone
(88 images) will not support a primary claim.

## Hypotheses on record

- **H1**: `fed − local > 0` in the **small** stratum. Federation should help
  the centres that cannot train a good model alone. c5 has the largest point
  estimate in every arm measured so far, and c4 holds only 3 of 8 classes, so
  it can only gain classes it has never seen.
- **H2**: `fed − local ≤ 0` in the **large** stratum. A large silo already has
  enough data, and averaging drags it toward centres with different
  distributions. The probe arms support this: `fedavg − local` at c0 was
  significantly negative (−0.0430, CI [−0.0761, −0.0114]).

A result contradicting either is reportable as such. These are recorded to
prevent the direction being chosen after seeing the numbers.

## gap-closed

**Descriptive only. Never a primary claim.** Quoted only where the paired
`pooled − local` CI at that row excludes 0, since it divides by that headroom.
In the current finetune arms it does not (union delta +0.0065, CI
[−0.0201, +0.0304]), so gap-closed is expected to be unreportable at the union
level. `aggregate_results.py` enforces this: such cells print `(x%) n.s.`

## Protocol held fixed

LR = the locked sweep winner; batch 64; Adam; focal loss gamma 2.0, alpha from
pooled counts; 50 rounds x 100 local steps; eval every round; seed 0 first,
then 1 and 2. Extension past 50 rounds only via `--resume`, and only if the
curve is still climbing by a window sized against its own noise — the ep-20
lesson above, where a 5-epoch slope inside sd 0.0505 scatter read as a trend
that 20 more epochs showed was worth +0.0021.

## Numbers from external sources

Any FLamby comparison number is quoted from their paper at the point of use,
with the reference. This repo's README contains none. Nothing is quoted from
memory.

## LR decision log (2026-08-19, 22:41)

The winner of the original {1e-4, 5e-4, 1e-3} grid sat at the grid's edge, so
a fourth point was run below it. Protocol identical to the other three: pooled
finetune, 5 epochs, eval every epoch, seed 0, batch 64, Adam, focal gamma 2.0
with pooled alphas, bf16 AMP.

| LR | bal_acc_pooled @ ep 5 |
|---|---|
| 3e-5 | 0.616538 |
| **1e-4** | **0.690992** |
| 5e-4 | 0.670196 |
| 1e-3 | 0.646804 |

**Branch taken: clear loss.** 3e-5 scored 0.6165 against 1e-4's 0.6910, a
margin of **−0.0745**, well below the −0.0100 clear-loss threshold. No
tie-break at 15 epochs was needed. **LR is locked at 1e-4.**

### The grid-edge caveat is retired

1e-4 is no longer the smallest value tried, and the four points bracket it:
0.6165 < **0.6910** > 0.6702 > 0.6469. It is an interior maximum of the
searched range, not an endpoint, so the concern that the optimum might lie
below the grid is resolved rather than merely noted. The corresponding entry
under "Standing caveats" no longer applies.

### Unplanned reproducibility check

`pooled_lrsweep_1e4` and `pooled_finetune_s0` are independent runs sharing LR,
seed, and protocol. Their epoch-5 values agree to every digit recorded:
**0.690992** in both. Same-seed runs are reproducing bit-for-bit on this
machine, which means the seed-variance analysis for s1/s2 will measure real
run-to-run variation from the seed, not incidental nondeterminism on top of it.

---

# Pre-registration amendment — seed aggregation for paired endpoints (2026-08-20)

Amends the 2026-08-19 pre-registration. Written before the D1 session starts
and before any federated finetune number exists. **This freezes the analysis:
after this, the only thing that changes before D1 is seed 2 finishing.**

## Why an amendment was needed

The original pre-registration fixed the paired estimator but not how it
combines seeds. When seed 1 landed, the report was averaging seeds in the
headline while the paired section silently used the first seed only — half the
data discarded, and two different meanings for the same arm on one page. D1
makes it worse: one fed seed against two or three pooled/local seeds.

## Estimator (primary endpoints)

**Difference of seed-means under a shared-resample paired bootstrap.**

Per replicate:

1. draw **one** stratified test-index resample — independently within each
   centre, so every centre keeps its own size;
2. score **every available seed of both arms** on that same draw;
3. delta = `mean-over-seeds(A) − mean-over-seeds(B)`;
4. CI = percentile over replicates. Significant iff it excludes 0.

Sharing the draw cancels the test-sampling noise the arms have in common, since
they are evaluated on identical images — that is what makes the interval a test
rather than a description. Unequal seed counts are handled natively: each arm
contributes the mean of whatever seeds it has. No seed is discarded and no
pairing is invented between unrelated runs.

Headline cells use the identical path: `mean ± seed-std [CI of the seed-mean]`.
The same number means the same thing everywhere in the report.

One definitional consequence, recorded: the union row is now resampled
**stratified by centre** rather than as one undifferentiated pool, because all
rows share a single draw. This preserves each centre's proportion in every
replicate.

## Reported alongside, never merged into the CI

- **per-arm seed std** of the metric;
- **per-seed deltas**, listed individually;
- **sign consistency** across seeds — a robustness descriptor, not a second
  test, and only meaningful once both arms have ≥ 2 seeds;
- any finding involving a **single-seed arm is labelled provisional**
  regardless of how narrow its CI is. At D1 the fed arm will have one seed, so
  every fed finding is provisional by this rule.

## What it changes in the current two-seed table

| row | pooled − local (seed-mean) | 95% CI | per-seed | sign-consistent |
|---|---|---|---|---|
| pooled union | **−0.0004** | [−0.0193, +0.0175] | s0 +0.0065, s1 −0.0073 | **no** |
| mean over centres | +0.0543 | [−0.0208, +0.0829] | s0 +0.0464, s1 +0.0622 | yes |

The union delta **changes sign between seeds** and the seed-mean is −0.0004 —
pooled and local are not merely indistinguishable there, they swap order on a
rerun. Still 0 of 8 comparisons significant. The mean-over-centres delta is
sign-consistent and the larger effect, but its CI includes 0 too.

This is precisely the robustness question the CI cannot answer, which is why
sign consistency is reported next to it rather than folded into it.

## Tests pinning the amendment (23/23)

- a single-seed seed-mean delta reproduces the previous single-seed paired
  delta **exactly**, so the amendment does not move the existing answer;
- unequal seed counts run, and dropping a seed changes the result — proof every
  seed is used rather than one being silently preferred;
- an arm compared against itself returns exactly `[0, 0]` on every row at any
  seed count, which only holds because the draw is shared;
- the headline seed-mean and the paired point estimate agree by construction.
