# PROJECT LOG — Fed-ISIC2019 benchmark

The work record: what was done, why, what was wrong, and what is still open.
**No number tables** — every figure lives in `results/RESULTS.md`, cited here by
section. Phase narratives are in `results/PHASE3_SUMMARY.md` and
`results/PHASE4_SUMMARY.md`.

---

## 1. Question and design

Does federated training recover the accuracy advantage of centralised training
on Fed-ISIC2019, and **for whom**? Six dermoscopy centres hold 18,597 training
images split 28:1 between the largest and smallest silo, across 8 diagnosis
classes whose prevalence spans 49% to under 1%. Three arms bound the answer:
**pooled** (all data centrally, the upper bound), **local** (each centre alone,
the lower bound), and **federated** (the question). The metric is balanced
accuracy, because a 49%-prevalence majority class makes plain accuracy
clinically meaningless.

**"% of gap closed" was demoted to descriptive-only.** It is
`(fed − local) / (pooled − local)`, so it divides by the pooled-minus-local
headroom. On the finetune arms that headroom is not distinguishable from zero
(RESULTS §7), and dividing by a quantity whose CI spans 0 produces a figure
that swings to ±hundreds of percent on noise — it stops meaning "share of the
headroom recovered". The paired deltas `fed − local` and `fed − pooled` are
reported directly instead; they need no denominator. `aggregate_results.py`
enforces this by printing `n.s.` rather than a number when the headroom is not
significant.

---

## 2. Phase log

| phase | what ran | acceptance criterion | result | date | produced |
|---|---|---|---|---|---|
| 0 data | `phase0_verify_data.py` | exit 0; per-centre counts match FLamby | **pass** — counts confirmed (RESULTS §3) | 2026-08-19 | `inspection/` |
| 1 features | `phase1_cache_features.py` | `train.npz` (18597, 1280), `test.npz` (4650, 1280) | **pass** — shapes and counts exact, no NaN | 2026-08-19 | `data/features/` |
| 2 harness | `pytest tests/` | 11/11 on the target machine | **pass** — now 26/26 after later additions | 2026-08-19 | — |
| 3 probe arms | pooled / local / fed, linear probe on frozen features | ordering local ≤ fed ≤ pooled; no degenerate arm | **pass** — ordering held, 8/8 classes predicted (RESULTS §4) | 2026-08-19 | `PHASE3_SUMMARY.md` |
| 4 finetune A+B | pooled + local, full fine-tuning, seeds 0/1/2 | pooled ≈ 0.65+; curves not still climbing | **pass** — pooled well above target; convergence confirmed by slope test (RESULTS §5, §8) | 2026-08-19 → 2026-08-20 | `PHASE4_SUMMARY.md` |
| 4 finetune C | federated arm | FedAvg in 0.59–0.66 | **not run** — packaged in `D1_COLAB.md` | pending | — |

Phase 4 is **not** formally complete: its acceptance criterion is stated on
FedAvg's number, and gap-closed has no meaning without all three arms.

---

## 3. Decision log

**2026-08-19 — Dataset: Fed-ISIC2019**, chosen over Fed-Heart-Disease and
SynDelay. *Evidence:* the rationale is **not recorded anywhere in this repo** —
neither alternative is mentioned in `SPEC.md`, the README, or any commit. The
decision predates the recorded history, so the reasoning is **UNVERIFIED
here**. What the repo does support is that Fed-ISIC2019 supplies the properties
the question needs: a real 28:1 silo-size skew, natural non-IID structure from
imaging device and site, and a published FLamby baseline to compare against.
*Changed:* everything downstream.

**2026-08-19 — Final-epoch reporting, never best-epoch.** *Evidence:* there is
no validation split and the budget is fixed in advance, so selecting the best
round on the test set would be test-set model selection. *Changed:* every
headline number is the final epoch/round; peak values are quarantined in
RESULTS §10 under an explicit not-reportable header, and no other section may
cite them. This survived direct pressure later — when drift looked severe, the
remedy was explicitly **not** to switch to best-epoch reporting.

**2026-08-19 — LR locked at 1e-4.** *Evidence:* a three-point sweep
{1e-4, 5e-4, 1e-3} put the winner at the grid edge, so a fourth point at 3e-5
was run and scored well below it (RESULTS §2). The four points bracket 1e-4,
making it an interior maximum rather than an endpoint. *Changed:* retired the
grid-edge caveat outright instead of carrying it into the writeup, and locked
the LR for every arm including the federated one.

**2026-08-20 — s0 keeps its Adam reset; no clean rerun.** s0 reached 40 epochs
by resuming at epoch 20, before optimizer state was saved, so its Adam moments
restarted once mid-run; s1 and s2 are continuous. *Evidence:* across three
seeds s0 is the **middle value** on both pooled union and local
mean-over-centres (RESULTS §5, §6) — bracketed by the other two on both, so
nothing marks it anomalous. *Changed:* nothing; the contingency did not fire.
Recorded as the weak claim it is, since three points cannot establish much.

**2026-08-20 — Protocol left unchanged after the drift investigation.**
Final-vs-peak drift exceeded 1 point on 17 of 21 curves, which looked like a
convergence failure. *Evidence:* a slope test over the final half of training
found only 2 of 21 curves declining, while 8 were significantly **improving**
(RESULTS §8); and drift correlates −0.81 with each centre's metric quantum
against only +0.44 with log train size (RESULTS §3, §7). *Changed:* no
schedule, constant, or script was touched — the drift **rule** was amended
instead (§5 below). Acting on the original reading would have applied cosine
decay to fix what was quantisation noise.

**2026-08-20 — Repository visibility.** The remote
`hariprasath299/fed-isic2019` currently reads as **private**: an unauthenticated
GitHub API request returns 404. A decision to make it public is **not evidenced
at the time of writing**, and is recorded here as not-yet-done rather than as a
completed decision. `D1_COLAB.md` accordingly clones with a token via `getpass`
and strips it from `.git/config` afterwards.

---

## 4. Retraction log

**(a) Pooled-vs-local significance read from marginal-CI overlap.**
*Claimed:* that the arms were "separable at the union level", and that c5's
+0.278 was "the one difference that survives". *Wrong because:* both arms are
scored on the **same** test images, so their bootstrap noise is shared and the
overlap of two marginal intervals is not a test — in either direction.
*Replaced by:* a paired bootstrap — one resample per replicate, both arms
scored on it, CI on the difference, significant iff it excludes 0 (RESULTS §7).
Under the correct test nothing separated the arms. The probe data showed the
error runs both ways: `fedavg − local` at c0 is significant when paired while
its marginal intervals overlap heavily, so the old method was also discarding
real effects, not only inventing them.

**(b) "Drift scales inversely with train size."** *Claimed:* that the local
arm's larger drift tracked silo size. *Wrong because:* it does not hold at
centre level — c3 (1807 train images) drifts −1.29 points while c1 (3163,
nearly twice the data) drifts −6.06; the correlation with log train size is
only +0.44 (RESULTS §3). *Replaced by:* drift scales with **metric resolution**
(corr −0.81), which merely correlates with silo size.

**(c) The balanced-accuracy quantum denominator.** *Claimed:*
`1 / (8 × min_class_count)`. *Wrong because:* sklearn's
`balanced_accuracy_score` averages recall over the classes **present in
`y_true`**, not over all 8. *Replaced by:*
`1 / (classes_present × min_class_count)`, which doubles c5's quantum from 12.5
to **25 points** (RESULTS §3). The correlation with drift is essentially
unchanged, but c5's resolution ceiling is twice as severe as first stated.

**Withdrawn: the n=3 z-score outlier test.** A z-score was quoted to argue s0
was not an outlier. At n=3 the maximum attainable |z| is `(n−1)/√n = 1.155`, so
no observation can ever reach a conventional threshold — the test is
structurally incapable of detecting an outlier, and quoting one implied a test
that was never performed. The conclusion was re-derived from order statistics
instead (§3 above).

**Also corrected in passing:** a claim of "no transient at the resume seam" was
asserted without the comparison that would support it. Measured, the seam step
sits inside the curve's own epoch-to-epoch scatter — so a transient of that size
*cannot be detected*, which is a weaker statement than none occurred.

---

## 5. Pre-registration amendments

Two, both dated, both made **before any federated result existed**. The analysis
is frozen at tag `d1-frozen`.

### Amendment 1 — commit `d7235ea`

Evidence state when made: pooled and local arms at seeds 0 and 1; no federated
finetune run; seed 2 not yet started.

> # Pre-registration amendment — seed aggregation for paired endpoints (2026-08-20)
>
> Amends the 2026-08-19 pre-registration. Written before the D1 session starts
> and before any federated finetune number exists. **This freezes the analysis:
> after this, the only thing that changes before D1 is seed 2 finishing.**
>
> ## Why an amendment was needed
>
> The original pre-registration fixed the paired estimator but not how it
> combines seeds. When seed 1 landed, the report was averaging seeds in the
> headline while the paired section silently used the first seed only — half the
> data discarded, and two different meanings for the same arm on one page. D1
> makes it worse: one fed seed against two or three pooled/local seeds.
>
> ## Estimator (primary endpoints)
>
> **Difference of seed-means under a shared-resample paired bootstrap.**
>
> Per replicate:
>
> 1. draw **one** stratified test-index resample — independently within each
>    centre, so every centre keeps its own size;
> 2. score **every available seed of both arms** on that same draw;
> 3. delta = `mean-over-seeds(A) − mean-over-seeds(B)`;
> 4. CI = percentile over replicates. Significant iff it excludes 0.
>
> Sharing the draw cancels the test-sampling noise the arms have in common, since
> they are evaluated on identical images — that is what makes the interval a test
> rather than a description. Unequal seed counts are handled natively: each arm
> contributes the mean of whatever seeds it has. No seed is discarded and no
> pairing is invented between unrelated runs.
>
> Headline cells use the identical path: `mean ± seed-std [CI of the seed-mean]`.
> The same number means the same thing everywhere in the report.
>
> One definitional consequence, recorded: the union row is now resampled
> **stratified by centre** rather than as one undifferentiated pool, because all
> rows share a single draw. This preserves each centre's proportion in every
> replicate.
>
> ## Reported alongside, never merged into the CI
>
> - **per-arm seed std** of the metric;
> - **per-seed deltas**, listed individually;
> - **sign consistency** across seeds — a robustness descriptor, not a second
>   test, and only meaningful once both arms have ≥ 2 seeds;
> - any finding involving a **single-seed arm is labelled provisional**
>   regardless of how narrow its CI is. At D1 the fed arm will have one seed, so
>   every fed finding is provisional by this rule.
>
> ## What it changes in the current two-seed table
>
> | row | pooled − local (seed-mean) | 95% CI | per-seed | sign-consistent |
> |---|---|---|---|---|
> | pooled union | **−0.0004** | [−0.0193, +0.0175] | s0 +0.0065, s1 −0.0073 | **no** |
> | mean over centres | +0.0543 | [−0.0208, +0.0829] | s0 +0.0464, s1 +0.0622 | yes |
>
> The union delta **changes sign between seeds** and the seed-mean is −0.0004 —
> pooled and local are not merely indistinguishable there, they swap order on a
> rerun. Still 0 of 8 comparisons significant. The mean-over-centres delta is
> sign-consistent and the larger effect, but its CI includes 0 too.
>
> This is precisely the robustness question the CI cannot answer, which is why
> sign consistency is reported next to it rather than folded into it.
>
> ## Tests pinning the amendment (23/23)
>
> - a single-seed seed-mean delta reproduces the previous single-seed paired
>   delta **exactly**, so the amendment does not move the existing answer;
> - unequal seed counts run, and dropping a seed changes the result — proof every
>   seed is used rather than one being silently preferred;
> - an arm compared against itself returns exactly `[0, 0]` on every row at any
>   seed count, which only holds because the draw is shared;
> - the headline seed-mean and the paired point estimate agree by construction.

### Amendment 2 — commit `3502782`

Evidence state when made: pooled and local arms at all three seeds; drift and
quantum magnitudes measured; **no federated finetune file present in
`results/` at that commit**.

> # Pre-registration amendment — drift rule resolution-scaling (2026-08-20)
>
> **Made on drift magnitudes alone, before any federated result was observed.**
> No fed run exists at the time of writing; `results/` contains no
> `fed_*finetune*` file. This amendment cannot have been shaped by a fed outcome.
>
> ## What was wrong with the >1pt drift rule
>
> Phase 3 policy 3 set a >1 point final-vs-peak drift as a protocol trigger. That
> threshold was calibrated on the **pooled union curve**, which is scored over
> 4,650 test images across 8 classes and moves in steps of ~0.5 points.
>
> It does not transfer to per-silo curves. Balanced accuracy averages recall over
> the classes *present* in a centre's test set, so its smallest possible change —
> its **quantum** — is `1 / (n_classes_present × min_class_count)`:
>
> | centre | test n | classes present | min class count | quantum (pts) |
> |---|---|---|---|---|
> | c0 | 2483 | 8 | 24 | 0.52 |
> | c1 | 791 | 5 | 4 | 5.00 |
> | c2 | 672 | 7 | 1 | 14.29 |
> | c3 | 452 | 7 | 10 | 1.43 |
> | c4 | 164 | 3 | 49 | 0.68 |
> | c5 | 88 | 4 | 1 | **25.00** |
>
> At c5 a single test image changing class flips the metric by 25 points. A
> 1-point trigger on such a curve fires on a quantity the metric cannot even
> represent.
>
> The quantum predicts drift far better than silo size does:
>
> | predictor | corr with mean drift |
> |---|---|
> | quantum `1/(n_present × min)` | **−0.812** |
> | quantum `1/(8 × min)` | −0.818 |
> | min class count | +0.558 |
> | log train size | +0.439 |
>
> The earlier reading that drift "scales inversely with train size" was wrong.
> It scales with **metric resolution**, which merely correlates with size.
>
> Direct evidence: `local c5` final is 0.4724 at **both** s0 and s2. The
> checkpoints differ (distinct weight hashes) and the predictions differ on
> **5 of 88 images**, yet every class's correct-count is unchanged
> (6/9, 72/74, 1/4, 0/1), so the metric returns the identical value. The curve
> is quantised so coarsely that five prediction changes are invisible to it.
>
> ## The change
>
> 1. **Final-vs-peak drift is reclassified as a curve-noise diagnostic, not a
>    protocol trigger, for any curve whose quantum exceeds 0.5 points.** That is
>    every centre except c0 (0.52) and c4 (0.68 — retained as diagnostic-only,
>    being above 0.5). It remains a trigger for the pooled union curve.
> 2. **The protocol trigger becomes: a significant negative OLS slope over the
>    final half of training** (epochs/rounds 21–40 of 40, or 26–50 of 50),
>    significance at the curve level, two-sided, α = 0.05.
>
> Measured under the new rule on the existing arms:
>
> - **2 of 21** curves show a significant negative slope: `local s1 c1`
>   (−0.424 pts/ep, t = −4.82) and `local s2 c4` (−0.178, t = −3.52).
> - **8 of 21** show a significant *positive* slope, including `pooled s2`.
> - **11 of 21** show no distinguishable slope.
>
> Under the old rule, 17 of 21 curves "drifted". Under the new one, 2 decline.
> The old rule was measuring quantisation noise, not decline.
>
> ## Secondary estimator: tail mean
>
> The **mean of the last 5 epochs/rounds** is pre-specified as a secondary
> estimator, reported alongside the final-epoch primary, never replacing it.
>
> **It applies to per-centre and mean-over-centres metrics only.** It is **not
> computable for the union or the strata** on the existing pooled/local
> checkpoints: those statistics need the concatenated predictions of all six
> per-silo models at each epoch, and only each silo's final checkpoint was
> retained — the rolling checkpoint is overwritten every epoch. Union balanced
> accuracy is not a weighted average of per-centre values, so it cannot be
> reconstructed from the CSVs.
>
> Consequently the tail mean **cannot be applied symmetrically to the strata
> endpoints**, which are the pre-registered primary endpoints. It is therefore a
> diagnostic for curve stability, and no primary claim may rest on it.

---

## 6. Standing caveats

**c5 has a resolution ceiling.** Its test set is 88 images across 4 present
classes, one of which has a **single** image, so balanced accuracy there moves
in steps of 25 points (RESULTS §3). Two different checkpoints whose predictions
differ on 5 of 88 images return the identical value. No per-centre claim about
c5 can be finer than its quantum — and c5 carries the largest point estimate in
the study.

**The small stratum is only half-resolved.** Pooling c4 with c5 was meant to
buy test-set power, and it does: 252 images instead of 88. But c5's
single-image class still sets the stratum's quantum at 25 points, so the
pooling fixes the sample size without fixing the resolution.

**Per-centre denominators differ.** Each centre's balanced accuracy averages
over a different number of classes — 8 at c0, 3 at c4, 4 at c5 (RESULTS §3).
"Balanced accuracy" is therefore not the same statistic across the columns of
any per-centre table, and mean-over-centres averages six differently-defined
quantities.

**Within-run oscillation is a third uncertainty channel.** The bootstrap CI
covers test-set sampling; the seed std covers run-to-run variance; neither
covers how far a curve moves between adjacent epochs within a single run
(RESULTS §8, mean absolute step). A difference smaller than that step is not
resolvable by this protocol even with unlimited seeds and test images.

**The federated arm's local-step budget is asymmetric across silos.** Rounds
are defined in local *steps*, not epochs, so every silo takes 100 updates per
round regardless of size. The smallest silo cycles its 351 images many times
per round while the largest sees a fraction of its 9,930 once. This is FLamby's
convention and is deliberate — it stops the biggest silo taking 28× more
gradient updates — but it means "one round" is a different amount of learning
per centre, and any per-centre federated result inherits that asymmetry.

**Hardware is confounded with arm.** Pooled and local ran on a local GTX 1650
in bfloat16; the federated arm will run on a Colab A100. Same dtype and same
code, but different hardware, different accumulation order, and a different
`num_workers`. Nothing suggests this matters at the precision being reported,
but arm and machine are not independent in this study.

**The union metric is dominated by c0.** The routed union pools all 4,650 test
images, of which 2,483 — 53% — come from c0. The union is therefore close to a
c0-weighted statistic, which is why it and mean-over-centres can disagree in
sign (RESULTS §7). Both are reported, always labelled, never merged.

---

## 7. Open questions and next steps

1. **Run D1** — the federated arm, per `D1_COLAB.md`, on a Colab A100 at tag
   `d1-frozen`. This is the study's actual question and the only thing blocking
   Phase 4 completion. Every fed finding will be **provisional**: one seed, so
   its run-to-run variance is unmeasured.
2. **Does federation help the small stratum?** H1 predicts `fed − local > 0`
   there. The pooled-vs-local contrast at `small` is the one row where all three
   seeds agree in sign — but its CI still includes 0 and its quantum is 25
   points, so even a real effect may not be resolvable.
3. **Can c5 be measured at all?** A per-centre claim needs a metric not
   quantised at 25 points. Options worth weighing: report c5 only within strata;
   report per-class recall instead of their average; or accept that c5 supports
   no per-centre claim and say so plainly.
4. **Fed seeds 1–2**, then Phase 5 (fedprox / fedadam / fedyogi / fedadagrad),
   then Phase 6 plots — the cross-silo generalisation heatmap (RESULTS §6) and
   the who-benefits scatter.
5. **Is the local arm's oscillation worth fixing?** The slope test says only 2
   of 21 curves decline, so probably not — but c2 and c5 swing by more than any
   effect being measured, and a shorter schedule would cost little to test.

---

*Numbers: `results/RESULTS.md`. Phase detail: `results/PHASE3_SUMMARY.md`,
`results/PHASE4_SUMMARY.md`. Runbook: `D1_COLAB.md`. Orientation:
`HANDOFF.md`.*
