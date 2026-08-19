# Handoff — Fed-ISIC2019 benchmark

State as of **2026-08-20 07:09**. Repo `hariprasath299/fed-isic2019`, branch
`main` at `7191936`, tag `d1-frozen`. Tests **26/26**.

Read `SPEC.md` for the project's design. This file is only "where things stand
and what would bite you".

---

## One-paragraph status

Phases 0–3 (data, feature cache, harness, probe arms) are complete and
verified. Phase 4 (full fine-tuning) has its two endpoint arms done at seeds 0
and 1, with seed 2 running now. The federated arm — the actual research
question — **has not been run at all** in finetune mode and is the next
substantive step; it is packaged and waiting in `D1_COLAB.md`. The analysis is
frozen at tag `d1-frozen` so the federated result cannot influence how it is
measured.

## What is running right now

| job | state | ETA |
|---|---|---|
| `pooled_finetune_s2` | epoch 11/40 | ~11:00 |
| `local_finetune_s2` | queued behind it | ~15:00 |

Driver `./run_seeds12.sh 1e-4 2`, log `logs/seed2_driver.log`. Both legs use
`--resume`, so a crash costs one epoch. Running ~7.5 min/epoch against s0/s1's
5.7 — probably contention from analysis jobs earlier; flag only if it worsens
or the curve leaves family (it has not: ep 11 = 0.7314).

**Nothing waits on s2 except the final aggregate.** D1 can start immediately.

## Locked protocol — do not re-derive

| parameter | value | fixed by |
|---|---|---|
| learning rate | **1e-4** | 4-point sweep, interior maximum |
| AMP dtype | **bfloat16** | fp16 NaNs EfficientNet-B0 (below) |
| batch / optimizer | 64 / Adam | SPEC §6 |
| loss | focal γ=2.0, α from pooled counts | SPEC §3 |
| pooled/local schedule | 40 epochs, eval every epoch | convergence check |
| fed schedule | 50 rounds × 100 local steps | SPEC §6 |
| seeds | 0, 1, 2 | SPEC §6 |

LR sweep, 5 epochs each: 3e-5 **0.6165**, 1e-4 **0.6910**, 5e-4 **0.6702**,
1e-3 **0.6469**. The winner is bracketed on both sides, so it is an interior
maximum rather than a grid edge.

## The fp16 incident — read before touching AMP

The first three finetune runs were **void**: every weight NaN. `--amp` used
float16, and EfficientNet-B0's `features[6]` overflows float16 on a real batch
(activations reach absmax ~300 by `features[3]`), so `inf − inf` made the
logits NaN on the *first* forward pass, before any weight update.

It hid for three hours because `argmax` over an all-NaN vector returns class 0,
which scores exactly 1/(classes present) — a plausible bad-hyperparameter
result, not a crash.

- Fixed in tag `amp-fp16-nan-fix` (`9068bb1`): `autocast_dtype()` picks bf16
  where supported; `local_train` raises on a non-finite loss.
- Void runs archived at `results/_nan_amp_bug/` with a README. Moved, never
  deleted — `CsvLogger` appends, so leaving them would have mixed fresh rows
  into NaN ones.
- **All s0/s1/s2 runs postdate the fix and are clean** (every checkpoint
  audited at 0 NaN, all 8 classes predicted).
- Detection rule that came out of it: **identical metrics across differing
  hyperparameters is presumed corruption**, not a finding.

## Results so far

Finetune, seeds 0–1, 40 epochs (`results/AGGREGATE_finetune.md`):

| row | test n | local (routed) | pooled |
|---|---|---|---|
| stratum small (c4,c5) | 252 | 0.5284 ± 0.0230 | 0.7812 ± 0.0109 |
| stratum mid (c2,c3) | 1124 | 0.7094 ± 0.0264 | 0.6914 ± 0.0144 |
| stratum large (c0,c1) | 3274 | 0.7928 ± 0.0098 | 0.7977 ± 0.0023 |
| pooled union | 4650 | 0.7662 ± 0.0024 | 0.7658 ± 0.0074 |
| mean over centres | 4650 | 0.6577 ± 0.0085 | 0.7120 ± 0.0026 |

**0 of 8 paired comparisons are significant.** Pooled fine-tuning is not
measurably better than per-silo training on this dataset, at any centre or
aggregation. The union delta even changes sign between seeds (s0 +0.0065,
s1 −0.0073). The `small` stratum is the one place both seeds agree in sign and
by a wide margin (+0.2528, s0 +0.2443 / s1 +0.2614), and pooling c4 with c5
tightened its interval enough to almost exclude 0.

Pooled fine-tuning beats the frozen-feature probe by **+0.205** (0.5638 →
0.7711), so the accuracy lives in the backbone, not the head.

## Analysis rules that are easy to get wrong

1. **Never compare arms by whether their marginal CIs overlap.** Both arms are
   scored on the same test images, so their bootstrap noise is shared.
   Overlapping marginals can hide a real difference and disjoint ones can
   suggest a false one. Use the paired deltas — significance is the delta CI
   excluding 0. This invalidated two of my earlier conclusions.
2. **Seeds and bootstrap CIs measure different things.** The CI is test-set
   sampling; the std is run-to-run variance. Neither bounds the other, adding
   seeds does not narrow the CI, and both must be shown.
3. **Final epoch, never best epoch.** Selecting the best round on the test set
   is test-set model selection (Phase 3 policy 1).
4. **gap-closed divides by `pooled − local`.** That headroom is currently
   −0.0004 at the union with a CI spanning 0, so the report prints `n.s.`
   rather than a number. Expected, not a bug.
5. **"Still climbing" needs a window sized against the curve's own noise.**
   A 5-epoch slope inside sd 0.05 scatter read as a trend; 20 more epochs
   bought +0.0021.

Primary endpoints and hypotheses are pre-registered in `PHASE4_SUMMARY.md`
(2026-08-19) and amended 2026-08-20. **H1**: `fed − local > 0` in the small
stratum. **H2**: `fed − local ≤ 0` in the large stratum.

## Next actions, in order

1. **Run D1** — follow `D1_COLAB.md` end to end on a Colab A100. Needs a
   GitHub PAT for the private clone. Produces `fed_finetune_fedavg_s0`. Stops
   before fed seeds. *Ready now.*
2. **When s2 finishes** (~15:00): rerun
   `python scripts/aggregate_results.py --mode finetune` and check whether s0
   is an outlier across the three seeds. Contingency on record: if it is, s0
   gets a clean continuous rerun (it reached 40 epochs by resuming at 20 and
   carries one Adam-moment reset there; s1/s2 are continuous).
3. **After D1**: fed seeds 1–2, then Phase 5 (fedprox/fedadam/fedyogi/
   fedadagrad), then Phase 6 plots.

## Repo map

| path | what |
|---|---|
| `SPEC.md` | project design, phase plan, acceptance criteria |
| `D1_COLAB.md` | the federated-arm runbook — start here for D1 |
| `results/PHASE4_SUMMARY.md` | pre-registration, LR decision log, fp16 record |
| `results/PHASE3_SUMMARY.md` | probe-phase results + paired-test correction |
| `results/AGGREGATE_*.md` | generated reports, both modes |
| `scripts/run.py` | all three arms, both modes, resumable |
| `scripts/aggregate_results.py` | Phase 6 analysis; recomputes from weights |
| `scripts/d1_preflight.py` | one-round timing projection before committing a session |
| `results/_nan_amp_bug/` | the void fp16 runs, kept as evidence |

## Gotchas

- `results/*.pt` is gitignored. Checkpoints are local only, and
  `aggregate_results.py` **needs them** — it recomputes every number from
  weights. Sync `_final.pt` out of Colab or the aggregate cannot run.
- The prediction cache (`results/_cache/`) is keyed by checkpoint mtime, so
  reruns of the aggregate are CPU-only and safe to run beside training.
- `--resume` on the local arm skips silos whose `_final.pt` exists but which
  have no epoch record. `scripts/migrate_local_ckpt.py` promotes such runs.
- The GPU here is a 4 GB GTX 1650. bf16 fits at 2.5 GB; fp32 spills past VRAM
  and halves throughput.
- Windows + Git Bash. Use `.venv/Scripts/python.exe`.
