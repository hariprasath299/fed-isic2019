# D1 — federated finetune arm on a Colab A100

Runbook for the one session that produces `fed_finetune_fedavg_s0`. Every
parameter here is already fixed by the sweep and the pre-registration; nothing
in this file is a fresh choice.

**Analysis frozen at commit `d7235ea`** (2026-08-20 06:21:35 +1000), which
predates any federated finetune run. Do not amend the analysis after this
session starts — that is the point of the freeze.

Locked settings, for reference while reading the cells below:

| setting | value | fixed by |
|---|---|---|
| lr | `1e-4` | 4-point sweep; interior maximum |
| strategy | `fedavg` | SPEC §5 phase 4 |
| rounds × local steps | 50 × 100 | SPEC §6 |
| batch / optimizer | 64 / Adam | SPEC §6 |
| loss | focal γ=2.0, α from pooled counts | SPEC §3 |
| eval | every round | SPEC §6 |
| seed | 0 only this session | reviewer, "stop before fed seeds" |
| AMP | on → bf16 on A100 | `autocast_dtype`; fp16 NaNs B0 |

---

## Cell 1 — GPU check

```python
!nvidia-smi
```

Confirm **A100**. On a T4 or V100 the projection below will not hold, and bf16
is emulated rather than native on pre-Ampere cards — stop and restart the
runtime rather than proceeding on the wrong accelerator.

## Cell 2 — clone at the frozen commit

```bash
%cd /content
!git clone <REMOTE_URL> skin-care
%cd /content/skin-care
!git checkout d7235ea
!git log --oneline -1
```

Everything lives on `/content` (local SSD), never a mounted Drive — SPEC §8.
Drive-backed I/O will dominate the run time on an image workload.

## Cell 3 — dependencies

```bash
!pip install -q -r requirements.txt
!python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"
```

`is_bf16_supported()` must print `True`.

## Cell 4 — phase 0 verify, on /content

```bash
!python scripts/phase0_verify_data.py --out inspection
```

Must exit 0. This re-checks the HF mirror's schema and the 6×(train/test)
counts — `9930/3163/2691/1807/655/351` train and `2483/791/672/452/164/88`
test — on *this* machine. A silent schema drift on the mirror would otherwise
surface as an inexplicable result three hours later.

## Cell 5 — pre-flight, one round with units

```bash
!python scripts/d1_preflight.py --lr 1e-4 --rounds-planned 50 --budget-hours 8
```

Reports ms/step per client, eval seconds, images per round, that round
expressed in pooled epochs, peak VRAM, and the 50-round projection against the
budget. **Read the GO/NO-GO line before starting cell 6.**

If it says NO-GO / TIGHT, do not start the full run and reduce nothing
unilaterally — the schedule is pre-registered. Report the projection and get a
decision.

## Cell 6 — the run

```bash
!python scripts/run.py \
  --arm fed --mode finetune --strategy fedavg \
  --rounds 50 --local-steps 100 --eval-every 1 \
  --lr 1e-4 --optimizer adam --batch-size 64 --gamma 2.0 --alpha pooled \
  --amp --num-workers 2 --seed 0 --device cuda --resume \
  --run-name fed_finetune_fedavg_s0 2>&1 | tee /content/fed_d1.log
```

`--resume` is deliberate: a Colab disconnect then costs one round, not the
session. Re-running this identical cell after a drop continues from the last
checkpoint.

## Cell 7 — sync results out (run this before the runtime dies)

```bash
!zip -r /content/fed_d1_results.zip \
    results/fed_finetune_fedavg_s0*.csv \
    results/fed_finetune_fedavg_s0*_config.json \
    results/fed_finetune_fedavg_s0*_per_class.csv \
    results/fed_finetune_fedavg_s0_final.pt \
    /content/fed_d1.log
from google.colab import files; files.download('/content/fed_d1_results.zip')
```

Sync small artefacts out, never in (SPEC §8). The `_final.pt` is included
because Phase 6 recomputes every number from weights rather than trusting the
CSV — without it the aggregate cannot run.

---

## Extend / drift rules

Both are decided from the curve in `results/fed_finetune_fedavg_s0.csv`, not
by eye on the log.

**Extend past 50 rounds** only if still climbing, and "still climbing" is
measured against the curve's own noise — the ep-20 lesson, where a 5-epoch
slope inside sd 0.0505 read as a trend that 20 further epochs showed was worth
+0.0021. Concretely: compare the mean of the last 10 rounds against the mean of
the preceding 10. Extend only if that difference exceeds the sd of the last 20
rounds. Extension is `--resume` with a larger `--rounds`, never a fresh run.

**Drift**: if the final round sits more than 1 point below the best round, that
is the Phase 3 drift trigger. Do **not** switch to best-round reporting — that
would be test-set model selection, which reporting policy 1 forbids. Record it
and propose cosine decay or a shorter schedule for the seeded runs.

## Aggregate, with the pre-registered endpoints

```bash
!python scripts/aggregate_results.py --mode finetune
```

Primary endpoints, per the 2026-08-20 amendment: paired `fed − local` and
`fed − pooled` seed-mean deltas at mean-over-centres and per size stratum.

- **H1**: `fed − local > 0` in the small stratum {c4, c5}, test n = 252.
- **H2**: `fed − local ≤ 0` in the large stratum {c0, c1}, test n = 3274.

Every fed finding this session is **provisional** — one fed seed, so its
run-to-run variance is unmeasured. The report labels this automatically.

`gap-closed` is expected to print `n.s.` at the union: the pooled−local
headroom there is −0.0004 with a CI spanning 0, and at two seeds it changes
sign between them. That is not a bug in the report.

## Stop

Stop after the aggregate. **Fed seeds 1 and 2 are out of scope for this
session** and are a separate decision.

---

## Not yet implemented

The size-stratum rows (small/mid/large) are pre-registered but
`aggregate_results.py` currently emits per-centre, union and mean-over-centres
only. The stratum estimator is a direct reuse of `score_rows_on_draw` — the
draw is already stratified within centre — but it is not written yet. It must
land **before** the D1 aggregate, and being analysis code frozen by the
amendment, it should be reviewed as an implementation of the existing
pre-registration rather than a change to it.
