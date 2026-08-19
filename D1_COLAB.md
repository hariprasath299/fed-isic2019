# D1 — federated finetune arm on a Colab A100

Runbook for the one session that produces `fed_finetune_fedavg_s0`. Every
parameter here is already fixed by the sweep and the pre-registration; nothing
in this file is a fresh choice.

**Analysis frozen at tag `d1-frozen`.** The tag is the pin: it is stable
across later doc-only commits, and it predates any federated finetune run. Do
not amend the analysis after this session starts — that is the point of the
freeze. The seed-aggregation amendment it carries is commit `d7235ea`
(2026-08-20 06:21:35 +1000).

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

The repo is private, so the clone needs a token. Use `getpass` — never paste a
PAT into a notebook cell, because Colab saves cell text and a committed or
shared notebook then leaks it.

```python
from getpass import getpass

GH_USER = "hariprasath299"
GH_REPO = "fed-isic2019"
GH_TOKEN = getpass("GitHub PAT (repo scope, not echoed): ")

import subprocess
url = f"https://{GH_USER}:{GH_TOKEN}@github.com/{GH_USER}/{GH_REPO}.git"
subprocess.run(["git", "clone", url, "/content/skin-care"], check=True)
del GH_TOKEN, url  # keep the token out of later cell output and history
```

```bash
%cd /content/skin-care
!git remote set-url origin https://github.com/hariprasath299/fed-isic2019.git
!git checkout d1-frozen
!git log --oneline -1
```

The `set-url` strips the token back out of `.git/config`, so a later `!git
remote -v` or an exported notebook cannot expose it. Expect the checkout to
report a detached HEAD at `d1-frozen` — that is the frozen analysis point and
is deliberate.

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

## Implemented and verified before the freeze

The size-stratum rows (small/mid/large) are implemented and render. They come
from the same centre-subset statistic as the per-centre and union rows, so a
singleton stratum equals its centre's row and the all-centre stratum equals
the union — asserted by exact equality in the tests, 26/26 green.

Pooled-vs-local strata already show why pooling mattered: `small` (n=252) is
+0.2528 with both seeds agreeing in sign, against c5 alone (n=88) whose
interval spans 0.42. H1 and H2 concern `fed − local` and remain untested,
which is what this session is for.
