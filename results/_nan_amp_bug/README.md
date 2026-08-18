# Discarded: NaN sweep (float16 AMP overflow)

The three `pooled_lrsweep_*` finetune runs of 2026-08-19 (05:42-07:02) are void.

Every weight in every `_final.pt` here is NaN. Cause: `--amp` used float16
autocast, and EfficientNet-B0's `features[6]` overflows float16 on a real
Fed-ISIC2019 batch -> inf -> NaN logits on the *first* forward pass, before any
weight update. The runs did not diverge from a bad LR; they never trained.

They completed all 5 epochs and wrote plausible-looking CSVs because
`argmax` over an all-NaN logit vector returns class 0. Predicting one class
scores exactly 1/(classes present) per centre - which is why all three LRs,
spanning 10x, produced byte-identical chance-level numbers.

Fixed in `fedisic/utils.py::autocast_dtype` (bfloat16 where supported) plus a
non-finite-loss guard in `local_train`. Kept only as the evidence trail.
