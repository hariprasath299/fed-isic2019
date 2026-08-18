"""Phase 0 — verify the data before believing anything else.

Downloads the HF mirror, prints its schema, checks per-center train/test
counts against FLamby's published numbers, dumps the per-center class
distribution (both a table and a CSV for the writeup), and saves one sample
image per center so you can eyeball whether the mirror ships FLamby's
preprocessed images (shorter edge 224) or raw ISIC images.

Exit code 0 iff all counts match. Do not start training until they do.

Usage:
    python scripts/phase0_verify_data.py --out inspection
"""

import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from fedisic.data import (  # noqa: E402
    EXPECTED_TEST_COUNTS,
    EXPECTED_TRAIN_COUNTS,
    HF_DATASET_ID,
    NUM_CLASSES,
    NUM_CLIENTS,
    _to_int_array,
    detect_schema,
    load_hf_dataset,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=None, help="HF datasets cache dir")
    ap.add_argument("--out", default="inspection", help="where to write CSV + sample images")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"Loading {HF_DATASET_ID} ...")
    ds = load_hf_dataset(args.cache_dir)

    print("\n== Splits ==")
    for split in ds:
        print(f"  {split}: {len(ds[split])} rows")

    print("\n== Columns (train) ==")
    print(f"  {ds['train'].column_names}")
    print(f"  features: {ds['train'].features}")

    schema = detect_schema(ds)
    print(f"\n== Detected schema ==\n  {schema}")

    ok = True
    per_center_class = {}
    for split, expected in (("train", EXPECTED_TRAIN_COUNTS), ("test", EXPECTED_TEST_COUNTS)):
        raw_centers = ds[split][schema["center"]]
        raw_labels = ds[split][schema["label"]]
        centers = _to_int_array(raw_centers)
        labels = _to_int_array(raw_labels)

        if not isinstance(raw_centers[0], (int, np.integer)):
            print(f"\n  NOTE: '{schema['center']}' has non-integer values; "
                  f"mapping (sorted) -> ids: {sorted(set(raw_centers))}")
        if not isinstance(raw_labels[0], (int, np.integer)):
            print(f"  NOTE: '{schema['label']}' has non-integer values; "
                  f"mapping (sorted) -> ids: {sorted(set(raw_labels))}")

        print(f"\n== {split} counts per center (expected from FLamby) ==")
        for cid in range(NUM_CLIENTS):
            n = int((centers == cid).sum())
            exp = expected[cid]
            status = "OK" if n == exp else "MISMATCH"
            if n != exp:
                ok = False
            print(f"  center {cid}: {n:6d}  (expected {exp:6d})  {status}")
        extra = int((centers >= NUM_CLIENTS).sum())
        if extra:
            ok = False
            print(f"  WARNING: {extra} rows with center id >= {NUM_CLIENTS}")

        if split == "train":
            for cid in range(NUM_CLIENTS):
                mask = centers == cid
                per_center_class[cid] = np.bincount(labels[mask], minlength=NUM_CLASSES)[:NUM_CLASSES]

    print("\n== Train class distribution per center ==")
    header = "center " + " ".join(f"c{k:>5d}" for k in range(NUM_CLASSES)) + "  total"
    print("  " + header)
    for cid in range(NUM_CLIENTS):
        row = per_center_class[cid]
        print(f"  {cid:>6d} " + " ".join(f"{int(v):6d}" for v in row) + f" {int(row.sum()):6d}")
    pooled = np.sum(list(per_center_class.values()), axis=0)
    print(f"  pooled " + " ".join(f"{int(v):6d}" for v in pooled) + f" {int(pooled.sum()):6d}")

    csv_path = os.path.join(args.out, "class_distribution.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["center"] + [f"class_{k}" for k in range(NUM_CLASSES)] + ["total"])
        for cid in range(NUM_CLIENTS):
            row = per_center_class[cid]
            w.writerow([cid] + [int(v) for v in row] + [int(row.sum())])
        w.writerow(["pooled"] + [int(v) for v in pooled] + [int(pooled.sum())])
    print(f"\nSaved class distribution to {csv_path}")

    # One sample image per center: size tells you whether images are
    # FLamby-preprocessed (shorter edge 224, variable other edge) or raw ISIC.
    samples_dir = os.path.join(args.out, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    centers_train = _to_int_array(ds["train"][schema["center"]])
    for cid in range(NUM_CLIENTS):
        idx = int(np.where(centers_train == cid)[0][0])
        img = ds["train"][idx][schema["image"]]
        if getattr(img, "mode", "RGB") != "RGB":
            img = img.convert("RGB")
        path = os.path.join(samples_dir, f"center{cid}.png")
        img.save(path)
        print(f"  center {cid}: sample image size {img.size} -> {path}")

    print("\n" + ("ALL COUNTS MATCH — safe to proceed." if ok else
                  "COUNT MISMATCH — fix fedisic/data.py before training."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
