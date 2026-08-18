"""Phase 1 — cache frozen EfficientNet-B0 features (one GPU pass, ~minutes).

Runs every train/test image through the ImageNet-pretrained backbone with the
deterministic eval transform and saves 1280-d penultimate features. All of
Phases 2-3 (harness debugging + the first three-arm comparison) then run on a
~100 MB tensor problem where a full federated linear-probe run takes minutes,
so harness mistakes cost minutes instead of GPU-days.

Output: {out}/train.npz and {out}/test.npz with arrays
  features [N, 1280] float32, labels [N] int64, centers [N] int64.

Usage:
    python scripts/phase1_cache_features.py --out data/features --device cuda
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from fedisic.data import (  # noqa: E402
    HFImageDataset,
    _to_int_array,
    detect_schema,
    eval_transforms,
    load_hf_dataset,
)
from fedisic.models import Backbone  # noqa: E402
from fedisic.utils import resolve_device  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/features")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--img-size", type=int, default=200)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    os.makedirs(args.out, exist_ok=True)

    ds = load_hf_dataset(args.cache_dir)
    schema = detect_schema(ds)
    backbone = Backbone(pretrained=True).to(device).eval()

    for split in ("train", "test"):
        centers = _to_int_array(ds[split][schema["center"]])
        wrapped = HFImageDataset(
            ds[split], schema["image"], schema["label"], eval_transforms(args.img_size)
        )
        loader = DataLoader(
            wrapped,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.startswith("cuda"),
        )
        feats = []
        done = 0
        with torch.no_grad():
            for xb, _ in loader:
                feats.append(backbone(xb.to(device, non_blocking=True)).cpu().numpy())
                done += len(xb)
                if done % (args.batch_size * 20) < args.batch_size:
                    print(f"  {split}: {done}/{len(wrapped)}")
        features = np.concatenate(feats).astype(np.float32)
        out_path = os.path.join(args.out, f"{split}.npz")
        np.savez(out_path, features=features, labels=wrapped.labels, centers=centers)
        print(f"{split}: features {features.shape} -> {out_path}")


if __name__ == "__main__":
    main()
