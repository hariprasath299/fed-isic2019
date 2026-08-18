"""Generate synthetic Phase-1 feature caches for offline smoke tests.

Creates data/features_fake/{train,test}.npz with the real 6-silo structure
(shrunk counts, real class imbalance, weak linear signal) so every arm and
strategy in scripts/run.py can be exercised end-to-end in seconds on CPU:

    python scripts/make_fake_features.py
    python scripts/run.py --arm fed --mode probe --strategy fedavg \
        --rounds 3 --local-steps 5 --features-dir data/features_fake \
        --out /tmp/smoke --num-workers 0 --device cpu
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

OUT = "data/features_fake"
TRAIN_COUNTS = [200, 90, 70, 50, 30, 20]
TEST_COUNTS = [60, 30, 25, 20, 12, 8]
CLASS_P = [0.49, 0.20, 0.12, 0.07, 0.05, 0.04, 0.02, 0.01]


def main() -> None:
    rng = np.random.default_rng(0)
    os.makedirs(OUT, exist_ok=True)
    for split, counts in (("train", TRAIN_COUNTS), ("test", TEST_COUNTS)):
        feats, labels, centers = [], [], []
        for cid, n in enumerate(counts):
            f = rng.normal(size=(n, 1280)).astype("float32")
            y = rng.choice(8, size=n, p=CLASS_P)
            f += y[:, None] * 0.15  # weak signal so probes can learn something
            feats.append(f)
            labels.append(y)
            centers.append(np.full(n, cid))
        path = os.path.join(OUT, f"{split}.npz")
        np.savez(
            path,
            features=np.concatenate(feats),
            labels=np.concatenate(labels).astype("int64"),
            centers=np.concatenate(centers).astype("int64"),
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
