"""Fed-ISIC2019 data loading from the Hugging Face mirror (flwrlabs/fed-isic2019).

Verified facts (from FLamby, the benchmark of record):
  - 6 clients, split by imaging site/device, 8 diagnosis classes.
  - Per-center (train/test) counts:
      (9930/2483) (3163/791) (2691/672) (1807/452) (655/164) (351/88)
    => 18,597 train / 4,650 test / 23,247 total.
  - FLamby's preprocessing: color constancy + resize so the shorter edge is
    224 px (aspect ratio kept). Its benchmark then crops 200x200.

The HF mirror's exact column names/values are auto-detected below and MUST be
confirmed with scripts/phase0_verify_data.py before any training run. If
detection fails, phase 0 prints everything needed to fix the constants here.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

HF_DATASET_ID = "flwrlabs/fed-isic2019"
NUM_CLIENTS = 6
NUM_CLASSES = 8
IMG_SIZE = 200  # FLamby's benchmark crop size
EXPECTED_TRAIN_COUNTS = [9930, 3163, 2691, 1807, 655, 351]
EXPECTED_TEST_COUNTS = [2483, 791, 672, 452, 164, 88]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_LABEL_CANDIDATES = ("label", "target", "labels", "diagnosis")
_CENTER_CANDIDATES = ("center", "centre", "client", "datacenter", "center_id")
_IMAGE_CANDIDATES = ("image", "img")


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #

def train_transforms(img_size: int = IMG_SIZE):
    """Torchvision analogue of FLamby's albumentations pipeline
    (RandomScale/Rotate(50)/BrightnessContrast/Flip/Shear/RandomCrop(200)/CoarseDropout)."""
    from torchvision import transforms as T

    return T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(50),
            T.ColorJitter(brightness=0.15, contrast=0.10),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.5, scale=(0.005, 0.02)),
        ]
    )


def eval_transforms(img_size: int = IMG_SIZE):
    """Deterministic test-time pipeline (also used for Phase 1 feature caching)."""
    from torchvision import transforms as T

    return T.Compose(
        [
            T.Resize(224),  # no-op if the mirror ships FLamby-preprocessed images
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# --------------------------------------------------------------------------- #
# HF loading + schema detection
# --------------------------------------------------------------------------- #

def _detect_column(colnames: Sequence[str], candidates: Sequence[str], kind: str) -> str:
    for c in candidates:
        if c in colnames:
            return c
    raise KeyError(
        f"Could not find a {kind} column among {list(colnames)}. "
        f"Run scripts/phase0_verify_data.py, inspect the schema, and update "
        f"_{kind.upper()}_CANDIDATES in fedisic/data.py."
    )


def load_hf_dataset(cache_dir: Optional[str] = None):
    """Load the mirror. Expects 'train' and 'test' splits (FLamby's fixed split)."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_ID, cache_dir=cache_dir)
    if "train" not in ds:
        raise RuntimeError(f"No 'train' split found; splits present: {list(ds.keys())}")
    if "test" not in ds:
        raise RuntimeError(
            f"Expected a 'test' split, found {list(ds.keys())}. The mirror may encode "
            "the split differently (e.g. a 'fold' column) — inspect via phase 0 and "
            "adapt load_hf_dataset()."
        )
    return ds


def detect_schema(ds) -> Dict[str, str]:
    cols = ds["train"].column_names
    return {
        "image": _detect_column(cols, _IMAGE_CANDIDATES, "image"),
        "label": _detect_column(cols, _LABEL_CANDIDATES, "label"),
        "center": _detect_column(cols, _CENTER_CANDIDATES, "center"),
    }


def _to_int_array(values) -> np.ndarray:
    """Map a column (ints or strings) to integer ids; strings map by sorted order."""
    arr = np.asarray(values)
    if arr.dtype.kind in ("i", "u"):
        return arr.astype(np.int64)
    uniques = sorted(set(arr.tolist()))
    lut = {u: i for i, u in enumerate(uniques)}
    return np.asarray([lut[v] for v in arr.tolist()], dtype=np.int64)


class HFImageDataset(Dataset):
    """One (center, split) slice of the HF dataset as (image_tensor, label)."""

    def __init__(self, hf_slice, image_col: str, label_col: str, transform):
        self.ds = hf_slice
        self.image_col = image_col
        self.label_col = label_col
        self.transform = transform
        self.labels = _to_int_array(hf_slice[label_col])

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        row = self.ds[int(idx)]
        img = row[self.image_col]
        if getattr(img, "mode", "RGB") != "RGB":
            img = img.convert("RGB")
        return self.transform(img), int(self.labels[idx])


@dataclass
class Silo:
    """Everything one client's data amounts to, before loaders are built."""

    id: int
    name: str
    train_ds: Dataset
    test_ds: Dataset
    n_train: int
    train_label_counts: np.ndarray  # shape [NUM_CLASSES]


def label_counts(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> np.ndarray:
    return np.bincount(labels, minlength=num_classes)[:num_classes]


def build_image_silos(
    cache_dir: Optional[str] = None,
    img_size: int = IMG_SIZE,
    augment_train: bool = True,
) -> List[Silo]:
    """Six per-center Silo objects with torch datasets attached."""
    ds = load_hf_dataset(cache_dir)
    schema = detect_schema(ds)
    t_train = train_transforms(img_size) if augment_train else eval_transforms(img_size)
    t_eval = eval_transforms(img_size)

    silos: List[Silo] = []
    centers = {split: _to_int_array(ds[split][schema["center"]]) for split in ("train", "test")}
    n_centers = int(max(centers["train"].max(), centers["test"].max())) + 1
    if n_centers != NUM_CLIENTS:
        raise RuntimeError(
            f"Detected {n_centers} centers, expected {NUM_CLIENTS}. Run phase 0."
        )
    for cid in range(NUM_CLIENTS):
        tr_idx = np.where(centers["train"] == cid)[0]
        te_idx = np.where(centers["test"] == cid)[0]
        tr = HFImageDataset(ds["train"].select(tr_idx), schema["image"], schema["label"], t_train)
        te = HFImageDataset(ds["test"].select(te_idx), schema["image"], schema["label"], t_eval)
        silos.append(
            Silo(
                id=cid,
                name=f"center{cid}",
                train_ds=tr,
                test_ds=te,
                n_train=len(tr),
                train_label_counts=label_counts(tr.labels),
            )
        )
    return silos


# --------------------------------------------------------------------------- #
# Cached features (Phases 2-3)
# --------------------------------------------------------------------------- #

class FeatureDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.x = torch.as_tensor(features, dtype=torch.float32)
        self.y = torch.as_tensor(labels, dtype=torch.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def build_feature_silos(features_dir: str) -> List[Silo]:
    """Silos over Phase-1 cached features: {features_dir}/train.npz and test.npz,
    each with arrays 'features' [N,1280], 'labels' [N], 'centers' [N]."""
    import os

    data = {}
    for split in ("train", "test"):
        path = os.path.join(features_dir, f"{split}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found — run scripts/phase1_cache_features.py first."
            )
        data[split] = np.load(path)

    silos: List[Silo] = []
    for cid in range(NUM_CLIENTS):
        tr = data["train"]
        te = data["test"]
        tr_idx = np.where(tr["centers"] == cid)[0]
        te_idx = np.where(te["centers"] == cid)[0]
        labels = tr["labels"][tr_idx]
        silos.append(
            Silo(
                id=cid,
                name=f"center{cid}",
                train_ds=FeatureDataset(tr["features"][tr_idx], labels),
                test_ds=FeatureDataset(te["features"][te_idx], te["labels"][te_idx]),
                n_train=len(tr_idx),
                train_label_counts=label_counts(labels),
            )
        )
    return silos


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def make_loader(
    ds: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    device: str = "cpu",
    seed: int = 0,
) -> DataLoader:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        generator=gen,
        drop_last=False,
    )


def pooled_train_dataset(silos: List[Silo]) -> Dataset:
    return ConcatDataset([s.train_ds for s in silos])


def pooled_label_counts(silos: List[Silo]) -> np.ndarray:
    return np.sum([s.train_label_counts for s in silos], axis=0)
