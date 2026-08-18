"""Seeding, CSV logging, and atomic checkpoints.

Design rule for the whole project: every run appends metrics to a CSV and
saves an atomic checkpoint at every evaluation, so a Colab disconnect costs
at most one round/epoch, never a run.
"""

import csv
import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class CsvLogger:
    """Append-only CSV logger. Header comes from the first row's keys.

    Rows are flushed + fsynced immediately so results survive a crash.
    If the file already exists (resume), the existing header is reused and
    extra keys in new rows are ignored.
    """

    def __init__(self, path: str):
        self.path = str(path)
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fieldnames = None
        if os.path.exists(self.path):
            with open(self.path, newline="") as f:
                header = next(csv.reader(f), None)
                if header:
                    self._fieldnames = header

    def log(self, row: dict) -> None:
        row = {
            k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()
        }
        new_file = self._fieldnames is None
        if new_file:
            self._fieldnames = list(row.keys())
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())


def save_checkpoint(obj, path: str) -> None:
    """Atomic save: write to a temp file, then rename.

    A disconnect mid-write must never leave a corrupt checkpoint behind.
    """
    path = str(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, map_location: str = "cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)
