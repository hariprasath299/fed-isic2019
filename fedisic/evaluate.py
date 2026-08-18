"""Evaluation.

The benchmark metric is balanced accuracy (mean per-class recall) — with a
~49%-prevalence majority class, plain accuracy would reward a clinically
useless model. Every evaluation reports one number per client plus the pooled
test metric, because the mean of the per-client numbers averages two opposite
effects (big silos gain little, small silos gain a lot) into a figure that
describes neither.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, recall_score


@torch.no_grad()
def collect_predictions(model, loader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.to(device)
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        logits = model(xb.to(device, non_blocking=True))
        ps.append(logits.argmax(dim=1).cpu().numpy())
        ys.append(yb.numpy())
    return np.concatenate(ys), np.concatenate(ps)


def evaluate_clients(model, clients, device: str) -> Dict[str, float]:
    """Balanced accuracy on each client's own test set + on the pooled test set
    (the union of all client test sets)."""
    out: Dict[str, float] = {}
    all_y: List[np.ndarray] = []
    all_p: List[np.ndarray] = []
    for c in clients:
        if c.test_loader is None:
            continue
        y, p = collect_predictions(model, c.test_loader, device)
        out[f"bal_acc_c{c.id}"] = float(balanced_accuracy_score(y, p))
        all_y.append(y)
        all_p.append(p)
    if all_y:
        y = np.concatenate(all_y)
        p = np.concatenate(all_p)
        out["bal_acc_pooled"] = float(balanced_accuracy_score(y, p))
    return out


def per_class_recall(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    return recall_score(
        y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0
    )


def bootstrap_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Percentile-bootstrap CI for balanced accuracy.

    Essential on the small silos: centre 5's test set is 88 images across 8
    classes — some classes have 2-3 examples, and a point estimate there is
    not a measurement. Resamples that drop a class entirely are scored over
    the classes present (sklearn's behaviour), which is the standard choice.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = balanced_accuracy_score(y_true[idx], y_pred[idx])
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(balanced_accuracy_score(y_true, y_pred)), float(lo), float(hi)
