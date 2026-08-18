"""Class-imbalance-aware losses.

FLamby's Fed-ISIC2019 baseline loss is a weighted focal loss (gamma=2) with
fixed per-class alphas. With 8 classes whose prevalence spans ~49% down to
<1%, this is the single biggest lever for balanced accuracy: plain
cross-entropy strands the model well below target no matter what else is done.
"""

import torch
import torch.nn.functional as F
from torch import nn

# FLamby's published alphas for the 8 ISIC2019 classes, in FLamby's label order
# (MEL, NV, BCC, AK, BKL, DF, VASC, SCC). Only use these if Phase 0 confirms the
# HF mirror uses the same label-id ordering; otherwise prefer alpha computed
# from the actually-loaded label counts (see inverse_frequency_alpha).
FLAMBY_ALPHA = torch.tensor(
    [5.5813, 2.0472, 7.0204, 26.1194, 9.5369, 101.0707, 92.5224, 38.3443]
)


class WeightedFocalLoss(nn.Module):
    """FL(p_t) = -alpha_c * (1 - p_t)^gamma * log(p_t).

    With gamma=0 and alpha=1 this reduces exactly to cross-entropy
    (asserted in tests/test_harness.py).
    """

    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha.detach().clone().float()
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, targets.view(-1, 1)).view(-1)
        pt = logpt.exp()
        at = self.alpha.to(logits.device)[targets]
        return (-at * (1.0 - pt) ** self.gamma * logpt).mean()


def inverse_frequency_alpha(class_counts, num_classes: int) -> torch.Tensor:
    """alpha_c = N / (C_present * n_c) over classes present in the counts.

    Classes absent from a silo get alpha 0: they contribute no local samples,
    so their weight never enters that silo's loss. Normalisation is over
    present classes so the mean weight of seen samples stays ~1.
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float)
    if counts.numel() != num_classes:
        raise ValueError(f"expected {num_classes} counts, got {counts.numel()}")
    present = counts > 0
    alpha = torch.zeros(num_classes)
    alpha[present] = counts.sum() / (present.sum() * counts[present])
    return alpha
