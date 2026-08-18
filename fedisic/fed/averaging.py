"""Sample-weighted state-dict averaging (the heart of FedAvg).

Two deliberate details:
  - Accumulation happens in float64 and is cast back to each tensor's original
    dtype. EfficientNet's state dict contains integer buffers
    (BatchNorm's num_batches_tracked); naive float averaging would silently
    change their dtype and crash load_state_dict a round later.
  - Weights are validated (non-negative, sum to 1) on every call. This is one
    of the three harness invariants under test.
"""

from typing import Dict, List, Sequence

import torch

StateDict = Dict[str, torch.Tensor]


def normalized_client_weights(n_samples: Sequence[int]) -> List[float]:
    """FedAvg weights: each client's share of the total training samples."""
    total = float(sum(n_samples))
    if total <= 0:
        raise ValueError("Total sample count must be positive.")
    return [float(n) / total for n in n_samples]


def check_weights(weights: Sequence[float], tol: float = 1e-6) -> None:
    if any(w < 0 for w in weights):
        raise ValueError(f"Client weights must be non-negative, got {list(weights)}")
    s = float(sum(weights))
    if abs(s - 1.0) > tol:
        raise ValueError(f"Client weights must sum to 1 (got {s:.8f}).")


def weighted_average(state_dicts: List[StateDict], weights: Sequence[float]) -> StateDict:
    """Return sum_k w_k * sd_k, key by key, preserving each tensor's dtype."""
    if len(state_dicts) != len(weights):
        raise ValueError("Need exactly one weight per client state dict.")
    check_weights(weights)

    keys = list(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        if list(sd.keys()) != keys:
            raise ValueError("Client state dicts have mismatched keys.")

    out: StateDict = {}
    for k in keys:
        ref = state_dicts[0][k]
        acc = torch.zeros(ref.shape, dtype=torch.float64)
        for sd, w in zip(state_dicts, weights):
            acc += float(w) * sd[k].detach().to("cpu", torch.float64)
        out[k] = acc.to(ref.dtype)
    return out
