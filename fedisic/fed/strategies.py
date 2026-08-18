"""Local training and server-side optimizers.

- FedAvg and FedProx share one local loop; FedProx adds the proximal term
  (mu/2) * ||w - w_global||^2 to the local loss (Li et al., 2020).
- FedAdam / FedYogi / FedAdagrad ("FedOpt", Reddi et al., 2021,
  arXiv:2003.00295) keep the same local loop and change only the server
  update: the weighted average of client deltas is treated as a
  pseudo-gradient fed to an adaptive optimizer.

Rounds are defined in local *steps*, not epochs. With Fed-ISIC2019's 28x
client-size skew, "one epoch each" would let the biggest silo take 28x more
gradient updates per round than the smallest — on top of its 53% share of the
averaging weight. FLamby standardizes on update counts for the same reason.
"""

from typing import Dict, List, Optional, Sequence

import torch

from fedisic.utils import autocast_dtype

FEDOPT_STRATEGIES = ("fedadam", "fedyogi", "fedadagrad")
ALL_STRATEGIES = ("fedavg", "fedprox") + FEDOPT_STRATEGIES


def make_optimizer(name: str, params, lr: float) -> torch.optim.Optimizer:
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    raise ValueError(f"Unknown optimizer '{name}' (use 'adam' or 'sgd').")


def local_train(
    model: torch.nn.Module,
    loader,
    loss_fn,
    steps: int,
    lr: float,
    device: str,
    optimizer: str = "adam",
    opt: Optional[torch.optim.Optimizer] = None,
    prox_mu: float = 0.0,
    global_params: Optional[List[torch.Tensor]] = None,
    use_amp: bool = False,
) -> torch.optim.Optimizer:
    """Run `steps` gradient updates, cycling the loader as needed.

    Pass a persistent `opt` to keep optimizer state across calls (pooled/local
    epoch training). Leave it None for federated rounds: re-initialising the
    client optimizer each round is the standard FedAvg setup.

    Returns the optimizer so callers can reuse it.
    """
    model.to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if opt is None:
        opt = make_optimizer(optimizer, params, lr)
    amp_dtype = autocast_dtype() if use_amp else None
    # A GradScaler is only needed for float16; bfloat16 has float32's range.
    scaler = (
        torch.amp.GradScaler("cuda") if use_amp and amp_dtype is torch.float16 else None
    )

    data_iter = iter(loader)
    for step in range(int(steps)):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, yb = next(data_iter)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = loss_fn(model(xb), yb)
        else:
            loss = loss_fn(model(xb), yb)

        # A non-finite loss poisons every weight within a few steps and the run
        # keeps going: argmax over NaN logits always returns class 0, which scores
        # exactly chance and reads like a plausible bad-hyperparameter result.
        # Stop instead of writing hours of that to disk.
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss ({loss.item()}) at local step {step + 1}. "
                f"amp={use_amp} dtype={amp_dtype} lr={lr}. Training aborted."
            )

        if prox_mu > 0.0 and global_params is not None:
            prox = sum(((p - g) ** 2).sum() for p, g in zip(params, global_params))
            loss = loss + 0.5 * prox_mu * prox

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
    return opt


class ServerOptimizer:
    """FedOpt server update (Reddi et al., 2021).

    delta_t = avg_client_weights - global_weights   (per trainable key)
    m_t = beta1 * m + (1 - beta1) * delta
    v_t = { v + delta^2                                (FedAdagrad)
          { v - (1-beta2) * delta^2 * sign(v - delta^2) (FedYogi)
          { beta2 * v + (1-beta2) * delta^2             (FedAdam)
    global <- global + server_lr * m / (sqrt(v) + tau)

    Only trainable parameter keys go through the adaptive update; BatchNorm
    buffers (running stats, counters) take the plain FedAvg average.
    v is initialised to tau^2 as in the paper.
    """

    def __init__(
        self,
        variant: str,
        trainable_keys: Sequence[str],
        lr: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3,
    ):
        variant = variant.lower()
        if variant not in FEDOPT_STRATEGIES:
            raise ValueError(f"variant must be one of {FEDOPT_STRATEGIES}, got '{variant}'")
        self.variant = variant
        self.keys = set(trainable_keys)
        self.lr, self.b1, self.b2, self.tau = float(lr), float(beta1), float(beta2), float(tau)
        self.m: Dict[str, torch.Tensor] = {}
        self.v: Dict[str, torch.Tensor] = {}

    def step(self, global_sd: Dict[str, torch.Tensor], avg_sd: Dict[str, torch.Tensor]):
        new_sd: Dict[str, torch.Tensor] = {}
        for k, gv in global_sd.items():
            if k not in self.keys:
                # Non-trainable buffers: plain weighted average.
                new_sd[k] = avg_sd[k]
                continue
            g = gv.detach().to("cpu", torch.float64)
            a = avg_sd[k].detach().to("cpu", torch.float64)
            delta = a - g
            m = self.m.get(k)
            v = self.v.get(k)
            if m is None:
                m = torch.zeros_like(delta)
            if v is None:
                v = torch.full_like(delta, self.tau**2)
            m = self.b1 * m + (1.0 - self.b1) * delta
            d2 = delta * delta
            if self.variant == "fedadagrad":
                v = v + d2
            elif self.variant == "fedyogi":
                v = v - (1.0 - self.b2) * d2 * torch.sign(v - d2)
            else:  # fedadam
                v = self.b2 * v + (1.0 - self.b2) * d2
            self.m[k], self.v[k] = m, v
            new_sd[k] = (g + self.lr * m / (v.sqrt() + self.tau)).to(gv.dtype)
        return new_sd

    def state_dict(self) -> dict:
        return {
            "variant": self.variant,
            "keys": sorted(self.keys),
            "lr": self.lr,
            "beta1": self.b1,
            "beta2": self.b2,
            "tau": self.tau,
            "m": self.m,
            "v": self.v,
        }

    def load_state_dict(self, state: dict) -> None:
        self.variant = state["variant"]
        self.keys = set(state["keys"])
        self.lr, self.b1 = state["lr"], state["beta1"]
        self.b2, self.tau = state["beta2"], state["tau"]
        self.m, self.v = state["m"], state["v"]
