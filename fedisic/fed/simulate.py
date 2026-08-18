"""Serial single-machine federated simulation.

The "federation" is a for-loop over clients on one GPU: each round, every
client starts from the current global weights, trains locally for a fixed
number of steps, and the server combines the results. Nothing but weights
crosses the client boundary.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import torch

from .averaging import normalized_client_weights, weighted_average
from .strategies import FEDOPT_STRATEGIES, ServerOptimizer, local_train


@dataclass
class Client:
    id: int
    name: str
    train_loader: object
    test_loader: Optional[object]
    n_train: int
    loss_fn: torch.nn.Module


@dataclass
class FedConfig:
    strategy: str = "fedavg"        # fedavg | fedprox | fedadam | fedyogi | fedadagrad
    rounds: int = 50
    local_steps: int = 100          # gradient updates per client per round
    lr: float = 5e-4                # client learning rate (FLamby baseline: 5e-4)
    optimizer: str = "adam"         # client optimizer, re-initialised each round
    prox_mu: float = 0.0            # FedProx proximal coefficient
    server_lr: float = 1e-2         # FedOpt server learning rate
    beta1: float = 0.9
    beta2: float = 0.99
    tau: float = 1e-3
    eval_every: int = 1
    device: str = "cpu"
    use_amp: bool = False


def run_federated(
    model: torch.nn.Module,
    clients: List[Client],
    cfg: FedConfig,
    eval_fn: Optional[Callable[[torch.nn.Module], Dict[str, float]]] = None,
    on_round_end: Optional[Callable[[int, dict, Optional[ServerOptimizer], dict], None]] = None,
    start_round: int = 0,
    server_opt: Optional[ServerOptimizer] = None,
) -> Dict[str, torch.Tensor]:
    """Run cfg.rounds of federated training; returns (and loads) final global weights.

    The model passed in provides the initial global weights — to resume, load a
    checkpoint into the model (and pass its ServerOptimizer + start_round) first.

    on_round_end(round_idx_1based, global_sd, server_opt, metrics) fires at every
    evaluation point and is the hook for CSV logging + checkpointing.
    """
    if cfg.strategy not in ("fedavg", "fedprox") + FEDOPT_STRATEGIES:
        raise ValueError(f"Unknown strategy '{cfg.strategy}'")

    model.to(cfg.device)
    global_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if server_opt is None and cfg.strategy in FEDOPT_STRATEGIES:
        trainable_keys = [n for n, p in model.named_parameters() if p.requires_grad]
        server_opt = ServerOptimizer(
            cfg.strategy, trainable_keys, lr=cfg.server_lr,
            beta1=cfg.beta1, beta2=cfg.beta2, tau=cfg.tau,
        )

    weights = normalized_client_weights([c.n_train for c in clients])

    for rnd in range(start_round, cfg.rounds):
        client_sds = []
        for c in clients:
            model.load_state_dict(global_sd)
            gparams = None
            use_prox = cfg.strategy == "fedprox" and cfg.prox_mu > 0.0
            if use_prox:
                gparams = [p.detach().clone() for p in model.parameters() if p.requires_grad]
            local_train(
                model,
                c.train_loader,
                c.loss_fn,
                steps=cfg.local_steps,
                lr=cfg.lr,
                device=cfg.device,
                optimizer=cfg.optimizer,
                prox_mu=cfg.prox_mu if use_prox else 0.0,
                global_params=gparams,
                use_amp=cfg.use_amp,
            )
            client_sds.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        avg_sd = weighted_average(client_sds, weights)
        global_sd = server_opt.step(global_sd, avg_sd) if server_opt is not None else avg_sd

        if (rnd + 1) % cfg.eval_every == 0 or (rnd + 1) == cfg.rounds:
            model.load_state_dict(global_sd)
            metrics = eval_fn(model) if eval_fn is not None else {}
            if on_round_end is not None:
                on_round_end(rnd + 1, global_sd, server_opt, metrics)

    model.load_state_dict(global_sd)
    return global_sd
