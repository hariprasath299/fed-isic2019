"""Universal runner for the three arms of the benchmark.

Arms:
  pooled  — upper bound: one model on all 18,597 training images.
  local   — lower bound: six models, each trained only on its own silo.
  fed     — the question: FedAvg / FedProx / FedAdam / FedYogi / FedAdagrad.

Modes:
  probe    — linear head on Phase-1 cached features (fast; Phases 2-3).
  finetune — full EfficientNet-B0 fine-tuning on images (Phases 4-5).

Every run writes {out}/{run_name}.csv (one row per eval), {out}/{run_name}.pt
(atomic resume checkpoint), {out}/{run_name}_final.pt, and a per-class recall
CSV at the end. --resume continues an interrupted run in any arm, restoring
optimizer state along with the weights.

Examples:
  python scripts/run.py --arm pooled --mode probe --epochs 20
  python scripts/run.py --arm local  --mode probe --epochs 20
  python scripts/run.py --arm fed    --mode probe --strategy fedavg --rounds 50
  python scripts/run.py --arm fed    --mode finetune --strategy fedprox \
      --prox-mu 0.1 --rounds 50 --local-steps 100 --amp --seed 1
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from fedisic.data import (  # noqa: E402
    NUM_CLASSES,
    build_feature_silos,
    build_image_silos,
    make_loader,
    pooled_label_counts,
    pooled_train_dataset,
)
from fedisic.evaluate import (  # noqa: E402
    collect_predictions,
    evaluate_clients,
    per_class_recall,
)
from fedisic.fed.simulate import Client, FedConfig, run_federated  # noqa: E402
from fedisic.fed.strategies import (  # noqa: E402
    ALL_STRATEGIES,
    FEDOPT_STRATEGIES,
    ServerOptimizer,
    local_train,
    make_optimizer,
)
from fedisic.losses import (  # noqa: E402
    FLAMBY_ALPHA,
    WeightedFocalLoss,
    inverse_frequency_alpha,
)
from fedisic.models import LinearProbe, build_finetune_model  # noqa: E402
from fedisic.utils import (  # noqa: E402
    CsvLogger,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    set_seed,
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=["pooled", "local", "fed"])
    ap.add_argument("--mode", default="probe", choices=["probe", "finetune"])
    ap.add_argument("--strategy", default="fedavg", choices=list(ALL_STRATEGIES))
    # schedule
    ap.add_argument("--rounds", type=int, default=50, help="federated rounds")
    ap.add_argument("--local-steps", type=int, default=100, help="client steps per round")
    ap.add_argument("--epochs", type=int, default=20, help="epochs for pooled/local arms")
    ap.add_argument("--eval-every", type=int, default=1)
    # optimisation
    ap.add_argument("--lr", type=float, default=5e-4, help="client/base LR (FLamby: 5e-4)")
    ap.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--gamma", type=float, default=2.0, help="focal loss gamma")
    ap.add_argument(
        "--alpha",
        default="pooled",
        choices=["pooled", "local", "flamby"],
        help="focal alphas: computed from pooled counts (default), per-client counts, "
        "or FLamby's published constants (only if phase 0 confirms label order)",
    )
    ap.add_argument("--prox-mu", type=float, default=0.1, help="FedProx mu")
    ap.add_argument("--server-lr", type=float, default=1e-2, help="FedOpt server LR")
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=1e-3)
    # data / io
    ap.add_argument("--features-dir", default="data/features")
    ap.add_argument("--cache-dir", default=None, help="HF datasets cache dir")
    ap.add_argument("--out", default="results")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--amp",
        action="store_true",
        help="mixed precision (finetune on GPU); uses bfloat16 where supported, "
        "because float16 overflows EfficientNet-B0 and NaNs the run",
    )
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-augment", action="store_true", help="disable train-time augmentation")
    return ap.parse_args()


# ------------------------------ builders ------------------------------------ #

def build_silos(args):
    if args.mode == "probe":
        return build_feature_silos(args.features_dir)
    return build_image_silos(cache_dir=args.cache_dir, augment_train=not args.no_augment)


def make_model(args) -> torch.nn.Module:
    return LinearProbe() if args.mode == "probe" else build_finetune_model()


def make_alpha(args, silo_counts, pooled_counts) -> torch.Tensor:
    if args.alpha == "flamby":
        return FLAMBY_ALPHA
    counts = silo_counts if args.alpha == "local" else pooled_counts
    return inverse_frequency_alpha(counts, NUM_CLASSES)


def clients_from_silos(silos, args, device) -> list:
    pooled_counts = pooled_label_counts(silos)
    clients = []
    for s in silos:
        loss = WeightedFocalLoss(
            alpha=make_alpha(args, s.train_label_counts, pooled_counts), gamma=args.gamma
        )
        clients.append(
            Client(
                id=s.id,
                name=s.name,
                train_loader=make_loader(
                    s.train_ds, args.batch_size, shuffle=True,
                    num_workers=args.num_workers, device=device, seed=args.seed + s.id,
                ),
                test_loader=make_loader(
                    s.test_ds, args.batch_size, shuffle=False,
                    num_workers=args.num_workers, device=device,
                ),
                n_train=s.n_train,
                loss_fn=loss,
            )
        )
    return clients


def write_per_class_recall(model, clients, device, path):
    ys, ps = [], []
    for c in clients:
        y, p = collect_predictions(model, c.test_loader, device)
        ys.append(y)
        ps.append(p)
    rec = per_class_recall(np.concatenate(ys), np.concatenate(ps), NUM_CLASSES)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "recall"])
        for k, r in enumerate(rec):
            w.writerow([k, f"{r:.6f}"])
    print(f"per-class recall (pooled test): {[f'{r:.3f}' for r in rec]} -> {path}")


def restore_optimizer(optimizer_name, lr, model, ck):
    """Rebuild the client optimizer and reload its state from a checkpoint.

    Adam carries first and second moment estimates. Dropping them on resume
    makes the first steps after the resume behave like the start of training,
    a visible transient, so a resumed run is not the same trajectory as a
    continuous one. Restoring them removes that discontinuity.

    Returns None when the checkpoint predates optimizer-state saving, in which
    case the caller starts fresh moments exactly as before.
    """
    state = ck.get("opt")
    if state is None:
        return None
    params = [p for p in model.parameters() if p.requires_grad]
    opt = make_optimizer(optimizer_name, params, lr)
    opt.load_state_dict(state)
    return opt


def _resume_note(opt):
    return "" if opt is not None else \
        " (checkpoint carries no optimizer state; Adam moments restart)"


# -------------------------------- arms -------------------------------------- #

def run_pooled(args, silos, device, paths):
    clients = clients_from_silos(silos, args, device)  # for eval loaders
    counts = pooled_label_counts(silos)
    loss = WeightedFocalLoss(alpha=make_alpha(args, counts, counts), gamma=args.gamma)
    loader = make_loader(
        pooled_train_dataset(silos), args.batch_size, shuffle=True,
        num_workers=args.num_workers, device=device, seed=args.seed,
    )
    model = make_model(args).to(device)
    logger = CsvLogger(paths["csv"])

    opt = None
    start_epoch = 0
    if args.resume and os.path.exists(paths["ckpt"]):
        ck = load_checkpoint(paths["ckpt"])
        model.load_state_dict(ck["model"])
        start_epoch = ck["epoch"]
        opt = restore_optimizer(args.optimizer, args.lr, model, ck)
        print(f"Resumed pooled run at epoch {start_epoch}" + _resume_note(opt))

    use_amp = args.amp and device.startswith("cuda")
    for ep in range(start_epoch + 1, args.epochs + 1):
        opt = local_train(
            model, loader, loss, steps=len(loader), lr=args.lr,
            device=device, optimizer=args.optimizer, opt=opt, use_amp=use_amp,
        )
        if ep % args.eval_every == 0 or ep == args.epochs:
            metrics = evaluate_clients(model, clients, device)
            logger.log({"arm": "pooled", "mode": args.mode, "strategy": "-",
                        "seed": args.seed, "round": ep, **metrics})
            save_checkpoint(
                {"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict()},
                paths["ckpt"],
            )
            print(f"[pooled ep {ep:03d}] " + _fmt(metrics))

    save_checkpoint({"model": model.state_dict()}, paths["final"])
    write_per_class_recall(model, clients, device, paths["per_class"])


def run_local(args, silos, device, paths):
    clients = clients_from_silos(silos, args, device)
    logger = CsvLogger(paths["csv"])
    xeval_logger = CsvLogger(paths["csv"].replace(".csv", "_xeval.csv"))
    use_amp = args.amp and device.startswith("cuda")

    for c in clients:
        final_path = paths["final"].replace("_final.pt", f"_c{c.id}_final.pt")
        ckpt_path = paths["ckpt"].replace(".pt", f"_c{c.id}.pt")
        model = make_model(args).to(device)
        opt = None
        start_epoch = 0
        if args.resume and os.path.exists(ckpt_path):
            ck = load_checkpoint(ckpt_path)
            model.load_state_dict(ck["model"])
            start_epoch = ck["epoch"]
            opt = restore_optimizer(args.optimizer, args.lr, model, ck)
            print(f"[local c{c.id}] resumed at epoch {start_epoch}" + _resume_note(opt))
        elif args.resume and os.path.exists(final_path):
            # A finished run from before per-epoch checkpointing existed: its
            # epoch count is not recoverable from the file, so never silently
            # retrain over it. scripts/migrate_local_ckpt.py promotes such a
            # run to a resumable checkpoint using its config's epoch count.
            print(f"[local c{c.id}] final checkpoint exists with no epoch record, skipping")
            continue
        if start_epoch >= args.epochs:
            print(f"[local c{c.id}] already trained to epoch {start_epoch}, skipping")
            continue
        for ep in range(start_epoch + 1, args.epochs + 1):
            opt = local_train(
                model, c.train_loader, c.loss_fn, steps=len(c.train_loader), lr=args.lr,
                device=device, optimizer=args.optimizer, opt=opt, use_amp=use_amp,
            )
            if ep % args.eval_every == 0 or ep == args.epochs:
                y, p = collect_predictions(model, c.test_loader, device)
                from sklearn.metrics import balanced_accuracy_score

                own = float(balanced_accuracy_score(y, p))
                logger.log({"arm": "local", "mode": args.mode, "strategy": "-",
                            "seed": args.seed, "client": c.id, "round": ep,
                            "bal_acc_own": own})
                print(f"[local c{c.id} ep {ep:03d}] own bal_acc {own:.4f}")
                save_checkpoint(
                    {"epoch": ep, "model": model.state_dict(), "opt": opt.state_dict()},
                    ckpt_path,
                )
        # cross-silo row: how this silo's model does everywhere (generalisation gap)
        metrics = evaluate_clients(model, clients, device)
        xeval_logger.log({"arm": "local_xeval", "mode": args.mode, "strategy": "-",
                          "seed": args.seed, "client": c.id, "round": args.epochs, **metrics})
        save_checkpoint({"model": model.state_dict()}, final_path)


def run_fed(args, silos, device, paths):
    clients = clients_from_silos(silos, args, device)
    model = make_model(args).to(device)
    logger = CsvLogger(paths["csv"])

    cfg = FedConfig(
        strategy=args.strategy, rounds=args.rounds, local_steps=args.local_steps,
        lr=args.lr, optimizer=args.optimizer, prox_mu=args.prox_mu,
        server_lr=args.server_lr, beta1=args.beta1, beta2=args.beta2, tau=args.tau,
        eval_every=args.eval_every, device=device,
        use_amp=args.amp and device.startswith("cuda"),
    )

    start_round = 0
    server_opt = None
    if args.resume and os.path.exists(paths["ckpt"]):
        ck = load_checkpoint(paths["ckpt"])
        model.load_state_dict(ck["model"])
        start_round = ck["round"]
        if ck.get("server_opt") is not None:
            trainable = [n for n, p in model.named_parameters() if p.requires_grad]
            server_opt = ServerOptimizer(
                args.strategy, trainable, lr=args.server_lr,
                beta1=args.beta1, beta2=args.beta2, tau=args.tau,
            )
            server_opt.load_state_dict(ck["server_opt"])
        print(f"Resumed fed run at round {start_round}")

    def eval_fn(m):
        return evaluate_clients(m, clients, device)

    def on_round_end(rnd, global_sd, sopt, metrics):
        logger.log({"arm": "fed", "mode": args.mode, "strategy": args.strategy,
                    "seed": args.seed, "round": rnd, **metrics})
        save_checkpoint(
            {"round": rnd, "model": global_sd,
             "server_opt": sopt.state_dict() if sopt is not None else None},
            paths["ckpt"],
        )
        print(f"[{args.strategy} rd {rnd:03d}] " + _fmt(metrics))

    run_federated(model, clients, cfg, eval_fn=eval_fn, on_round_end=on_round_end,
                  start_round=start_round, server_opt=server_opt)
    save_checkpoint({"model": model.state_dict()}, paths["final"])
    write_per_class_recall(model, clients, device, paths["per_class"])


def _fmt(metrics: dict) -> str:
    parts = [f"pooled {metrics.get('bal_acc_pooled', float('nan')):.4f}"]
    parts += [f"c{i} {metrics[f'bal_acc_c{i}']:.4f}" for i in range(6)
              if f"bal_acc_c{i}" in metrics]
    return " | ".join(parts)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    name = args.run_name or (
        f"{args.arm}_{args.mode}"
        + (f"_{args.strategy}" if args.arm == "fed" else "")
        + f"_s{args.seed}"
    )
    os.makedirs(args.out, exist_ok=True)
    paths = {
        "csv": os.path.join(args.out, f"{name}.csv"),
        "ckpt": os.path.join(args.out, f"{name}.pt"),
        "final": os.path.join(args.out, f"{name}_final.pt"),
        "per_class": os.path.join(args.out, f"{name}_per_class.csv"),
        "config": os.path.join(args.out, f"{name}_config.json"),
    }
    with open(paths["config"], "w") as f:
        json.dump({**vars(args), "device": device}, f, indent=2)

    print(f"run: {name} | device: {device}")
    silos = build_silos(args)
    print("silos: " + ", ".join(f"c{s.id}={s.n_train}" for s in silos))

    if args.arm == "pooled":
        run_pooled(args, silos, device, paths)
    elif args.arm == "local":
        run_local(args, silos, device, paths)
    else:
        run_fed(args, silos, device, paths)


if __name__ == "__main__":
    main()
