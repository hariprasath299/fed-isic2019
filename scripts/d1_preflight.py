"""D1 pre-flight: time one federated round and project the full run.

Run this before committing an A100 session to 50 rounds. It executes exactly
one real round with the real config -- same model, batch size, local steps,
client count and AMP path -- then reports the units the projection rests on so
the estimate can be checked rather than trusted.

It writes nothing into results/: the run name is prefixed so a pre-flight can
never be mistaken for an arm, and the checkpoint it leaves behind is not a
resume point for the real run.

    python scripts/d1_preflight.py --lr 1e-4 --rounds-planned 50

Exit status is 0 whether or not the projection fits the budget; the GO/NO-GO
line is for a human to act on.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from fedisic.data import (  # noqa: E402
    NUM_CLASSES,
    build_image_silos,
    make_loader,
    pooled_label_counts,
)
from fedisic.evaluate import evaluate_clients  # noqa: E402
from fedisic.fed.simulate import Client  # noqa: E402
from fedisic.fed.strategies import local_train  # noqa: E402
from fedisic.losses import WeightedFocalLoss, inverse_frequency_alpha  # noqa: E402
from fedisic.models import build_finetune_model  # noqa: E402
from fedisic.utils import autocast_dtype, resolve_device, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--local-steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--rounds-planned", type=int, default=50)
    ap.add_argument("--budget-hours", type=float, default=None,
                    help="session budget; prints GO/NO-GO when given")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true", default=True)
    args = ap.parse_args()

    device = resolve_device(args.device)
    set_seed(0)
    use_amp = args.amp and device.startswith("cuda")

    print("=" * 68)
    print("D1 PRE-FLIGHT")
    print("=" * 68)
    print(f"device        : {device}")
    if device.startswith("cuda"):
        print(f"gpu           : {torch.cuda.get_device_name(0)}")
        print(f"vram total    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"amp dtype     : {autocast_dtype() if use_amp else 'off (fp32)'}")
    print(f"torch         : {torch.__version__}")

    t0 = time.time()
    silos = build_image_silos(cache_dir=args.cache_dir, augment_train=True)
    t_data = time.time() - t0
    counts = pooled_label_counts(silos)
    alpha = inverse_frequency_alpha(counts, NUM_CLASSES)

    clients = []
    for s in silos:
        clients.append(Client(
            id=s.id, name=s.name,
            train_loader=make_loader(s.train_ds, args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, device=device, seed=s.id),
            test_loader=make_loader(s.test_ds, args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, device=device),
            n_train=s.n_train,
            loss_fn=WeightedFocalLoss(alpha=alpha, gamma=2.0),
        ))
    print(f"data ready    : {t_data:.1f}s  |  silos " +
          ", ".join(f"c{c.id}={c.n_train}" for c in clients))

    model = build_finetune_model().to(device)

    # ---- one real round: every client trains local_steps ----
    print("\n-- one federated round --")
    per_client = []
    for c in clients:
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t = time.time()
        local_train(model, c.train_loader, c.loss_fn, steps=args.local_steps,
                    lr=args.lr, device=device, optimizer="adam", opt=None,
                    use_amp=use_amp)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = time.time() - t
        per_client.append(dt)
        print(f"  c{c.id}: {dt:7.1f}s  ({dt / args.local_steps * 1000:6.1f} ms/step)")
    t_train = sum(per_client)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t = time.time()
    evaluate_clients(model, clients, device)
    t_eval = time.time() - t
    print(f"  eval (4650 test images): {t_eval:.1f}s")

    # ---- units the projection rests on ----
    steps_per_round = len(clients) * args.local_steps
    imgs_per_round = steps_per_round * args.batch_size
    round_s = t_train + t_eval
    total_h = round_s * args.rounds_planned / 3600.0
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else 0.0

    print("\n" + "=" * 68)
    print("UNITS")
    print("=" * 68)
    print(f"  steps/round        : {len(clients)} clients x {args.local_steps} = {steps_per_round}")
    print(f"  images/round       : {steps_per_round} x {args.batch_size} = {imgs_per_round:,}")
    print(f"  = pooled epochs    : {imgs_per_round / sum(c.n_train for c in clients):.2f}"
          f"  (18,597 train images)")
    print(f"  train time/round   : {t_train:.1f}s")
    print(f"  eval time/round    : {t_eval:.1f}s   (--eval-every 1)")
    print(f"  ROUND TOTAL        : {round_s:.1f}s")
    print(f"  peak vram          : {peak:.2f} GB")

    print("\n" + "=" * 68)
    print("PROJECTION")
    print("=" * 68)
    print(f"  {args.rounds_planned} rounds        : {total_h:.2f} h "
          f"({round_s * args.rounds_planned / 60:.0f} min)")
    print(f"  + data load        : {t_data / 60:.1f} min one-off")
    for extra in (25, 50):
        print(f"  if extended +{extra:<3d}   : "
              f"{round_s * (args.rounds_planned + extra) / 3600.0:.2f} h total")
    if args.budget_hours:
        verdict = "GO" if total_h <= args.budget_hours * 0.8 else "NO-GO / TIGHT"
        print(f"\n  budget {args.budget_hours:.1f} h  ->  {verdict} "
              f"(projection is {100 * total_h / args.budget_hours:.0f}% of budget; "
              f"20% headroom reserved for eval drift and checkpoint I/O)")
    print("=" * 68)


if __name__ == "__main__":
    main()
