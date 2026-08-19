"""Promote a pre-checkpointing local run to a resumable one.

run_local gained per-epoch checkpoints after the seed-0 finetune arm had
already run, so those silos have {run}_c{i}_final.pt (weights only, no epoch)
but no {run}_c{i}.pt to resume from. This writes the resume checkpoints using
the epoch count recorded in the run's config, so the arm can be extended
instead of retrained from scratch.

    python scripts/migrate_local_ckpt.py --run-name local_finetune_s0
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from fedisic.utils import save_checkpoint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--clients", type=int, default=6)
    args = ap.parse_args()

    cfg_path = os.path.join(args.out, f"{args.run_name}_config.json")
    with open(cfg_path) as f:
        epochs = int(json.load(f)["epochs"])
    print(f"{cfg_path}: run reached epoch {epochs}")

    for cid in range(args.clients):
        final = os.path.join(args.out, f"{args.run_name}_c{cid}_final.pt")
        ckpt = os.path.join(args.out, f"{args.run_name}_c{cid}.pt")
        if not os.path.exists(final):
            print(f"  c{cid}: no final checkpoint, skipping")
            continue
        if os.path.exists(ckpt):
            print(f"  c{cid}: resume checkpoint already exists, leaving it alone")
            continue
        sd = torch.load(final, map_location="cpu", weights_only=False)["model"]
        save_checkpoint({"epoch": epochs, "model": sd}, ckpt)
        print(f"  c{cid}: wrote {ckpt} at epoch {epochs}")


if __name__ == "__main__":
    main()
