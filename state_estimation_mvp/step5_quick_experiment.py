from __future__ import annotations

import argparse

from train import main as train_main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="state_estimation_mvp/config.yaml")
    parser.add_argument("--subset-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    train_main(
        config_path=args.config,
        subset_ratio=args.subset_ratio,
        epochs_override=args.epochs,
    )
