from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from dataset import StateEstimationDataset
from model import TextOnlyStateEstimator


def seed_everything(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    texts = [x["text"] for x in batch]
    y = torch.stack([x["targets"] for x in batch], dim=0)
    ids = [x["sample_id"] for x in batch]
    return {"sample_id": ids, "text": texts, "targets": y}


def run_epoch(model, loader, optimizer, device, train: bool = True):
    model.train(train)
    total = {"loss": 0.0, "reg": 0.0}

    for batch in loader:
        y = batch["targets"].to(device)
        texts = batch["text"]

        out = model(texts=texts, device=device)
        loss = F.mse_loss(out["pred"], y)

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = y.size(0)
        total["loss"] += loss.item() * bs
        total["reg"] += loss.item() * bs

    n = len(loader.dataset)
    return {k: v / max(n, 1) for k, v in total.items()}


def _maybe_subset(ds, subset_ratio: float):
    if subset_ratio >= 1.0:
        return ds
    if not 0.0 < subset_ratio <= 1.0:
        raise ValueError("subset_ratio must be in (0, 1]")
    n = max(2, int(len(ds) * subset_ratio))
    return Subset(ds, list(range(n)))


def main(config_path: str, subset_ratio: float = 1.0, epochs_override: int | None = None):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(int(cfg["seed"]))

    device_str = cfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    train_ds = StateEstimationDataset(
        csv_path=cfg["paths"]["train_csv"],
    )
    val_ds = StateEstimationDataset(
        csv_path=cfg["paths"]["val_csv"],
    )

    train_ds = _maybe_subset(train_ds, subset_ratio=subset_ratio)
    val_ds = _maybe_subset(val_ds, subset_ratio=min(1.0, max(subset_ratio, 0.2)))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_fn,
    )

    model = TextOnlyStateEstimator(
        text_model_name=cfg["model"]["text_model_name"],
        projection_dim=int(cfg["model"]["projection_dim"]),
        dropout=float(cfg["model"]["projection_dropout"]),
        reg_hidden_dim=int(cfg["model"]["reg_hidden_dim"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    out_dir = Path("outputs") / "state_estimation_mvp"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    best_val = float("inf")

    num_epochs = int(epochs_override) if epochs_override is not None else int(cfg["train"]["epochs"])

    for epoch in range(1, num_epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, train=True)
        va = run_epoch(model, val_loader, optimizer, device, train=False)

        print(
            f"[Epoch {epoch}] "
            f"train loss={tr['loss']:.4f} (reg={tr['reg']:.4f}) | "
            f"val loss={va['loss']:.4f} (reg={va['reg']:.4f})"
        )

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="state_estimation_mvp/config.yaml")
    parser.add_argument("--subset-ratio", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    main(args.config, subset_ratio=args.subset_ratio, epochs_override=args.epochs)
