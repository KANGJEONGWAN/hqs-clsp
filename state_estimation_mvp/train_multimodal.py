from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from dataset_multimodal import MultimodalStateEstimationDataset
from model_multimodal import MultimodalStateEstimator
from train import seed_everything


def collate_multimodal(batch):
    return {
        "sample_id": [x["sample_id"] for x in batch],
        "text": [x["text"] for x in batch],
        "ppg_features": torch.stack([x["ppg_features"] for x in batch], dim=0),
        "targets": torch.stack([x["targets"] for x in batch], dim=0),
    }


def clsp_loss(text_z: torch.Tensor, ppg_z: torch.Tensor) -> torch.Tensor:
    t = F.normalize(text_z, dim=-1)
    p = F.normalize(ppg_z, dim=-1)
    logits = t @ p.T
    targets = F.softmax((p @ p.T + t @ t.T) / 2, dim=-1)
    text_loss = (-targets * F.log_softmax(logits, dim=-1)).sum(dim=1)
    ppg_loss = (-targets.T * F.log_softmax(logits.T, dim=-1)).sum(dim=1)
    return ((text_loss + ppg_loss) / 2).mean()


def run_epoch(model, loader, optimizer, device, lambda_align: float, train: bool = True):
    model.train(train)
    total = {"loss": 0.0, "reg": 0.0, "align": 0.0}

    for b in loader:
        y = b["targets"].to(device)
        ppg = b["ppg_features"].to(device)
        texts = b["text"]

        out = model(texts=texts, ppg_features=ppg, device=device)
        l_reg = F.mse_loss(out["pred"], y)
        l_align = clsp_loss(out["text_z"], out["ppg_z"])
        loss = l_reg + float(lambda_align) * l_align

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        bs = y.size(0)
        total["loss"] += loss.item() * bs
        total["reg"] += l_reg.item() * bs
        total["align"] += l_align.item() * bs

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

    train_ds = MultimodalStateEstimationDataset(
        text_csv=cfg["paths"]["text_train_csv"],
        ppg_csv=cfg["paths"]["ppg_feature_csv"],
        allow_missing_ppg=bool(cfg["train"].get("allow_missing_ppg", False)),
    )
    val_ds = MultimodalStateEstimationDataset(
        text_csv=cfg["paths"]["text_val_csv"],
        ppg_csv=cfg["paths"]["ppg_feature_csv"],
        allow_missing_ppg=bool(cfg["train"].get("allow_missing_ppg", False)),
    )

    train_ds = _maybe_subset(train_ds, subset_ratio=subset_ratio)
    val_ds = _maybe_subset(val_ds, subset_ratio=min(1.0, max(subset_ratio, 0.2)))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_multimodal,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=collate_multimodal,
    )

    ppg_input_dim = train_ds.dataset.ppg_dim if isinstance(train_ds, Subset) else train_ds.ppg_dim
    model = MultimodalStateEstimator(
        text_model_name=cfg["model"]["text_model_name"],
        ppg_input_dim=int(ppg_input_dim),
        projection_dim=int(cfg["model"]["projection_dim"]),
        projection_dropout=float(cfg["model"]["projection_dropout"]),
        ppg_hidden_dim=int(cfg["model"]["ppg_hidden_dim"]),
        fusion_hidden_dim=int(cfg["model"]["fusion_hidden_dim"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    lambda_align = float(cfg["loss"].get("lambda_align", 0.0))

    out_dir = Path("outputs") / "state_estimation_mvp_multimodal"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.pt"
    best_val = float("inf")

    num_epochs = int(epochs_override) if epochs_override is not None else int(cfg["train"]["epochs"])
    for epoch in range(1, num_epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, lambda_align=lambda_align, train=True)
        va = run_epoch(model, val_loader, optimizer, device, lambda_align=lambda_align, train=False)

        print(
            f"[Epoch {epoch}] train loss={tr['loss']:.4f} (reg={tr['reg']:.4f}, align={tr['align']:.4f}) | "
            f"val loss={va['loss']:.4f} (reg={va['reg']:.4f}, align={va['align']:.4f})"
        )

        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(model.state_dict(), best_path)
            print(f"Saved best multimodal model: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="state_estimation_mvp/config_multimodal.yaml")
    parser.add_argument("--subset-ratio", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    main(args.config, subset_ratio=args.subset_ratio, epochs_override=args.epochs)
