from __future__ import annotations

import argparse
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import StateEstimationDataset
from model import TextOnlyStateEstimator
from train import collate_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="state_estimation_mvp/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if (cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available()) else "cpu")

    ds = StateEstimationDataset(
        csv_path=cfg["paths"]["train_csv"],
    )
    dl = DataLoader(ds, batch_size=min(8, len(ds)), shuffle=False, collate_fn=collate_fn)

    model = TextOnlyStateEstimator(
        text_model_name=cfg["model"]["text_model_name"],
        projection_dim=int(cfg["model"]["projection_dim"]),
        dropout=float(cfg["model"]["projection_dropout"]),
        reg_hidden_dim=int(cfg["model"]["reg_hidden_dim"]),
    ).to(device)

    batch = next(iter(dl))
    y = batch["targets"].to(device)
    text = batch["text"]

    out = model(texts=text, device=device)
    pred = out["pred"]
    loss = torch.nn.functional.mse_loss(pred, y)

    print("Shapes")
    print(f"  pred: {tuple(pred.shape)}")
    print(f"  text_z: {tuple(out['text_z'].shape)}")

    print("NaN checks")
    print(f"  pred has NaN: {bool(torch.isnan(pred).any().item())}")
    print(f"  total loss has NaN: {bool(torch.isnan(loss).any().item())}")

    print("Loss")
    print(f"  total_loss (mse): {loss.item():.6f}")


if __name__ == "__main__":
    main()
