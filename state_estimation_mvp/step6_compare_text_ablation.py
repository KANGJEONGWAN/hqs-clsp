from __future__ import annotations

import argparse
import copy

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from dataset import StateEstimationDataset
from model import TextOnlyStateEstimator
from train import collate_fn, seed_everything


def run_epoch(model, loader, optimizer, device, use_text: bool):
    model.train(optimizer is not None)
    total = 0.0
    count = 0

    for batch in loader:
        y = batch["targets"].to(device)
        texts = batch["text"] if use_text else ["No additional context provided." for _ in batch["text"]]

        out = model(texts=texts, device=device)
        loss = F.mse_loss(out["pred"], y)

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total += loss.item() * y.size(0)
        count += y.size(0)

    return total / max(count, 1)


def train_short(cfg, device, use_text: bool, subset_ratio: float, epochs: int) -> float:
    ds = StateEstimationDataset(cfg["paths"]["train_csv"])
    n = max(2, int(len(ds) * subset_ratio))
    ds = Subset(ds, list(range(n)))

    dl = DataLoader(
        ds,
        batch_size=min(int(cfg["train"]["batch_size"]), n),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = TextOnlyStateEstimator(
        text_model_name=cfg["model"]["text_model_name"],
        projection_dim=int(cfg["model"]["projection_dim"]),
        dropout=float(cfg["model"]["projection_dropout"]),
        reg_hidden_dim=int(cfg["model"]["reg_hidden_dim"]),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["learning_rate"]))
    final_loss = None
    for _ in range(epochs):
        final_loss = run_epoch(model, dl, opt, device, use_text=use_text)

    return float(final_loss)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="state_estimation_mvp/config.yaml")
    parser.add_argument("--subset-ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed_everything(int(cfg["seed"]))
    device = torch.device("cuda" if (cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available()) else "cpu")

    loss_no_text = train_short(copy.deepcopy(cfg), device, use_text=False, subset_ratio=args.subset_ratio, epochs=args.epochs)
    loss_with_text = train_short(copy.deepcopy(cfg), device, use_text=True, subset_ratio=args.subset_ratio, epochs=args.epochs)

    print("Ablation Result (lower is better)")
    print(f"  A) Blank text baseline: {loss_no_text:.6f}")
    print(f"  B) Structured text:     {loss_with_text:.6f}")
    print(f"  Delta (A-B):            {loss_no_text - loss_with_text:.6f}")


if __name__ == "__main__":
    main()
