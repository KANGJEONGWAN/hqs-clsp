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


def _loader(cfg, subset_ratio: float):
    ds = StateEstimationDataset(cfg["paths"]["train_csv"])
    n = max(2, int(len(ds) * subset_ratio))
    sub = Subset(ds, list(range(n)))
    return DataLoader(sub, batch_size=min(int(cfg["train"]["batch_size"]), n), shuffle=True, num_workers=0, collate_fn=collate_fn)


def _minimal_text(t: str, n_words: int = 4) -> str:
    w = t.strip().split()
    return t if len(w) <= n_words else " ".join(w[:n_words])


def train_variant(cfg, device, subset_ratio: float, epochs: int, mode: str) -> float:
    dl = _loader(cfg, subset_ratio)
    model = TextOnlyStateEstimator(
        text_model_name=cfg["model"]["text_model_name"],
        projection_dim=int(cfg["model"]["projection_dim"]),
        dropout=float(cfg["model"]["projection_dropout"]),
        reg_hidden_dim=int(cfg["model"]["reg_hidden_dim"]),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["learning_rate"]))

    last = 0.0
    for _ in range(epochs):
        model.train()
        total, count = 0.0, 0
        for b in dl:
            y = b["targets"].to(device)
            if mode == "structured":
                texts = b["text"]
            elif mode == "minimal":
                texts = [_minimal_text(t) for t in b["text"]]
            elif mode == "blank":
                texts = ["No additional context provided." for _ in b["text"]]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            out = model(texts=texts, device=device)
            loss = F.mse_loss(out["pred"], y)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * y.size(0)
            count += y.size(0)
        last = total / max(count, 1)
    return float(last)


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

    loss_structured = train_variant(copy.deepcopy(cfg), device, args.subset_ratio, args.epochs, mode="structured")
    loss_minimal = train_variant(copy.deepcopy(cfg), device, args.subset_ratio, args.epochs, mode="minimal")
    loss_blank = train_variant(copy.deepcopy(cfg), device, args.subset_ratio, args.epochs, mode="blank")

    print("Three-way ablation (lower is better)")
    print(f"  Structured text: {loss_structured:.6f}")
    print(f"  Minimal text:    {loss_minimal:.6f}")
    print(f"  Blank text:      {loss_blank:.6f}")


if __name__ == "__main__":
    main()
