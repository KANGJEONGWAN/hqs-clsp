from __future__ import annotations

import argparse
import copy
import random

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

from dataset import StateEstimationDataset
from model import TextOnlyStateEstimator
from train import collate_fn, seed_everything


def _to_minimal(text: str, n_words: int = 4) -> str:
    words = text.strip().split()
    if len(words) <= n_words:
        return text
    return " ".join(words[:n_words])


def _batch_text(mode: str, texts: list[str], rng: random.Random) -> list[str]:
    if mode == "full":
        return texts
    if mode == "minimal":
        return [_to_minimal(t) for t in texts]
    if mode == "random":
        arr = texts[:]
        rng.shuffle(arr)
        return arr
    raise ValueError(f"Unknown mode: {mode}")


def run_mode(cfg, device, mode: str, subset_ratio: float, epochs: int) -> float:
    ds = StateEstimationDataset(cfg["paths"]["train_csv"])
    n = max(2, int(len(ds) * subset_ratio))
    dl = DataLoader(Subset(ds, list(range(n))), batch_size=min(int(cfg["train"]["batch_size"]), n), shuffle=True, num_workers=0, collate_fn=collate_fn)

    model = TextOnlyStateEstimator(
        text_model_name=cfg["model"]["text_model_name"],
        projection_dim=int(cfg["model"]["projection_dim"]),
        dropout=float(cfg["model"]["projection_dropout"]),
        reg_hidden_dim=int(cfg["model"]["reg_hidden_dim"]),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["learning_rate"]))
    rng = random.Random(int(cfg["seed"]))

    last = 0.0
    for _ in range(epochs):
        total, count = 0.0, 0
        for b in dl:
            y = b["targets"].to(device)
            texts = _batch_text(mode, b["text"], rng)
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

    full = run_mode(copy.deepcopy(cfg), device, "full", args.subset_ratio, args.epochs)
    minimal = run_mode(copy.deepcopy(cfg), device, "minimal", args.subset_ratio, args.epochs)
    random_mode = run_mode(copy.deepcopy(cfg), device, "random", args.subset_ratio, args.epochs)

    print("Text quality ablation (lower is better)")
    print(f"  full-context:    {full:.6f}")
    print(f"  minimal-context: {minimal:.6f}")
    print(f"  random-context:  {random_mode:.6f}")


if __name__ == "__main__":
    main()
