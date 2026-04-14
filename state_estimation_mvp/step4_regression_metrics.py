from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import StateEstimationDataset
from model import TextOnlyStateEstimator
from train import collate_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="state_estimation_mvp/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/state_estimation_mvp/best.pt")
    parser.add_argument("--out-csv", type=str, default="outputs/state_estimation_mvp/regression_metrics.csv")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if (cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available()) else "cpu")

    ds = StateEstimationDataset(cfg["paths"]["val_csv"])
    dl = DataLoader(ds, batch_size=min(64, len(ds)), shuffle=False, num_workers=0, collate_fn=collate_fn)

    model = TextOnlyStateEstimator(
        text_model_name=cfg["model"]["text_model_name"],
        projection_dim=int(cfg["model"]["projection_dim"]),
        dropout=float(cfg["model"]["projection_dropout"]),
        reg_hidden_dim=int(cfg["model"]["reg_hidden_dim"]),
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    preds, trues = [], []
    with torch.inference_mode():
        for b in dl:
            out = model(texts=b["text"], device=device)
            preds.append(out["pred"].cpu().numpy())
            trues.append(b["targets"].numpy())

    pred = np.vstack(preds)
    true = np.vstack(trues)

    arousal_mse = float(np.mean((pred[:, 0] - true[:, 0]) ** 2))
    valence_mse = float(np.mean((pred[:, 1] - true[:, 1]) ** 2))
    cognitive_load_proxy_mse = float(np.mean((pred[:, 2] - true[:, 2]) ** 2))

    out_df = pd.DataFrame(
        {
            "metric": ["arousal_mse", "valence_mse", "cognitive_load_proxy_mse", "avg_mse"],
            "value": [
                arousal_mse,
                valence_mse,
                cognitive_load_proxy_mse,
                (arousal_mse + valence_mse + cognitive_load_proxy_mse) / 3.0,
            ],
        }
    )
    out_df.to_csv(args.out_csv, index=False)
    print(out_df)
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
