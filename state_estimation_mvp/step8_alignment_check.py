from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from dataset import StateEstimationDataset
from model import TextOnlyStateEstimator
from train import collate_fn


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = torch.nn.functional.normalize(a, dim=-1)
    b = torch.nn.functional.normalize(b, dim=-1)
    return torch.sum(a * b, dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="state_estimation_mvp/config.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/state_estimation_mvp/best.pt")
    parser.add_argument("--out-csv", type=str, default="outputs/state_estimation_mvp/alignment_metrics.csv")
    parser.add_argument("--out-tsne", type=str, default="outputs/state_estimation_mvp/alignment_tsne.csv")
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

    text_all, ids_all = [], []
    with torch.inference_mode():
        for b in dl:
            out = model(texts=b["text"], device=device)
            text_all.append(out["text_z"].cpu())
            ids_all.extend(b["sample_id"])

    text_z = torch.cat(text_all, dim=0)

    val_df = pd.read_csv(cfg["paths"]["val_csv"])
    sig_map = {str(r["sample_id"]): str(r.get("context_signature", "")) for _, r in val_df.iterrows()}

    signature_to_indices: dict[str, list[int]] = {}
    for i, sid in enumerate(ids_all):
        sig = sig_map.get(sid, "")
        signature_to_indices.setdefault(sig, []).append(i)

    same_vals, diff_vals = [], []
    for i, sid in enumerate(ids_all):
        sig = sig_map.get(sid, "")
        same_candidates = [j for j in signature_to_indices.get(sig, []) if j != i]
        diff_candidates = [j for j in range(len(ids_all)) if sig_map.get(ids_all[j], "") != sig]
        if same_candidates:
            j = same_candidates[0]
            same_vals.append(float(cosine(text_z[i : i + 1], text_z[j : j + 1]).item()))
        if diff_candidates:
            j = diff_candidates[0]
            diff_vals.append(float(cosine(text_z[i : i + 1], text_z[j : j + 1]).item()))

    same_mean = float(np.mean(same_vals)) if same_vals else float("nan")
    diff_mean = float(np.mean(diff_vals)) if diff_vals else float("nan")

    metrics = pd.DataFrame(
        {
            "metric": ["same_signature_cosine_mean", "different_signature_cosine_mean", "margin"],
            "value": [same_mean, diff_mean, same_mean - diff_mean],
        }
    )
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out_csv, index=False)

    stacked = text_z.numpy()
    tags = ["text"] * len(ids_all)
    sample_ids = ids_all

    perplexity = min(30, max(5, len(stacked) // 4))
    tsne = TSNE(n_components=2, random_state=42, init="random", learning_rate="auto", perplexity=perplexity)
    xy = tsne.fit_transform(stacked)

    tsne_df = pd.DataFrame({"sample_id": sample_ids, "modality": tags, "x": xy[:, 0], "y": xy[:, 1]})
    tsne_df.to_csv(args.out_tsne, index=False)

    print(metrics)
    print(f"Saved: {args.out_csv}")
    print(f"Saved: {args.out_tsne}")


if __name__ == "__main__":
    main()
