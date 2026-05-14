from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from dataset_multimodal import MultimodalStateEstimationDataset
from model_multimodal import MultimodalStateEstimator
from train_multimodal import clsp_loss, collate_multimodal
from torch.utils.data import DataLoader

CONFIG = "state_estimation_mvp/config_multimodal.yaml"


def main() -> None:
    with open(CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 데이터 로드 ──────────────────────────────────────────────────
    ds = MultimodalStateEstimationDataset(
        text_csv=cfg["paths"]["text_train_csv"],
        ppg_csv=cfg["paths"]["ppg_feature_csv"],
    )
    loader = DataLoader(ds, batch_size=8, shuffle=False,
                        num_workers=0, collate_fn=collate_multimodal)
    batch = next(iter(loader))

    ppg_dim = ds.ppg_dim
    print(f"\n[데이터] 샘플 수: {len(ds)} | PPG 차원: {ppg_dim}")
    print(f"  text  : {batch['text'][0][:60]}...")
    print(f"  ppg   : shape={batch['ppg_features'].shape}")
    print(f"  target: shape={batch['targets'].shape} | "
          f"[arousal, valence, cog_load] 범위: "
          f"{batch['targets'].min():.2f} ~ {batch['targets'].max():.2f}")

    # ── 모델 초기화 ──────────────────────────────────────────────────
    model = MultimodalStateEstimator(
        text_model_name=cfg["model"]["text_model_name"],
        ppg_input_dim=ppg_dim,
        projection_dim=int(cfg["model"]["projection_dim"]),
        projection_dropout=float(cfg["model"]["projection_dropout"]),
        ppg_hidden_dim=int(cfg["model"]["ppg_hidden_dim"]),
        fusion_hidden_dim=int(cfg["model"]["fusion_hidden_dim"]),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[모델] 파라미터 수: {total_params:,}")

    # ── Forward pass ─────────────────────────────────────────────────
    model.eval()
    with torch.inference_mode():
        ppg = batch["ppg_features"].to(device)
        y   = batch["targets"].to(device)
        out = model(texts=batch["text"], ppg_features=ppg, device=device)

    text_z = out["text_z"]
    ppg_z  = out["ppg_z"]
    pred   = out["pred"]

    print(f"\n[Forward Pass]")
    print(f"  text_z : shape={tuple(text_z.shape)} | "
          f"NaN={torch.isnan(text_z).any().item()} | "
          f"mean={text_z.mean():.4f}")
    print(f"  ppg_z  : shape={tuple(ppg_z.shape)} | "
          f"NaN={torch.isnan(ppg_z).any().item()} | "
          f"mean={ppg_z.mean():.4f}")
    print(f"  pred   : shape={tuple(pred.shape)} | "
          f"NaN={torch.isnan(pred).any().item()}")
    print(f"  pred   : {pred.cpu().numpy().round(3)}")

    # ── Loss ─────────────────────────────────────────────────────────
    lambda_align = float(cfg["loss"].get("lambda_align", 0.1))
    l_reg   = F.mse_loss(pred, y)
    l_align = clsp_loss(text_z, ppg_z)
    l_total = l_reg + lambda_align * l_align

    print(f"\n[Loss]")
    print(f"  MSE (regression) : {l_reg.item():.4f}")
    print(f"  CLSP (alignment) : {l_align.item():.4f}")
    print(f"  Total            : {l_total.item():.4f}  (λ={lambda_align})")

    # ── Alignment 품질 ───────────────────────────────────────────────
    t = F.normalize(text_z, dim=-1)
    p = F.normalize(ppg_z, dim=-1)
    cosine_sim = (t * p).sum(dim=-1)
    print(f"\n[CLSP Alignment]")
    print(f"  text_z ↔ ppg_z cosine similarity: "
          f"mean={cosine_sim.mean():.4f} | std={cosine_sim.std():.4f}")

    print("\n✅ 멀티모달 파이프라인 연결 정상")


if __name__ == "__main__":
    main()
