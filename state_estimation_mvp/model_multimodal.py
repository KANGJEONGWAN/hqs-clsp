from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        z = self.fc2(self.act(h))
        z = self.drop(z)
        return self.norm(z + h)


class TextBranch(nn.Module):
    def __init__(self, model_name: str, projection_dim: int, dropout: float = 0.1):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj = ProjectionHead(self.encoder.config.hidden_size, projection_dim, dropout)

    def forward(self, texts: list[str], device: torch.device) -> torch.Tensor:
        tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        out = self.encoder(**tokens)
        cls = out.last_hidden_state[:, 0, :]
        return self.proj(cls)


class PPGBranch(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim),
            nn.LayerNorm(projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalStateEstimator(nn.Module):
    def __init__(
        self,
        text_model_name: str,
        ppg_input_dim: int,
        projection_dim: int = 256,
        projection_dropout: float = 0.1,
        ppg_hidden_dim: int = 128,
        fusion_hidden_dim: int = 256,
    ):
        super().__init__()
        self.text_branch = TextBranch(text_model_name, projection_dim, projection_dropout)
        self.ppg_branch = PPGBranch(ppg_input_dim, projection_dim, hidden_dim=ppg_hidden_dim, dropout=projection_dropout)

        self.fusion_head = nn.Sequential(
            nn.Linear(projection_dim * 2, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, 3),  # arousal, valence, cognitive_load_proxy
        )

    def forward(self, texts: list[str], ppg_features: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
        text_z = self.text_branch(texts, device)
        ppg_z = self.ppg_branch(ppg_features)
        fused = torch.cat([text_z, ppg_z], dim=-1)
        pred = self.fusion_head(fused)
        return {"text_z": text_z, "ppg_z": ppg_z, "pred": pred}
