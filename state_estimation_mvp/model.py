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
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        out = self.encoder(**tokens)
        cls = out.last_hidden_state[:, 0, :]
        return self.proj(cls)


class TextOnlyStateEstimator(nn.Module):
    def __init__(
        self,
        text_model_name: str = "distilbert-base-uncased",
        projection_dim: int = 256,
        dropout: float = 0.1,
        reg_hidden_dim: int = 256,
    ):
        super().__init__()
        self.text_branch = TextBranch(text_model_name, projection_dim, dropout)

        self.reg_head = nn.Sequential(
            nn.Linear(projection_dim, reg_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reg_hidden_dim, 3),  # [arousal, valence, cognitive_load_proxy]
        )

    def forward(self, texts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
        text_z = self.text_branch(texts, device)
        pred = self.reg_head(text_z)

        return {
            "text_z": text_z,
            "pred": pred,
        }
