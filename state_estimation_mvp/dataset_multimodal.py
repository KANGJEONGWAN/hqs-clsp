from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset


class MultimodalStateEstimationDataset(Dataset):
    """
    Multimodal dataset: text + ppg_features -> targets

    Expected text CSV columns:
      sample_id, text, arousal, valence, cognitive_load_proxy

    Expected ppg CSV columns:
      sample_id, ppg_f0, ppg_f1, ...
    """

    def __init__(
        self,
        text_csv: str | Path,
        ppg_csv: str | Path,
        allow_missing_ppg: bool = False,
    ):
        text_df = pd.read_csv(text_csv)
        ppg_df = pd.read_csv(ppg_csv)

        req_text = {"sample_id", "text", "arousal", "valence", "cognitive_load_proxy"}
        miss_text = req_text.difference(set(text_df.columns))
        if miss_text:
            raise ValueError(f"text_csv missing columns: {sorted(miss_text)}")

        if "sample_id" not in ppg_df.columns:
            raise ValueError("ppg_csv must contain sample_id")

        self.ppg_cols = [c for c in ppg_df.columns if c != "sample_id"]
        if not self.ppg_cols:
            raise ValueError("ppg_csv must contain at least one ppg feature column")

        how = "left" if allow_missing_ppg else "inner"
        merged = text_df.merge(ppg_df, on="sample_id", how=how)
        if len(merged) == 0:
            raise ValueError("No rows after joining text_csv and ppg_csv on sample_id")

        merged[self.ppg_cols] = merged[self.ppg_cols].apply(pd.to_numeric, errors="coerce")
        if allow_missing_ppg:
            merged[self.ppg_cols] = merged[self.ppg_cols].fillna(0.0)
        else:
            merged = merged.dropna(subset=self.ppg_cols).copy()

        self.df = merged.reset_index(drop=True)

    @property
    def ppg_dim(self) -> int:
        return len(self.ppg_cols)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        ppg_features = torch.tensor([float(row[c]) for c in self.ppg_cols], dtype=torch.float32)
        targets = torch.tensor(
            [
                float(row["arousal"]),
                float(row["valence"]),
                float(row["cognitive_load_proxy"]),
            ],
            dtype=torch.float32,
        )
        return {
            "sample_id": str(row["sample_id"]),
            "text": str(row["text"]),
            "ppg_features": ppg_features,
            "targets": targets,
        }
