from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import json 
from torch.utils.data import Dataset

from text_context import build_text, validate_context, validate_context_text, context_from_state


def _to_int(value: Any) -> int:
    return int(float(value))


def _build_sample_id(participant_id: int, playlist_id: int, video_id: int) -> str:
    return f"p{participant_id}_pl{playlist_id}_v{video_id}"


def _context_from_cma(cma: str) -> dict[str, str]:
    # EEVR 실험은 전원 VR 시청 세팅 (sitting/sedentary/low/0/passive_viewing 고정)
    # CMA에 따라 environment와 temporal만 달라짐
    mapping: dict[str, dict[str, str]] = {
        "baseline": {
            "posture": "sitting",
            "movement": "sedentary",
            "social_engagement": "low",
            "interpersonal_density": "0",
            "device_interaction_behavior": "passive_viewing",
            "environment": "quiet",
            "temporal": "brief",
            "digital_summary": "",  # 추가
        },
        "lvla": {
            "posture": "sitting",
            "movement": "sedentary",
            "social_engagement": "low",
            "interpersonal_density": "0",
            "device_interaction_behavior": "passive_viewing",
            "environment": "indoor",
            "temporal": "intermittent",
            "digital_summary": "",  # 추가
        },
        "lvha": {
            "posture": "sitting",
            "movement": "sedentary",
            "social_engagement": "low",
            "interpersonal_density": "0",
            "device_interaction_behavior": "passive_viewing",
            "environment": "dynamic",
            "temporal": "sustained",
            "digital_summary": "",  # 추가
        },
        "hvla": {
            "posture": "sitting",
            "movement": "sedentary",
            "social_engagement": "low",
            "interpersonal_density": "0",
            "device_interaction_behavior": "passive_viewing",
            "environment": "outdoor",
            "temporal": "sustained",
            "digital_summary": "",  # 추가
        },
        "hvha": {
            "posture": "sitting",
            "movement": "sedentary",
            "social_engagement": "low",
            "interpersonal_density": "0",
            "device_interaction_behavior": "passive_viewing",
            "environment": "crowded",
            "temporal": "continuous",
            "digital_summary": "",  # 추가
        },
    }

    key = str(cma).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unknown CMA label for context mapping: {cma}")
    return mapping[key]


def build_text_only_manifest(
    text_csv: str | Path,
    vads_csv: str | Path,
    output_csv: str | Path,
    use_structured_context: bool = True,
    drop_invalid_context: bool = False,
    digital_summary_jsonl: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build text-only state-estimation manifest.

        Output columns:
            sample_id, text, arousal, valence, cognitive_load_proxy,
      participant_id, playlist_id, video_id,
      physical_context, social_context, task_context,
      digital_context, environment_context, temporal_context,
      context_signature
    """
    # digital_summary 로드
    digital_summary = ""
    if digital_summary_jsonl and Path(digital_summary_jsonl).exists():
        with open(digital_summary_jsonl, encoding="utf-8") as f:
            record = json.loads(f.readline())
            digital_summary = record.get("llm_summary", "")
            
    text_df = pd.read_csv(text_csv)
    vads_df = pd.read_csv(vads_csv)

    required_text = {"Participant ID", "Playlist ID", "Video ID", "Text Description", "CMA"}
    required_vads = {"Participant ID", "Playlist ID", "Video ID", "Arousal", "Valence", "significance"}

    miss_text = required_text.difference(set(text_df.columns))
    miss_vads = required_vads.difference(set(vads_df.columns))
    if miss_text:
        raise ValueError(f"Text CSV missing columns: {sorted(miss_text)}")
    if miss_vads:
        raise ValueError(f"VADS CSV missing columns: {sorted(miss_vads)}")

    text_df = text_df.copy()
    vads_df = vads_df.copy()

    for col in ["Participant ID", "Playlist ID", "Video ID"]:
        text_df[col] = text_df[col].apply(_to_int)
        vads_df[col] = vads_df[col].apply(_to_int)

    text_df = text_df.rename(
        columns={
            "Participant ID": "participant_id",
            "Playlist ID": "playlist_id",
            "Video ID": "video_id",
            "Text Description": "raw_text",
            "CMA": "cma",
        }
    )

    vads_df = vads_df.rename(
        columns={
            "Participant ID": "participant_id",
            "Playlist ID": "playlist_id",
            "Video ID": "video_id",
            "Arousal": "arousal",
            "Valence": "valence",
            "significance": "cognitive_load_proxy",
        }
    )

    # Significance is used as a proxy for cognitive load.
    # Normalize 1~5 scale to 0~1 for stable regression.
    vads_df["cognitive_load_proxy"] = pd.to_numeric(vads_df["cognitive_load_proxy"], errors="coerce")
    vads_df["cognitive_load_proxy"] = ((vads_df["cognitive_load_proxy"] - 1.0) / 4.0).clip(0.0, 1.0)

    merged = pd.merge(
        text_df[["participant_id", "playlist_id", "video_id", "raw_text", "cma"]],
        vads_df[["participant_id", "playlist_id", "video_id", "arousal", "valence", "cognitive_load_proxy"]],
        on=["participant_id", "playlist_id", "video_id"],
        how="inner",
        validate="many_to_one",
    )

    merged = merged.dropna(subset=["arousal", "valence", "cognitive_load_proxy"]).copy()

    merged = merged.drop_duplicates(subset=["participant_id", "playlist_id", "video_id"], keep="first")

    slot_rows: list[dict[str, str]] = []
    built_texts: list[str] = []
    validity: list[bool] = []
    issues_list: list[str] = []

    for _, row in merged.iterrows():
        raw = str(row["raw_text"])
        if use_structured_context:
            context = _context_from_cma(str(row["cma"]))
            context["digital_summary"] = digital_summary
            validate_context(context)
            text = build_text(context)
            slot_rows.append(
                {
                    "posture":                     context["posture"],
                    "movement":                    context["movement"],
                    "social_engagement":           context["social_engagement"],
                    "interpersonal_density":       context["interpersonal_density"],
                    "device_interaction_behavior": context["device_interaction_behavior"],
                    "environment":                 context["environment"],
                    "temporal":                    context["temporal"],
                    "digital_summary":             context.get("digital_summary", ""),  # 추가
                }
            )
        else:
            text = raw
            slot_rows.append(
                {
                    "posture": "",
                    "movement": "",
                    "social_engagement": "",
                    "interpersonal_density": "",
                    "device_interaction_behavior": "",
                    "environment": "",
                    "temporal": "",
                    "digital_summary": "",  # context.get 대신 그냥 빈 문자열
                }
            )

        ok, issues = validate_context_text(text)
        validity.append(ok)
        issues_list.append("; ".join(issues))
        built_texts.append(text)

    slots_df = pd.DataFrame(slot_rows)
    out = pd.concat([merged.reset_index(drop=True), slots_df], axis=1)
    out["text"] = built_texts
    out["context_valid"] = validity
    out["context_issues"] = issues_list

    if drop_invalid_context:
        out = out[out["context_valid"]].copy()

    out["sample_id"] = out.apply(
        lambda r: _build_sample_id(int(r["participant_id"]), int(r["playlist_id"]), int(r["video_id"])),
        axis=1,
    )

    out["context_signature"] = out.apply(
        lambda r: "|".join([
            str(r["posture"]),
            str(r["movement"]),
            str(r["social_engagement"]),
            str(r["interpersonal_density"]),
            str(r["device_interaction_behavior"]),
            str(r["environment"]),
            str(r["temporal"]),
        ]),
        axis=1,
    )

    out = out[
        [
            "sample_id",
            "text",
            "arousal",
            "valence",
            "cognitive_load_proxy",
            "participant_id",
            "playlist_id",
            "video_id",
            "posture",
            "movement",
            "social_engagement",
            "interpersonal_density",
            "device_interaction_behavior",
            "environment",
            "temporal",
            "digital_summary",  # 추가
            "context_signature",
            "context_valid",
            "context_issues",
            "raw_text",
        ]
    ].reset_index(drop=True)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def split_manifest_random(
    manifest_csv: str | Path,
    train_csv: str | Path,
    val_csv: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(manifest_csv)
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")

    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)

    n_val = max(1, int(len(df) * val_ratio))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_path = Path(train_csv)
    val_path = Path(val_csv)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    return train_df, val_df


def split_manifest_zero_shot_by_context(
    manifest_csv: str | Path,
    train_csv: str | Path,
    val_csv: str | Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Zero-shot split by holding out context signatures in validation.
    Validation context combinations do not appear in training.
    """

    df = pd.read_csv(manifest_csv)
    if "context_signature" not in df.columns:
        raise ValueError("context_signature column is required for zero-shot split")

    signatures = df["context_signature"].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(signatures)

    n_hold = max(1, int(len(signatures) * val_ratio))
    holdout = set(signatures[:n_hold])

    val_df = df[df["context_signature"].isin(holdout)].reset_index(drop=True)
    train_df = df[~df["context_signature"].isin(holdout)].reset_index(drop=True)

    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError("Zero-shot split produced an empty train or val set. Adjust val_ratio.")

    train_path = Path(train_csv)
    val_path = Path(val_csv)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    overlap = set(train_df["context_signature"].unique()).intersection(set(val_df["context_signature"].unique()))
    if len(overlap) != 0:
        raise RuntimeError("Zero-shot split failed: train/val context signatures overlap")

    return train_df, val_df


class StateEstimationDataset(Dataset):
    """
    Text-only dataset.

    Expected CSV columns:
      - sample_id
      - text
      - arousal
      - valence
            - cognitive_load_proxy
    """

    def __init__(self, csv_path: str | Path):
        self.df = pd.read_csv(csv_path)
        required = {"sample_id", "text", "arousal", "valence", "cognitive_load_proxy"}
        missing = required.difference(set(self.df.columns))
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        return {
            "sample_id": str(row["sample_id"]),
            "text": str(row["text"]),
            "targets": torch.tensor(
                [
                    float(row["arousal"]),
                    float(row["valence"]),
                    float(row["cognitive_load_proxy"]),
                ],
                dtype=torch.float32,
            ),
        }
