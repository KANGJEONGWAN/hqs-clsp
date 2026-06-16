"""
학습된 CLSP 멀티모달 모델로 단일 샘플 추론.

사용법:
    python state_estimation_mvp/infer.py \
        --ppg-npy Data_files/ppg/embeddings_20260609_155151_p.npy \
        --digital-jsonl Data_files/digital/events_20260615_pc_summarized.jsonl \
                        Data_files/digital/events_20260615_mobile_summarized.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from model_multimodal import MultimodalStateEstimator
from text_context import build_text


_DEFAULT_CONTEXT = {
    "posture":                     "sitting",
    "movement":                    "sedentary",
    "social_engagement":           "low",
    "interpersonal_density":       "0",
    "device_interaction_behavior": "passive_viewing",
    "environment":                 "indoor",
    "temporal":                    "intermittent",
    "digital_summary":             "",
}


def load_ppg(npy_path: str) -> np.ndarray:
    emb = np.load(npy_path)
    if emb.ndim == 2:
        emb = emb.mean(axis=0)
    return emb.astype(np.float32)


def load_digital_summary(jsonl_paths: list[str]) -> str:
    summaries = []
    for path in jsonl_paths:
        if Path(path).exists():
            with open(path, encoding="utf-8") as f:
                record = json.loads(f.readline())
            s = record.get("llm_summary", "")
            if s:
                summaries.append(s)
    return " ".join(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          default="state_estimation_mvp/config_multimodal.yaml")
    parser.add_argument("--model-path",      default="outputs/state_estimation_mvp_multimodal/best.pt")
    parser.add_argument("--ppg-npy",         required=True)
    parser.add_argument("--digital-jsonl",   nargs="*", default=None)
    parser.add_argument("--posture",         default=_DEFAULT_CONTEXT["posture"])
    parser.add_argument("--movement",        default=_DEFAULT_CONTEXT["movement"])
    parser.add_argument("--social-engagement",           default=_DEFAULT_CONTEXT["social_engagement"])
    parser.add_argument("--interpersonal-density",       default=_DEFAULT_CONTEXT["interpersonal_density"])
    parser.add_argument("--device-interaction-behavior", default=_DEFAULT_CONTEXT["device_interaction_behavior"])
    parser.add_argument("--environment",     default=_DEFAULT_CONTEXT["environment"])
    parser.add_argument("--temporal",        default=_DEFAULT_CONTEXT["temporal"])
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    context = {
        "posture":                     args.posture,
        "movement":                    args.movement,
        "social_engagement":           args.social_engagement,
        "interpersonal_density":       args.interpersonal_density,
        "device_interaction_behavior": args.device_interaction_behavior,
        "environment":                 args.environment,
        "temporal":                    args.temporal,
        "digital_summary":             "",
    }

    if args.digital_jsonl:
        context["digital_summary"] = load_digital_summary(args.digital_jsonl)
        print(f"Digital summary: {context['digital_summary']}")

    text = build_text(context)
    print(f"Context text: {text}")

    ppg_vec = load_ppg(args.ppg_npy)
    ppg_dim = ppg_vec.shape[0]
    print(f"PPG dim: {ppg_dim}")

    device_str = cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    model = MultimodalStateEstimator(
        text_model_name=cfg["model"]["text_model_name"],
        ppg_input_dim=ppg_dim,
        projection_dim=int(cfg["model"]["projection_dim"]),
        projection_dropout=float(cfg["model"]["projection_dropout"]),
        ppg_hidden_dim=int(cfg["model"]["ppg_hidden_dim"]),
        fusion_hidden_dim=int(cfg["model"]["fusion_hidden_dim"]),
    ).to(device)

    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    ppg_tensor = torch.tensor(ppg_vec).unsqueeze(0).to(device)

    with torch.inference_mode():
        out = model(texts=[text], ppg_features=ppg_tensor, device=device)

    pred = out["pred"][0].cpu().tolist()
    print("\n=== 추론 결과 ===")
    print(f"Arousal:        {pred[0]:.4f}  (1-5 scale)")
    print(f"Valence:        {pred[1]:.4f}  (1-5 scale)")
    print(f"Cognitive Load: {pred[2]:.4f}  (0-1 scale)")


if __name__ == "__main__":
    main()
