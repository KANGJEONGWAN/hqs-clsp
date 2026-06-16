"""
finalize.py — A-2 datastream -> final datastream(memory object) 생성.

핵심 기능
1) 기본 모드(window-level)
   - 각 record의 raw PPG 10초 window(1250 samples)를 PaPaGEI로 512-d embedding 변환
   - text_context + 512-d embedding을 학습된 MultimodalStateEstimator에 넣어
     arousal / valence / cognitive_load 예측

2) session/video pooling 모드(--pool-windows)
   - EEVR처럼 label/text는 영상당 1개인데 PPG는 10초 window 여러 개로 쪼개진 경우 사용
   - 같은 session/video에 속한 window embedding들을 mean pooling
   - 영상당 1개의 512-d physiological representation으로 예측

사용 예시

# 1) mock으로 스키마만 점검
python finalize.py --in datastream.jsonl --out final_mock.jsonl --mock

# 2) window-level 실제 추론
python finalize.py --in datastream.jsonl --out final_datastream.jsonl \
  --ckpt outputs/state_estimation_mvp_multimodal/best.pt \
  --mvp-dir state_estimation_mvp \
  --papagei-root ./papagei-foundation-model \
  --papagei-ckpt ./papagei-foundation-model/weights/papagei_s.pt \
  --papagei-variant s

# 3) 이미 embedding이 record 안에 있을 때
python finalize.py --in datastream.jsonl --out final_datastream.jsonl \
  --embedding-only \
  --ckpt outputs/state_estimation_mvp_multimodal/best.pt \
  --mvp-dir state_estimation_mvp
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

PAPAGEI_DIM = 512
TARGET_PPG_SAMPLES = 1250


# ---------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------
def clamp01(v: float | None) -> float | None:
    if v is None:
        return None
    try:
        return round(max(0.0, min(1.0, float(v))), 3)
    except (TypeError, ValueError):
        return None


def mean_vectors(vectors: list[list[float]], dim: int = PAPAGEI_DIM) -> list[float]:
    if not vectors:
        return [0.0] * dim
    n = len(vectors)
    out = [0.0] * len(vectors[0])
    for vec in vectors:
        if len(vec) != len(out):
            raise ValueError(f"embedding dimension mismatch: expected {len(out)}, got {len(vec)}")
        for i, value in enumerate(vec):
            out[i] += float(value)
    return [v / n for v in out]


def parse_embedding_from_record(rec: dict[str, Any]) -> list[float] | None:
    for key in ("ppg_embedding", "embedding", "ppg_features"):
        val = rec.get(key)
        if isinstance(val, list) and len(val) == PAPAGEI_DIM:
            return [float(x) for x in val]

    ppg_f_keys = [f"ppg_f{i}" for i in range(PAPAGEI_DIM)]
    if all(k in rec for k in ppg_f_keys):
        return [float(rec[k]) for k in ppg_f_keys]

    return None


def derive_session_id(rec: dict[str, Any], fields: list[str] | None = None, explicit_field: str | None = None) -> str:
    if explicit_field and rec.get(explicit_field) not in (None, ""):
        return str(rec[explicit_field])

    if fields:
        parts = []
        for field in fields:
            value = rec.get(field)
            if value is None and isinstance(rec.get("meta"), dict):
                value = rec["meta"].get(field)
            if value is None:
                value = "NA"
            safe_field = re.sub(r"\s+", "", field)
            safe_value = re.sub(r"\s+", "", str(value))
            parts.append(f"{safe_field}{safe_value}")
        return "_".join(parts)

    for key in ("session_id", "video_sample_id", "base_sample_id"):
        if rec.get(key) not in (None, ""):
            return str(rec[key])

    sid = str(rec.get("sample_id") or rec.get("Label") or rec.get("label") or rec.get("window_end_ms") or "sample")
    sid = re.sub(r"_w\d+(?:_\d+)?$", "", sid)
    sid = re.sub(r"_window\d+$", "", sid)
    return sid


def get_text_context(rec: dict[str, Any]) -> str:
    if rec.get("text_context"):
        return str(rec["text_context"])
    if rec.get("text"):
        return str(rec["text"])
    chunks = []
    for key in ("physical", "social", "digital"):
        val = rec.get(key)
        if val not in (None, ""):
            chunks.append(f"{key}: {val}")
    return ". ".join(chunks)


# ---------------------------------------------------------------------
# Seam 1) PaPaGEI embedding
# ---------------------------------------------------------------------
class PaPaGEIEmbedder:
    def __init__(self, root: str | Path, ckpt: str | Path, variant: str = "s", device: str | None = None):
        import numpy as np
        import torch

        self.np = np
        self.torch = torch
        self.root = Path(root)
        self.ckpt = Path(ckpt)
        self.variant = variant.lower()

        if not self.root.exists():
            raise FileNotFoundError(f"PaPaGEI root not found: {self.root}")
        if not self.ckpt.exists():
            raise FileNotFoundError(f"PaPaGEI checkpoint not found: {self.ckpt}")

        sys.path.insert(0, str(self.root))
        from linearprobing.utils import load_model_without_module_prefix
        from models.resnet import ResNet1D, ResNet1DMoE

        if self.variant == "p":
            model = ResNet1D(
                in_channels=1, base_filters=32, kernel_size=3,
                stride=2, groups=1, n_block=18, n_classes=PAPAGEI_DIM,
            )
        elif self.variant == "s":
            model = ResNet1DMoE(
                in_channels=1, base_filters=32, kernel_size=3,
                stride=2, groups=1, n_block=18, n_classes=PAPAGEI_DIM, n_experts=3,
            )
        else:
            raise ValueError("--papagei-variant must be 'p' or 's'")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_model_without_module_prefix(model, str(self.ckpt))
        self.model.to(self.device)
        self.model.eval()

    def embed(self, ppg: list[float]) -> list[float]:
        if len(ppg) != TARGET_PPG_SAMPLES:
            raise ValueError(f"Expected {TARGET_PPG_SAMPLES} PPG samples, got {len(ppg)}")
        arr = self.np.asarray(ppg, dtype=self.np.float32)[None, None, :]
        x = self.torch.from_numpy(arr).to(self.device)
        with self.torch.inference_mode():
            outputs = self.model(x)
            emb = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
            emb = emb.detach().cpu().numpy()[0]
        return [float(v) for v in emb.tolist()]


class MockEmbedder:
    def embed(self, ppg: list[float]) -> list[float]:
        if not ppg:
            return [0.0] * PAPAGEI_DIM
        mean = sum(float(x) for x in ppg) / len(ppg)
        rng = (max(ppg) - min(ppg)) or 1.0
        return [round(((mean + i * 0.001) % rng) / rng, 4) for i in range(PAPAGEI_DIM)]


# ---------------------------------------------------------------------
# Seam 2) Estimation model
# ---------------------------------------------------------------------
class EstimationModel:
    def __init__(self, ckpt: str | Path, mvp_dir: str | Path = ".",
                 text_model_name: str = "distilbert-base-uncased",
                 projection_dim: int = 256, projection_dropout: float = 0.1,
                 fusion_hidden_dim: int = 256, device: str | None = None):
        import torch

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        sys.path.insert(0, str(Path(mvp_dir) / "state_estimation_mvp"))
        from model_multimodal import MultimodalStateEstimator

        self.model = MultimodalStateEstimator(
            text_model_name=text_model_name,
            ppg_input_dim=PAPAGEI_DIM,
            projection_dim=projection_dim,
            projection_dropout=projection_dropout,
            fusion_hidden_dim=fusion_hidden_dim,
        ).to(self.device)

        state = torch.load(str(ckpt), map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, text_context: str, ppg_embedding: list[float]) -> dict[str, float | None]:
        torch = self.torch
        if len(ppg_embedding) != PAPAGEI_DIM:
            raise ValueError(f"Expected {PAPAGEI_DIM}-d embedding, got {len(ppg_embedding)}")
        with torch.inference_mode():
            emb = torch.tensor([ppg_embedding], dtype=torch.float32, device=self.device)
            out = self.model(texts=[text_context], ppg_features=emb, device=self.device)
            # 출력 순서: [arousal, valence, cognitive_load_proxy]
            # arousal/valence: 1-5 스케일, cognitive_load: 0-1 스케일
            arousal, valence, cog = out["pred"][0].detach().cpu().tolist()
        return {
            "cognitive_load": clamp01(cog),
            "valence":        round(float(valence), 3),
            "arousal":        round(float(arousal), 3),
        }


class MockModel:
    def predict(self, text_context: str, ppg_embedding: list[float]) -> dict[str, float | None]:
        if not ppg_embedding:
            return {"cognitive_load": None, "valence": None, "arousal": None}
        m = sum(ppg_embedding) / len(ppg_embedding)
        return {
            "cognitive_load": round(min(1.0, m), 3),
            "valence":        round(min(5.0, 1.0 + m * 4.0), 3),
            "arousal":        round(min(5.0, 1.0 + (m * 1.3 % 1.0) * 4.0), 3),
        }


# ---------------------------------------------------------------------
# output helpers
# ---------------------------------------------------------------------
def get_embedding_for_record(rec: dict[str, Any], embedder) -> list[float]:
    existing = parse_embedding_from_record(rec)
    if existing is not None:
        return existing
    ppg = rec.get("ppg", []) or []
    return embedder.embed([float(x) for x in ppg])


def finalize_window_level(records: list[dict[str, Any]], embedder, model, args) -> list[dict[str, Any]]:
    out_records = []
    for idx, rec in enumerate(records):
        rec = copy.deepcopy(rec)
        rec.setdefault("sample_id", str(rec.get("window_end_ms") or rec.get("timestamp") or idx))
        emb = get_embedding_for_record(rec, embedder)
        measures = model.predict(get_text_context(rec), emb)
        rec["measures"] = measures
        if args.keep_embedding:
            rec["ppg_embedding"] = emb
        if not args.keep_ppg:
            rec.pop("ppg", None)
        out_records.append(rec)
    return out_records


def finalize_pooled_sessions(records: list[dict[str, Any]], embedder, model, args) -> list[dict[str, Any]]:
    groups: OrderedDict[str, dict[str, Any]] = OrderedDict()

    for idx, rec in enumerate(records):
        session_id = derive_session_id(rec, fields=args.session_id_fields, explicit_field=args.session_id_field)
        emb = get_embedding_for_record(rec, embedder)
        if session_id not in groups:
            rep = copy.deepcopy(rec)
            rep["sample_id"] = session_id
            rep["session_id"] = session_id
            groups[session_id] = {"representative": rep, "embeddings": [], "window_sample_ids": []}
        groups[session_id]["embeddings"].append(emb)
        groups[session_id]["window_sample_ids"].append(str(rec.get("sample_id") or rec.get("window_end_ms") or idx))

    out_records = []
    for session_id, pack in groups.items():
        rec = pack["representative"]
        pooled = mean_vectors(pack["embeddings"])
        measures = model.predict(get_text_context(rec), pooled)
        rec["measures"] = measures
        rec["pooling"] = {"method": "mean", "unit": "session/video", "window_count": len(pack["embeddings"])}
        if args.keep_window_ids:
            rec["source_window_sample_ids"] = pack["window_sample_ids"]
        if args.keep_embedding:
            rec["ppg_embedding"] = pooled
        if not args.keep_ppg:
            rec.pop("ppg", None)
        out_records.append(rec)

    return out_records


# ---------------------------------------------------------------------
def build_embedder_and_model(args):
    if args.mock:
        return MockEmbedder(), MockModel()

    if not args.ckpt:
        raise SystemExit("[finalize] 실제 모드는 --ckpt 필요. 스키마 점검만 할 거면 --mock 사용")

    if args.embedding_only:
        embedder = MockEmbedder()
    else:
        if not args.papagei_root or not args.papagei_ckpt:
            raise SystemExit("[finalize] raw PPG를 embedding하려면 --papagei-root와 --papagei-ckpt 필요. 이미 embedding이 있으면 --embedding-only 사용")
        embedder = PaPaGEIEmbedder(
            root=args.papagei_root,
            ckpt=args.papagei_ckpt,
            variant=args.papagei_variant,
            device=args.device,
        )

    model = EstimationModel(
        ckpt=args.ckpt,
        mvp_dir=args.mvp_dir,
        text_model_name=args.text_model_name,
        device=args.device,
    )
    return embedder, model


def validate_embedding_only(records: list[dict[str, Any]]) -> None:
    missing = []
    for idx, rec in enumerate(records):
        if parse_embedding_from_record(rec) is None:
            missing.append(str(rec.get("sample_id") or rec.get("window_end_ms") or idx))
            if len(missing) >= 5:
                break
    if missing:
        raise SystemExit(
            "[finalize] --embedding-only인데 512-d embedding이 없는 record가 있음: "
            + ", ".join(missing)
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)

    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--embedding-only", action="store_true")
    ap.add_argument("--pool-windows", action="store_true")
    ap.add_argument("--session-id-field", default=None)
    ap.add_argument("--session-id-fields", nargs="*", default=None)
    ap.add_argument("--keep-window-ids", action="store_true")

    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--mvp-dir", default=".")
    ap.add_argument("--text-model-name", default="distilbert-base-uncased")
    ap.add_argument("--device", default=None)

    ap.add_argument("--papagei-root", default=None)
    ap.add_argument("--papagei-ckpt", default=None)
    ap.add_argument("--papagei-variant", choices=["p", "s"], default="s")

    ap.add_argument("--keep-ppg", action="store_true")
    ap.add_argument("--keep-embedding", action="store_true")
    args = ap.parse_args()

    with args.inp.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise SystemExit(f"[finalize] 입력이 비었음: {args.inp}")

    if args.embedding_only:
        validate_embedding_only(records)

    embedder, model = build_embedder_and_model(args)

    if args.pool_windows:
        out_records = finalize_pooled_sessions(records, embedder, model, args)
    else:
        out_records = finalize_window_level(records, embedder, model, args)

    with args.out.open("w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_filled = sum(
        1 for r in out_records
        if any((r.get("measures") or {}).get(k) is not None for k in ("arousal", "valence", "cognitive_load"))
    )
    print(f"[finalize] 완료 -> {args.out}")
    print(f"  입력 records     : {len(records)}")
    print(f"  출력 records     : {len(out_records)}")
    print(f"  pooling          : {'mean by session/video' if args.pool_windows else 'off(window-level)'}")
    print(f"  measures 채워짐  : {n_filled}/{len(out_records)}")
    print(f"  embedding source : {'existing record embedding' if args.embedding_only else ('mock' if args.mock else 'PaPaGEI')}")
    print(f"  model            : {'mock' if args.mock else 'MultimodalStateEstimator'}")


if __name__ == "__main__":
    main()
