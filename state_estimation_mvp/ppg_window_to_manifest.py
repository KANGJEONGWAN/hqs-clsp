"""
windowed PPG embeddings (10초 윈도우 단위) → ppg_feature_manifest.csv 변환
같은 세션(participant, playlist, video)의 윈도우 임베딩을 평균내어 세션당 1개로 합침.

sample_id 형식:
  p1_pl1_v0_Baseline_0_Baseline_Baseline_w000000_001250  (Baseline → 기본 스킵)
  p40_pl4_vV1_LVHA_V1_LVHA_Kidnapped_w018750_020000      (V1 → v1)
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def parse_session_key(sample_id: str) -> tuple[str, bool]:
    """
    윈도우 sample_id → (session_sample_id, is_baseline)

    p1_pl1_v0_..._0_..._w...   → ("p1_pl1_v0", True)
    p40_pl4_vV1_..._V1_..._w... → ("p40_pl4_v1", False)
    """
    parts = sample_id.split("_")
    participant = parts[0]   # p1, p40
    playlist   = parts[1]   # pl1, pl4
    video_raw  = parts[4]   # "0" for Baseline, "V1"/"V2"/... for videos

    if video_raw == "0":
        return f"{participant}_{playlist}_v0", True

    m = re.match(r"V(\d+)", video_raw)
    if m:
        return f"{participant}_{playlist}_v{int(m.group(1))}", False

    return f"{participant}_{playlist}_{video_raw}", False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="윈도우 PPG 임베딩을 세션 단위로 평균내어 ppg_feature_manifest.csv 생성"
    )
    parser.add_argument("--embeddings",        required=True, help=".npy 파일 경로 (N_windows, 512)")
    parser.add_argument("--sample-ids",        required=True, help="sample_id 목록 텍스트 파일")
    parser.add_argument("--out-csv",           default="Data_files/ppg_feature_manifest.csv")
    parser.add_argument("--include-baseline",  action="store_true", help="Baseline 세션 포함 여부")
    args = parser.parse_args()

    embeddings = np.load(args.embeddings)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (N, dim), got {embeddings.shape}")

    with open(args.sample_ids, encoding="utf-8") as f:
        sample_ids = [line.strip() for line in f if line.strip()]

    if len(sample_ids) != len(embeddings):
        raise ValueError(
            f"sample_id 수({len(sample_ids)})와 embedding 행 수({len(embeddings)})가 다릅니다"
        )

    session_embs: dict[str, list[np.ndarray]] = defaultdict(list)
    skipped = 0
    for sid, emb in zip(sample_ids, embeddings):
        key, is_baseline = parse_session_key(sid)
        if is_baseline and not args.include_baseline:
            skipped += 1
            continue
        session_embs[key].append(emb)

    print(f"윈도우: {len(sample_ids)} | Baseline 스킵: {skipped} | 세션: {len(session_embs)}")

    dim = embeddings.shape[1]
    rows = []
    for session_id, embs in sorted(session_embs.items()):
        avg = np.mean(embs, axis=0)
        row = {"sample_id": session_id, **{f"ppg_f{i}": float(avg[i]) for i in range(dim)}}
        rows.append(row)

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)

    print(f"저장 완료: {out} | 세션 수: {len(rows)} | 임베딩 차원: {dim}")


if __name__ == "__main__":
    main()
