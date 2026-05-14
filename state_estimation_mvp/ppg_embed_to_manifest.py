from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PaPaGEI .npy 임베딩 + sample_id 목록 → ppg_feature_manifest.csv 변환"
    )
    parser.add_argument("--embeddings", required=True, help="papagei-s.py / papagei-p.py 출력 .npy 파일 경로 (N, 512)")
    parser.add_argument("--sample-ids", required=True, help="sample_id 목록 텍스트 파일 (한 줄에 하나, e.g. p1_pl1_v1)")
    parser.add_argument("--out-csv", default="Data_files/ppg_feature_manifest.csv")
    args = parser.parse_args()

    embeddings = np.load(args.embeddings)
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D (N, dim), got shape {embeddings.shape}")

    with open(args.sample_ids, encoding="utf-8") as f:
        sample_ids = [line.strip() for line in f if line.strip()]

    if len(sample_ids) != len(embeddings):
        raise ValueError(f"sample_id 수({len(sample_ids)})와 embedding 행 수({len(embeddings)})가 다릅니다")

    dim = embeddings.shape[1]
    cols = {f"ppg_f{i}": embeddings[:, i] for i in range(dim)}
    df = pd.DataFrame({"sample_id": sample_ids, **cols})

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"저장 완료: {out}")
    print(f"Rows: {len(df)} | Embedding dim: {dim}")


if __name__ == "__main__":
    main()
