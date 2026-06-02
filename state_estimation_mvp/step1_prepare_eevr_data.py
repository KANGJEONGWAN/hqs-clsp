from __future__ import annotations

import argparse
from pathlib import Path

from dataset import build_text_only_manifest, split_manifest_random, split_manifest_zero_shot_by_context


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-csv", type=str, default="Data_files/Textdata.csv")
    parser.add_argument("--vads-csv", type=str, default="Data_files/VADS.csv")
    parser.add_argument("--manifest-csv", type=str, default="Data_files/text_only_manifest.csv")
    parser.add_argument("--train-csv", type=str, default="Data_files/train_text_only_zeroshot.csv")
    parser.add_argument("--val-csv", type=str, default="Data_files/val_text_only_zeroshot.csv")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-raw-text", action="store_true")
    parser.add_argument("--keep-invalid-context", action="store_true")
    parser.add_argument("--split", choices=["zero-shot", "random"], default="zero-shot")
    parser.add_argument("--digital-summary-jsonl", type=str, default=None)
    args = parser.parse_args()

    manifest = build_text_only_manifest(
        text_csv=args.text_csv,
        vads_csv=args.vads_csv,
        output_csv=args.manifest_csv,
        use_structured_context=not args.use_raw_text,
        drop_invalid_context=not args.keep_invalid_context,
        digital_summary_jsonl=args.digital_summary_jsonl,
    )

    if args.split == "zero-shot":
        train_df, val_df = split_manifest_zero_shot_by_context(
            manifest_csv=args.manifest_csv,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
        train_df, val_df = split_manifest_random(
            manifest_csv=args.manifest_csv,
            train_csv=args.train_csv,
            val_csv=args.val_csv,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )

    print(f"Manifest rows: {len(manifest)}")
    print(f"Train rows: {len(train_df)} | Val rows: {len(val_df)}")
    if "context_signature" in train_df.columns and "context_signature" in val_df.columns:
        inter = set(train_df["context_signature"].unique()).intersection(set(val_df["context_signature"].unique()))
        print(f"Shared context signatures (train ∩ val): {len(inter)}")
    print(f"Saved: {Path(args.manifest_csv)}")
    print(f"Saved: {Path(args.train_csv)}")
    print(f"Saved: {Path(args.val_csv)}")


if __name__ == "__main__":
    main()
