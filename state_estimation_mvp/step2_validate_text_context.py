from __future__ import annotations

import argparse

import pandas as pd

from text_context import validate_context_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="Data_files/text_only_manifest.csv")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--show", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns:
        raise ValueError(f"Column not found: {args.text_col}")

    total = len(df)
    bad = []
    for i, row in df.iterrows():
        ok, issues = validate_context_text(str(row[args.text_col]))
        if not ok:
            bad.append((i, issues, str(row[args.text_col])[:220]))

    print(f"Total texts: {total}")
    print(f"Valid texts: {total - len(bad)}")
    print(f"Invalid texts: {len(bad)}")

    if bad:
        print("\nExamples of invalid texts:")
        for i, issues, preview in bad[: args.show]:
            print(f"- row={i}, issues={issues}\n  text={preview}")

    print("\nSample valid texts:")
    shown = 0
    for t in df[args.text_col].astype(str).tolist():
        ok, _ = validate_context_text(t)
        if ok:
            print(f"- {t[:220]}")
            shown += 1
            if shown >= args.show:
                break


if __name__ == "__main__":
    main()
