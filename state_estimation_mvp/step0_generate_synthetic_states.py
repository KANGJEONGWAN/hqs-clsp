from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from state_schema import ALLOWED_STATE_VALUES, state_signature


STATE_KEYS = [
    "posture",
    "movement",
    "social_engagement",
    "device_interaction_behavior",
    "environment",
    "temporal",
]

DIGITAL_SUMMARY_SAMPLES = [
    # PC 패턴
    "Coding in VSCode while switching between Chrome and browser for reference.",
    "Reading research papers on Chrome and taking notes on Notion.",
    "Browsing academic papers and switching between browser tabs frequently.",
    "Writing code and searching documentation across multiple browser windows.",
    "Checking Notion tasks while switching between VSCode and browser.",
    # 모바일 패턴
    "Watching YouTube videos for about a minute on mobile.",
    "Searching on Naver for several minutes on mobile.",
    "Briefly checking app settings on mobile.",
    "Switching between YouTube and search apps on mobile.",
]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-csv", type=str, default="Data_files/text_only_manifest.csv")
    parser.add_argument("--out-csv", type=str, default="Data_files/session_state_synth.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest_csv)
    if "sample_id" not in df.columns:
        raise ValueError("sample_id column is required")

    rng = np.random.default_rng(args.seed)

    rows: list[dict[str, str]] = []
    for sid in df["sample_id"].astype(str).tolist():
        state = {k: ALLOWED_STATE_VALUES[k][int(rng.integers(0, len(ALLOWED_STATE_VALUES[k])))] for k in STATE_KEYS}
        sig = "|".join(state_signature(state))
        rows.append({"sample_id": sid, **state, "state_signature": sig})

    out = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Rows: {len(out)}")
    print(f"Unique state signatures: {out['state_signature'].nunique()}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
