"""
Build a training-ready JSON file from a downloaded LLaVA-Video-178K bucket.

Reads the parquet annotation files in `--data-dir`, filters to records that
have a resolvable video path, optionally subsamples, and writes a JSON file
in the format that train_xsa.py expects:

    [
      {
        "id": "...",
        "video": "relative/path/to/video.mp4",
        "conversations": [
          {"from": "human", "value": "<image>\\nWhat is happening?"},
          {"from": "gpt",   "value": "..."}
        ]
      },
      ...
    ]

Usage:
    python scripts/prepare_subset.py \
        --data-dir /workspace/data/train/LLaVA-Video-178K \
        --output   /workspace/data/train/llava_video_30_60s_subset.json \
        --max-samples 50000
"""

import argparse
import json
import os
import random
from pathlib import Path

import pandas as pd


def normalize_conversations(raw):
    """LLaVA-Video-178K conversations are stored as JSON strings or arrays.
    Normalize to a list of {"from": role, "value": text} dicts."""
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    if not isinstance(raw, list):
        return None
    out = []
    for turn in raw:
        if not isinstance(turn, dict):
            return None
        role = turn.get("from") or turn.get("role")
        text = turn.get("value") or turn.get("content")
        if role is None or text is None:
            return None
        out.append({"from": role, "value": text})
    return out if out else None


def main(args):
    random.seed(args.seed)
    root = Path(args.data_dir)
    if not root.is_dir():
        raise SystemExit(f"data dir not found: {root}")

    parquet_files = sorted(root.rglob("*.parquet"))
    json_files = sorted(root.rglob("*.json"))
    print(f"[scan] {len(parquet_files)} parquet, {len(json_files)} json files")

    all_records: list[dict] = []

    for pf in parquet_files:
        try:
            df = pd.read_parquet(pf)
        except Exception as e:
            print(f"  skip {pf.name}: {e}")
            continue

        # Different shards have slightly different schemas; try common names.
        video_col = next(
            (c for c in df.columns if c.lower() in ("video", "video_path", "video_file")),
            None,
        )
        conv_col = next(
            (c for c in df.columns if c.lower() in ("conversations", "conversation")),
            None,
        )
        if video_col is None or conv_col is None:
            print(f"  skip {pf.name}: missing video/conversations column "
                  f"(have {list(df.columns)[:6]}...)")
            continue

        for _, row in df.iterrows():
            video = row[video_col]
            conv = normalize_conversations(row[conv_col])
            if not video or not conv:
                continue
            all_records.append({
                "id": row.get("id", str(len(all_records))),
                "video": video,
                "conversations": conv,
                "_source_parquet": pf.name,
            })
        print(f"  + {pf.name}: {len(df)} rows")

    print(f"[total] {len(all_records)} records before filtering")

    # Verify videos exist on disk; drop missing ones
    if not args.skip_existence_check:
        before = len(all_records)
        all_records = [
            r for r in all_records
            if os.path.isfile(os.path.join(root, r["video"]))
        ]
        dropped = before - len(all_records)
        if dropped:
            print(f"[filter] dropped {dropped} records with missing video files")

    # Subsample
    if args.max_samples and len(all_records) > args.max_samples:
        random.shuffle(all_records)
        all_records = all_records[: args.max_samples]
        print(f"[sample] randomly subsampled to {len(all_records)}")

    # Strip the debug column
    for r in all_records:
        r.pop("_source_parquet", None)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_records, f, indent=2)

    size_mb = output.stat().st_size / 1e6
    print(f"[done] wrote {len(all_records)} records to {output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-existence-check",
        action="store_true",
        help="Don't verify that video files exist on disk (fast scan only)",
    )
    args = parser.parse_args()
    main(args)
