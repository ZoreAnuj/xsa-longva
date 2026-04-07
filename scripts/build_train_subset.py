"""
Build a unified training subset JSON from the LLaVA-Video-178K bucket dirs.

The downloaded structure (per bucket) is:
    /data/train/LLaVA-Video-178K/30_60_s_academic_v0_1/
        30_60_s_academic_mc_v0_1_qa_processed.json   # multi-choice QA
        30_60_s_academic_oe_v0_1_qa_processed.json   # open-ended QA
        30_60_s_academic_v0_1_cap_processed.json     # captioning
        academic_source/...                           # extracted .mp4 files

Each annotation record looks like:
    {
      "id": "Q04US",
      "conversations": [{"from": "human/gpt", "value": ...}, ...],
      "data_source": "30_60_s_academic_v0_1",
      "video": "academic_source/Charades/Q04US.mp4"   # relative to bucket dir
    }

This script:
  1. Walks all bucket dirs under --root
  2. Loads every *_qa_processed.json and *_cap_processed.json
  3. Resolves video paths to be relative to --root (so train_xsa.py's
     video_root can point at --root)
  4. Filters out records whose video file doesn't exist on disk
  5. Optionally samples down
  6. Writes the unified JSON

Usage:
    python scripts/build_train_subset.py \
        --root /data/train/LLaVA-Video-178K \
        --output /data/train/subset_5k.json \
        --max-samples 5000 \
        --kinds qa cap
"""

import argparse
import json
import os
import random
from pathlib import Path


def main(args):
    random.seed(args.seed)
    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"root not found: {root}")

    # Find all bucket directories. A bucket is any subdir of root that
    # contains *_processed.json files.
    bucket_dirs = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and any(d.glob("*_processed.json")):
            bucket_dirs.append(d)
    print(f"[scan] found {len(bucket_dirs)} bucket directories")

    all_records = []
    for bucket in bucket_dirs:
        bucket_name = bucket.name
        for json_path in sorted(bucket.glob("*_processed.json")):
            kind = "qa" if "qa_processed" in json_path.name else "cap"
            if args.kinds and kind not in args.kinds:
                continue

            try:
                with open(json_path) as f:
                    records = json.load(f)
            except Exception as e:
                print(f"  skip {json_path.name}: {e}")
                continue

            kept = 0
            for r in records:
                video = r.get("video")
                conv = r.get("conversations")
                if not video or not conv:
                    continue
                # Make video path relative to --root
                rel_path = f"{bucket_name}/{video}"
                all_records.append({
                    "id": r.get("id", str(len(all_records))),
                    "video": rel_path,
                    "conversations": conv,
                    "data_source": r.get("data_source", bucket_name),
                    "_kind": kind,
                })
                kept += 1
            print(f"  + {bucket_name}/{json_path.name}: {kept} records")

    print(f"[total] {len(all_records)} raw records")

    # Verify video files exist
    if not args.skip_existence_check:
        before = len(all_records)
        all_records = [
            r for r in all_records
            if (root / r["video"]).is_file()
        ]
        dropped = before - len(all_records)
        print(f"[filter] dropped {dropped} with missing video files")
        print(f"[filter] {len(all_records)} records with valid videos")

    # Subsample
    if args.max_samples and len(all_records) > args.max_samples:
        random.shuffle(all_records)
        all_records = all_records[: args.max_samples]
        print(f"[sample] randomly subsampled to {len(all_records)}")

    # Stats
    sources = {}
    kinds = {}
    for r in all_records:
        sources[r["data_source"]] = sources.get(r["data_source"], 0) + 1
        kinds[r["_kind"]] = kinds.get(r["_kind"], 0) + 1
    print("[stats] by data_source:")
    for s, c in sorted(sources.items()):
        print(f"  {s:>30s}: {c}")
    print("[stats] by kind:")
    for k, c in sorted(kinds.items()):
        print(f"  {k:>30s}: {c}")

    # Strip private debug field
    for r in all_records:
        r.pop("_kind", None)

    # Write
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(all_records, f, indent=2)
    size_mb = output.stat().st_size / 1e6
    print(f"[done] wrote {len(all_records)} records to {output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--kinds",
        nargs="*",
        default=["qa"],
        choices=["qa", "cap"],
        help="Which annotation kinds to include (qa = QA, cap = captioning)",
    )
    parser.add_argument(
        "--skip-existence-check",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)
