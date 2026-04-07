"""
Evaluate a LongVA model on LongVideoBench validation split (1,337 questions
with public ground truth — no leaderboard submission required).

Supports three modes via the --mode flag:
    baseline   - unmodified LongVA (vanilla CLIPAttention)
    xsa-zero   - LongVA with vision tower swapped to XSA (untrained)
    xsa-tuned  - LongVA with XSA fine-tuned vision tower (loads --xsa-ckpt)

Usage:
    # Baseline
    python eval_longvideobench.py \
        --model-path /workspace/xsa-longva/checkpoints/LongVA-7B-DPO \
        --data-path  /workspace/data/eval/LongVideoBench \
        --output     /workspace/results/lvb_baseline.json \
        --max-frames 64 \
        --mode baseline

    # Zero-shot XSA (expected to degrade — confirms training is needed)
    python eval_longvideobench.py \
        --model-path /workspace/xsa-longva/checkpoints/LongVA-7B-DPO \
        --data-path  /workspace/data/eval/LongVideoBench \
        --output     /workspace/results/lvb_xsa_zero.json \
        --max-frames 64 \
        --mode xsa-zero
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from tqdm import tqdm

# Make repo root importable so we can use longva_helpers + patch_longva
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from longva_helpers import load_longva  # noqa: E402
from patch_longva import patch_clip_model_with_xsa, count_xsa_layers  # noqa: E402


OPTION_LETTERS = ["A", "B", "C", "D", "E"]


def build_prompt(question: str, options: list, subtitles: str = "") -> str:
    """Format a LongVideoBench question as a multiple-choice prompt."""
    opts = "\n".join(f"{OPTION_LETTERS[i]}. {o}" for i, o in enumerate(options))
    sub_block = f"Subtitles:\n{subtitles}\n\n" if subtitles else ""
    return (
        f"{sub_block}{question}\n{opts}\n"
        f"Answer with the option letter only."
    )


def parse_answer(text: str, num_options: int) -> int:
    """Extract the chosen option index from the model's free-form output."""
    text = text.strip().upper()
    # Look for the first occurrence of A/B/C/D/E within the first ~10 chars
    for i, letter in enumerate(OPTION_LETTERS[:num_options]):
        if letter in text[:8]:
            return i
    return -1  # could not parse


@torch.no_grad()
def evaluate(args):
    # Lazy import so the script can be parsed without LongVA installed
    from longva.constants import IMAGE_TOKEN_INDEX
    from longva.mm_utils import tokenizer_image_token
    from longvideobench import LongVideoBenchDataset

    # ----------------------------------------------------------------
    # Load model
    # ----------------------------------------------------------------
    print(f"[load] mode={args.mode} ckpt={args.model_path}")
    tokenizer, model, image_processor, _ = load_longva(
        args.model_path,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )

    if args.mode in ("xsa-zero", "xsa-tuned"):
        vt = model.get_vision_tower().vision_tower
        patch_clip_model_with_xsa(vt, use_xsa=True)
        n_xsa = count_xsa_layers(vt)
        print(f"[load] patched vision tower with XSA ({n_xsa} layers)")

    if args.mode == "xsa-tuned":
        if not args.xsa_ckpt or not os.path.isdir(args.xsa_ckpt):
            print(f"[load] ERROR: --xsa-ckpt required for mode=xsa-tuned")
            sys.exit(1)
        # Load fine-tuned vision tower weights
        from safetensors.torch import load_file
        st_path = os.path.join(args.xsa_ckpt, "vision_tower_xsa.safetensors")
        if not os.path.isfile(st_path):
            print(f"[load] ERROR: vision_tower_xsa.safetensors not in {args.xsa_ckpt}")
            sys.exit(1)
        sd = load_file(st_path, device="cuda:0")
        missing, unexpected = vt.load_state_dict(sd, strict=False, assign=True)
        print(f"[load] fine-tuned vision tower loaded "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    model.eval()
    model_dtype = next(model.parameters()).dtype
    print(f"[load] model dtype: {model_dtype}")

    # ----------------------------------------------------------------
    # Load LongVideoBench
    # ----------------------------------------------------------------
    print(f"[data] loading from {args.data_path}")
    dataset = LongVideoBenchDataset(
        args.data_path,
        "lvb_val.json",
        max_num_frames=args.max_frames,
    )
    print(f"[data] {len(dataset)} questions")

    if args.limit:
        print(f"[data] limiting to first {args.limit} questions")

    # ----------------------------------------------------------------
    # Evaluate
    # ----------------------------------------------------------------
    results = []
    correct = 0
    total = 0
    parse_failures = 0
    errors = 0
    start_time = time.time()

    pbar = tqdm(range(len(dataset)), desc=f"Eval ({args.mode})")
    for idx in pbar:
        if args.limit and idx >= args.limit:
            break
        try:
            item = dataset[idx]
        except Exception as e:
            print(f"[eval] dataset[{idx}] failed: {e}")
            errors += 1
            continue

        # LongVideoBench item structure (from longvideobench package):
        #   item["inputs"]    -> list of PIL Images (sampled video frames)
        #   item["question"]  -> str
        #   item["candidates"]-> list[str] options
        #   item["correct_choice"] -> int index
        #   item["subtitles"] -> optional str
        #   item["duration_group"] -> bucket name
        frames = item.get("inputs") or item.get("frames")
        question = item["question"]
        options = item["candidates"]
        gt_idx = item["correct_choice"]
        subtitles = item.get("subtitles", "") or ""

        full_question = build_prompt(question, options, subtitles)

        # Tokenize prompt with the LongVA conversation template
        prompt = (
            f"<|im_start|>user\n<image>\n{full_question}"
            f"<|im_end|>\n<|im_start|>assistant\n"
        )
        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to("cuda:0")

        # Preprocess frames
        try:
            images = image_processor.preprocess(
                frames, return_tensors="pt"
            )["pixel_values"]
            images = images.to("cuda:0", dtype=model_dtype)
        except Exception as e:
            print(f"[eval] frame preprocess failed for idx {idx}: {e}")
            errors += 1
            continue

        # Generate
        try:
            output_ids = model.generate(
                input_ids,
                images=[images],
                modalities=["video"],
                do_sample=False,
                max_new_tokens=8,
                temperature=0.0,
            )
            response = tokenizer.decode(
                output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
            )
            pred_idx = parse_answer(response, len(options))
        except Exception as e:
            print(f"[eval] generate failed for idx {idx}: {e}")
            traceback.print_exc()
            errors += 1
            continue

        if pred_idx < 0:
            parse_failures += 1
            is_correct = False
        else:
            total += 1
            is_correct = pred_idx == gt_idx
            if is_correct:
                correct += 1

        results.append({
            "id": item.get("id", idx),
            "duration_group": item.get("duration_group", "unknown"),
            "gt": gt_idx,
            "pred": pred_idx,
            "response": response,
            "correct": is_correct,
        })

        pbar.set_postfix({
            "acc": f"{(correct / max(total, 1)) * 100:.1f}%",
            "n": total,
            "errs": errors + parse_failures,
        })

    elapsed = time.time() - start_time
    overall_acc = correct / max(total, 1) * 100
    print(f"\n[done] elapsed: {elapsed/60:.1f} min")
    print(f"[done] {correct}/{total} = {overall_acc:.2f}%")
    print(f"[done] parse failures: {parse_failures}, errors: {errors}")

    # Per-duration-group breakdown
    groups: dict[str, list[int]] = {}
    for r in results:
        g = r.get("duration_group", "unknown")
        groups.setdefault(g, [0, 0])
        if r["pred"] >= 0:
            groups[g][1] += 1
            if r["correct"]:
                groups[g][0] += 1
    print("\nPer-duration-group:")
    for g in sorted(groups.keys()):
        c, t = groups[g]
        print(f"  {g:>20s}: {c:>4d}/{t:>4d} = {c/max(t,1)*100:6.2f}%")

    # ----------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------
    output = {
        "args": vars(args),
        "model_path": args.model_path,
        "mode": args.mode,
        "max_frames": args.max_frames,
        "elapsed_seconds": elapsed,
        "overall_accuracy": overall_acc,
        "correct": correct,
        "total": total,
        "parse_failures": parse_failures,
        "errors": errors,
        "per_group": {
            g: {"correct": c, "total": t, "acc": c / max(t, 1) * 100}
            for g, (c, t) in groups.items()
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[done] saved to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="LongVA checkpoint dir")
    parser.add_argument("--data-path", required=True, help="LongVideoBench dataset dir")
    parser.add_argument("--output", required=True, help="Path to save results JSON")
    parser.add_argument(
        "--mode",
        choices=["baseline", "xsa-zero", "xsa-tuned"],
        default="baseline",
        help="baseline=vanilla, xsa-zero=patched untrained, xsa-tuned=patched+ckpt",
    )
    parser.add_argument("--xsa-ckpt", default=None, help="XSA fine-tuned vision tower dir")
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="Eval first N (0 = all)")
    args = parser.parse_args()
    evaluate(args)
