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


def split_inputs(inputs_list: list) -> tuple[list, str]:
    """LongVideoBench returns an interleaved list of PIL Images (sampled
    video frames) and str chunks (interleaved subtitles + question + options
    + final instruction). We split into (frames, prompt_text).

    The text chunks are concatenated in order with newlines between them.
    For each PIL image we insert a placeholder line so the model can see
    that frames are interleaved with the subtitles temporally — but for
    LongVA we just concatenate all the frames at the start (the LongVA
    chat template does not natively support per-frame text interleaving).
    """
    frames = []
    text_parts = []
    for x in inputs_list:
        if hasattr(x, "size"):  # PIL Image
            frames.append(x)
        elif isinstance(x, str):
            text_parts.append(x)
        # silently ignore other types
    full_text = "\n".join(text_parts)
    return frames, full_text


def parse_answer(text: str) -> str:
    """Extract A/B/C/D/E from the model's free-form output."""
    text = text.strip().upper()
    for letter in OPTION_LETTERS:
        if letter in text[:8]:
            return letter
    return ""  # could not parse


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

        # LongVideoBench item structure:
        #   item["inputs"]    -> interleaved list of PIL Images and str chunks
        #   item["correct_choice"] -> letter "A".."E"
        #   item["id"]        -> str
        frames, text_block = split_inputs(item["inputs"])
        gt_letter = item["correct_choice"].strip().upper()

        if not frames:
            errors += 1
            continue

        # Tokenize prompt with the LongVA conversation template, prepending
        # a single <image> placeholder (LongVA inserts all frames at this
        # position). The text_block already contains the question + options
        # + final "Answer with the option letter" instruction from the
        # LongVideoBench dataset.
        prompt = (
            f"<|im_start|>user\n<image>\n{text_block}"
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
            pred_letter = parse_answer(response)
        except Exception as e:
            print(f"[eval] generate failed for idx {idx}: {e}")
            traceback.print_exc()
            errors += 1
            continue

        if not pred_letter:
            parse_failures += 1
            is_correct = False
        else:
            total += 1
            is_correct = pred_letter == gt_letter
            if is_correct:
                correct += 1

        results.append({
            "id": item.get("id", str(idx)),
            "duration_group": item.get("duration_group", "unknown"),
            "gt": gt_letter,
            "pred": pred_letter,
            "response": response,
            "correct": is_correct,
        })

        if args.verbose or idx < 3:
            print(f"  [{idx}] gt={gt_letter} pred={pred_letter or '?'} "
                  f"resp={response[:60]!r}")

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
        if r["pred"]:  # non-empty letter means we parsed an answer
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
    parser.add_argument("--verbose", action="store_true", help="Print every response")
    args = parser.parse_args()
    evaluate(args)
