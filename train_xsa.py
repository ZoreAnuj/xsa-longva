"""
Fine-tune LongVA-7B-DPO with XSA-patched CLIP vision tower.

Strategy:
  - Vision tower: full fine-tune (LR 2e-6) — needs to learn XSA-compatible weights
  - LLM: LoRA adapters (rank 16, LR 1e-5) — adapt to new vision features
  - Projector: full fine-tune (LR 1e-5) — small, needs to bridge old↔new features

Trains on a video instruction dataset (LLaVA-Video-178K-style). The dataset
is expected to be a JSON file of records like:
    {
      "id": "...",
      "video": "relative/path/to/video.mp4",
      "conversations": [
        {"from": "human", "value": "<image>\\nWhat is happening?"},
        {"from": "gpt",   "value": "..."}
      ]
    }

Usage:
    python train_xsa.py \
        --model-path  /workspace/xsa-longva/checkpoints/LongVA-7B-DPO \
        --data-path   /workspace/data/train/llava_video_30_60s_subset.json \
        --video-root  /workspace/data/train/LLaVA-Video-178K \
        --output-dir  /workspace/checkpoints/xsa-longva-run1 \
        --num-frames 32 \
        --batch-size 1 \
        --grad-accum 16 \
        --num-epochs 1 \
        --lr 1e-5 \
        --vision-lr 2e-6 \
        --lora-r 16 \
        --lora-alpha 32

Single H100 80GB target: ~12 hours for 1 epoch over ~50K samples at 32 frames.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Make repo root + LongVA importable
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from longva_helpers import load_longva  # noqa: E402
from patch_longva import (  # noqa: E402
    patch_clip_model_with_xsa,
    count_xsa_layers,
)
from xsa_clip_attention import XSACLIPAttention  # noqa: E402


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

@dataclass
class TrainArgs:
    model_path: str
    data_path: str
    video_root: str
    output_dir: str
    num_frames: int = 32
    batch_size: int = 1
    grad_accum: int = 16
    num_epochs: int = 1
    max_steps: int = -1
    lr: float = 1e-5
    vision_lr: float = 2e-6
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    save_steps: int = 2000
    log_steps: int = 10
    seed: int = 42
    max_seq_len: int = 8192
    image_size: int = 336
    train_xsa_only: bool = False  # ablation: freeze vision tower except XSA params


class VideoInstructionDataset(Dataset):
    """Loads pre-built video instruction JSON. Returns raw fields; the
    collator handles tokenization + frame sampling."""

    def __init__(self, json_path: str, video_root: str):
        with open(json_path) as f:
            self.records = json.load(f)
        self.video_root = video_root
        # Filter out records without a resolvable video path
        before = len(self.records)
        self.records = [
            r for r in self.records
            if r.get("video") and os.path.isfile(os.path.join(video_root, r["video"]))
        ]
        if len(self.records) < before:
            print(f"[dataset] dropped {before - len(self.records)} records "
                  f"with missing video files")
        print(f"[dataset] {len(self.records)} samples")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        r = self.records[idx]
        return {
            "id": r.get("id", str(idx)),
            "video_path": os.path.join(self.video_root, r["video"]),
            "conversations": r["conversations"],
        }


def sample_video_frames(video_path: str, num_frames: int) -> list:
    """Uniformly sample N frames from a video file as numpy uint8 arrays."""
    import numpy as np
    from decord import VideoReader, cpu

    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total = len(vr)
    if total == 0:
        raise ValueError(f"empty video: {video_path}")
    indices = np.linspace(0, total - 1, num=num_frames).astype(np.int64)
    frames = vr.get_batch(indices).asnumpy()  # (N, H, W, 3) uint8
    return [frames[i] for i in range(frames.shape[0])]


# ----------------------------------------------------------------------------
# Collator: builds the multimodal batch the LongVA forward expects
# ----------------------------------------------------------------------------

class VideoCollator:
    def __init__(self, tokenizer, image_processor, num_frames, max_seq_len):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.num_frames = num_frames
        self.max_seq_len = max_seq_len

        # Lazy import LongVA constants
        from longva.constants import IMAGE_TOKEN_INDEX
        from longva.mm_utils import tokenizer_image_token
        self.image_token_index = IMAGE_TOKEN_INDEX
        self._tokenize_with_image = tokenizer_image_token

    def _build_prompt(self, conversations) -> tuple[str, str]:
        """Convert LLaVA-style conversations to (prompt, target_response).
        We supervise only the assistant turn(s) — system+user is the prompt."""
        # Find the human and gpt turns. We support 1-turn conversations.
        human_text = ""
        gpt_text = ""
        for turn in conversations:
            role = turn.get("from", "")
            value = turn.get("value", "")
            if role == "human":
                human_text = value
            elif role in ("gpt", "assistant"):
                gpt_text = value
        # Ensure exactly one <image> token in the human turn
        if "<image>" not in human_text:
            human_text = "<image>\n" + human_text

        prompt = (
            f"<|im_start|>user\n{human_text}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        target = f"{gpt_text}<|im_end|>"
        return prompt, target

    def __call__(self, batch):
        items = []
        for record in batch:
            try:
                frames = sample_video_frames(record["video_path"], self.num_frames)
            except Exception as e:
                print(f"[collate] skip {record['id']}: {e}")
                continue

            try:
                pixel_values = self.image_processor.preprocess(
                    frames, return_tensors="pt"
                )["pixel_values"]  # (num_frames, 3, H, W)
            except Exception as e:
                print(f"[collate] preprocess failed {record['id']}: {e}")
                continue

            prompt, target = self._build_prompt(record["conversations"])

            # Tokenize prompt (with <image> placeholder) and target separately
            prompt_ids = self._tokenize_with_image(
                prompt, self.tokenizer, self.image_token_index, return_tensors="pt"
            )  # 1D LongTensor
            target_ids = self.tokenizer(
                target, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]

            input_ids = torch.cat([prompt_ids, target_ids], dim=0)
            # Labels: -100 on prompt tokens, target ids on the response
            labels = torch.full_like(input_ids, fill_value=-100)
            labels[prompt_ids.shape[0]:] = target_ids

            if input_ids.shape[0] > self.max_seq_len:
                input_ids = input_ids[: self.max_seq_len]
                labels = labels[: self.max_seq_len]

            items.append({
                "input_ids": input_ids,
                "labels": labels,
                "pixel_values": pixel_values,
            })

        if not items:
            return None

        # Pad input_ids and labels to the max length in batch
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        max_len = max(it["input_ids"].shape[0] for it in items)

        input_ids_b = torch.full((len(items), max_len), pad_id, dtype=torch.long)
        labels_b = torch.full((len(items), max_len), -100, dtype=torch.long)
        attn_mask = torch.zeros((len(items), max_len), dtype=torch.long)
        pixel_list = []

        for i, it in enumerate(items):
            n = it["input_ids"].shape[0]
            input_ids_b[i, :n] = it["input_ids"]
            labels_b[i, :n] = it["labels"]
            attn_mask[i, :n] = 1
            pixel_list.append(it["pixel_values"])

        return {
            "input_ids": input_ids_b,
            "attention_mask": attn_mask,
            "labels": labels_b,
            "images": pixel_list,  # list of (num_frames, 3, H, W) tensors
        }


# ----------------------------------------------------------------------------
# Optimizer setup with two parameter groups
# ----------------------------------------------------------------------------

def build_optimizer(
    model,
    lr: float,
    vision_lr: float,
    weight_decay: float,
    train_xsa_only: bool,
):
    """
    Two parameter groups:
      1. Vision tower (XSACLIPAttention modules + LayerNorms): vision_lr
      2. Everything else trainable (LoRA adapters, projector): lr

    If train_xsa_only=True, only the XSACLIPAttention parameters in the
    vision tower are trained — useful for an ablation.
    """
    vision_params = []
    other_params = []

    vt = model.get_vision_tower().vision_tower

    if train_xsa_only:
        # Freeze everything in vision tower except XSACLIPAttention parameters
        for p in vt.parameters():
            p.requires_grad = False
        for m in vt.modules():
            if isinstance(m, XSACLIPAttention):
                for p in m.parameters():
                    p.requires_grad = True

    for name, p in vt.named_parameters():
        if p.requires_grad:
            vision_params.append(p)

    vt_param_ids = {id(p) for p in vision_params}
    for name, p in model.named_parameters():
        if p.requires_grad and id(p) not in vt_param_ids:
            other_params.append(p)

    print(f"[optim] vision params: {sum(p.numel() for p in vision_params) / 1e6:.1f}M "
          f"({len(vision_params)} tensors)")
    print(f"[optim] other  params: {sum(p.numel() for p in other_params) / 1e6:.1f}M "
          f"({len(other_params)} tensors)")

    return torch.optim.AdamW(
        [
            {"params": vision_params, "lr": vision_lr, "weight_decay": weight_decay},
            {"params": other_params, "lr": lr, "weight_decay": weight_decay},
        ],
        betas=(0.9, 0.999),
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(args: TrainArgs):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ----------------------------------------------------------------
    print("[load] LongVA + XSA")
    # ----------------------------------------------------------------
    tokenizer, model, image_processor, _ = load_longva(
        args.model_path,
        device_map="cuda:0",
        attn_implementation="sdpa",
    )

    # Convert model from fp16 (LongVA's builder hardcodes it) to bf16.
    # bf16 has the same exponent range as fp32, which prevents the
    # XSA projection from overflowing in long sequences. fp16 NaNs out
    # when v.norm() is small relative to (y*v).sum().
    print("[load] converting model fp16 -> bf16")
    model = model.to(dtype=torch.bfloat16)

    vt = model.get_vision_tower().vision_tower
    patch_clip_model_with_xsa(vt, use_xsa=True)
    print(f"[load] XSA layers: {count_xsa_layers(vt)}")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ----------------------------------------------------------------
    print("[load] LoRA adapters on the LLM")
    # ----------------------------------------------------------------
    # Use inject_adapter_in_model so LoRA layers replace the originals
    # in-place without wrapping in a PeftModel. PeftModel wrapping breaks
    # LlavaQwenForCausalLM's forward signature (it would route labels into
    # the inner Qwen2Model which doesn't accept them).
    from peft import LoraConfig, inject_adapter_in_model

    # Freeze base weights first
    for p in model.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    # Inject adapters in-place — modifies model.model layers, doesn't wrap.
    # We pass model.model so target_modules only match Qwen2 layers, not the
    # vision tower's CLIPAttention (which also has q_proj/k_proj/v_proj/out_proj).
    inject_adapter_in_model(lora_config, model.model)

    # Re-enable training on vision tower (was frozen above)
    for p in vt.parameters():
        p.requires_grad = True

    # Re-enable training on the projector (sits between vision and LLM)
    for n, p in model.named_parameters():
        if "mm_projector" in n or "projector" in n:
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[load] trainable: {trainable/1e6:.1f}M / {total/1e9:.2f}B "
          f"({100*trainable/total:.2f}%)")

    # ----------------------------------------------------------------
    print("[data] building dataset and dataloader")
    # ----------------------------------------------------------------
    dataset = VideoInstructionDataset(args.data_path, args.video_root)
    collator = VideoCollator(
        tokenizer, image_processor,
        num_frames=args.num_frames,
        max_seq_len=args.max_seq_len,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collator,
        drop_last=True,
    )

    # ----------------------------------------------------------------
    print("[optim] building optimizer + scheduler")
    # ----------------------------------------------------------------
    optimizer = build_optimizer(
        model, args.lr, args.vision_lr, args.weight_decay,
        train_xsa_only=args.train_xsa_only,
    )

    steps_per_epoch = max(1, len(loader) // args.grad_accum)
    total_steps = (
        args.max_steps
        if args.max_steps > 0
        else steps_per_epoch * args.num_epochs
    )
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step: int):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ----------------------------------------------------------------
    print(f"[train] {total_steps} optimizer steps "
          f"(epoch_size={steps_per_epoch}, warmup={warmup_steps})")
    # ----------------------------------------------------------------
    model.train()
    log_file = output_dir / "training_log.jsonl"
    start = time.time()

    global_step = 0
    micro_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.num_epochs):
        for batch in loader:
            if batch is None:
                continue

            input_ids = batch["input_ids"].to("cuda:0")
            attention_mask = batch["attention_mask"].to("cuda:0")
            labels = batch["labels"].to("cuda:0")
            images = [im.to("cuda:0", dtype=torch.bfloat16) for im in batch["images"]]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    images=images,
                    modalities=["video"] * len(images),
                )
                loss = out.loss / args.grad_accum

            loss.backward()
            running_loss += loss.item()
            micro_step += 1

            if micro_step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_steps == 0:
                    elapsed = time.time() - start
                    log = {
                        "step": global_step,
                        "epoch": epoch,
                        "loss": running_loss / args.log_steps,
                        "lr_other": scheduler.get_last_lr()[1],
                        "lr_vision": scheduler.get_last_lr()[0],
                        "elapsed_h": elapsed / 3600,
                    }
                    print(
                        f"[step {global_step:>6d}] loss={log['loss']:.4f} "
                        f"lr={log['lr_other']:.2e}/{log['lr_vision']:.2e} "
                        f"elapsed={log['elapsed_h']:.2f}h"
                    )
                    with open(log_file, "a") as f:
                        f.write(json.dumps(log) + "\n")
                    running_loss = 0.0

                if global_step % args.save_steps == 0 or global_step >= total_steps:
                    save_dir = output_dir / f"step_{global_step:07d}"
                    save_checkpoint(model, save_dir)

                if global_step >= total_steps:
                    break
        if global_step >= total_steps:
            break

    # Final save
    save_checkpoint(model, output_dir / "final")
    print(f"[done] total {global_step} steps in {(time.time()-start)/3600:.2f}h")


def save_checkpoint(model, save_dir: Path):
    """Save vision tower (XSA-tuned) and LoRA adapters separately."""
    save_dir.mkdir(parents=True, exist_ok=True)

    # Vision tower XSA weights — save just the vision_tower state_dict
    vt = model.get_vision_tower().vision_tower
    from safetensors.torch import save_file
    vt_sd = {k: v.detach().cpu() for k, v in vt.state_dict().items()}
    save_file(vt_sd, str(save_dir / "vision_tower_xsa.safetensors"))

    # LoRA adapters via PEFT
    if hasattr(model.model, "save_pretrained"):
        model.model.save_pretrained(str(save_dir / "lora"))

    # Projector
    proj_sd = {}
    for name, p in model.named_parameters():
        if "mm_projector" in name or "projector" in name:
            proj_sd[name] = p.detach().cpu()
    if proj_sd:
        save_file(proj_sd, str(save_dir / "projector.safetensors"))

    print(f"[save] checkpoint at {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--vision-lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--save-steps", type=int, default=2000)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--train-xsa-only", action="store_true",
                        help="Freeze vision tower except XSA params (ablation)")
    parsed = parser.parse_args()
    args = TrainArgs(**vars(parsed))
    main(args)
