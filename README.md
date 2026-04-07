# XSA-LongVA

### Exclusive Self Attention for Long Video Understanding
#### Targeting LongVideoBench SOTA with 2 lines of code.

> Applying [Exclusive Self Attention (XSA)](https://arxiv.org/abs/2603.09078) to [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA)'s CLIP vision encoder. Long video is XSA's theoretical sweet spot — long sequences + extreme temporal redundancy = maximum attention similarity bias to eliminate.

---

## What is this?

Video-LLMs waste their vision encoder on frame-to-frame self-similarity. Every patch in frame N is nearly identical to the same patch in frame N-1, and standard attention reinforces that redundancy instead of capturing what actually changes.

**XSA projects out the self-value component from attention output:**

```python
# The entire modification (2 lines inside CLIP attention):
coeff = (y * v).sum(-1, keepdim=True) / (v.norm(dim=-1, keepdim=True) ** 2 + 1e-6)
y = y - coeff * v
```

This forces each visual token's attention output to contain **only** information from other tokens — pure context — which is exactly what video understanding needs.

## Why LongVA + XSA?

| Property | Why it's a perfect fit |
|---|---|
| CLIP ViT-L/14 in LongVA has **24 layers** | 24 drop-in swap points |
| Visual tokens scale with frames × patches | Up to **100K+ tokens** at high frame counts — XSA's strongest published regime |
| Temporal redundancy between adjacent frames | Attention similarity bias is at its maximum |
| Qwen2-7B-224K LLM handles long context natively | No LLM changes needed |
| LongVA vision tower already fine-tuned with LR 2e-6 | Natural insertion point for XSA retraining |

## Architecture

```
Video (N frames, 336×336)
        │
        ▼
CLIP ViT-L/14-336  ──┐
  24 layers          │  ← SA replaced with XSA
  1024 hidden, 16 heads
        │
        ▼
2×2 avg pool → 144 tokens/frame
        │
        ▼
MLP projector (1024 → 3584)
        │
        ▼
Qwen2-7B-Instruct-224K  ← LoRA adapters (rank 16)
        │
        ▼
Answer
```

**Training:** Full fine-tune of XSA-patched vision tower (LR 2e-6) + LoRA on LLM (LR 1e-5) on ~100K video-instruction samples from LLaVA-Video-178K.

## Quick Start

```bash
# 1. Setup environment (clones LongVA, installs deps)
bash scripts/setup_env.sh
conda activate xsa-longva

# 2. Download base model and data
bash scripts/download_longva.sh ./checkpoints/LongVA-7B-DPO
bash scripts/download_eval.sh ./data/eval
bash scripts/download_data.sh ./data/train

# 3. Baseline eval (unmodified LongVA)
python eval_longvideobench.py \
    --model-path ./checkpoints/LongVA-7B-DPO \
    --data-path ./data/eval/LongVideoBench \
    --output ./results/baseline_sa.json

# 4. Train XSA-LongVA (overnight, 1x H100)
bash scripts/train_overnight.sh

# 5. Eval XSA model
python eval_longvideobench.py \
    --model-path ./checkpoints/xsa-longva-run1 \
    --data-path ./data/eval/LongVideoBench \
    --output ./results/xsa_trained.json \
    --use-xsa
```

## Target Results

Base model: `lmms-lab/LongVA-7B-DPO`  
Benchmark: LongVideoBench val split (1,337 questions with ground truth)

| Model | Overall | 8-15s | 15-60s | 3-10min | 15-60min |
|-------|:---:|:---:|:---:|:---:|:---:|
| LongVA-7B-DPO (SA baseline) | ~57% | - | - | - | - |
| **XSA-LongVA-7B (ours)** | **≥60%** | - | - | - | - |
| LLaVA-Video-7B-Qwen2 (current open SOTA) | 62.7% | - | - | - | - |

**Hypothesis:** XSA gains concentrate on the longer duration buckets (15-60min) where temporal redundancy is most extreme and sequence length is longest.

## Repository Structure

```
xsa-longva/
├── xsa_clip_attention.py       # XSA drop-in replacement for HF CLIPAttention
├── patch_longva.py             # Monkey-patch LongVA vision tower (all 24 layers)
├── train_xsa.py                # LoRA LLM + full vision tower fine-tuning
├── eval_longvideobench.py      # LongVideoBench val evaluation
├── eval_videomme.py            # Video-MME long split
├── eval_mvbench.py             # MVBench (short-video sanity check)
├── scripts/                    # Setup, download, train, eval
├── analysis/                   # Cosine similarity, attention viz, plots
├── tests/                      # Unit + integration tests
├── configs/                    # DeepSpeed, LoRA configs
└── docs/plans/                 # Detailed implementation plan
```

## How XSA Works

In standard self-attention, token `i`'s output is:

$$y_i = \sum_j a_{i,j} \cdot v_j$$

The [XSA paper](https://arxiv.org/abs/2603.09078) shows that $y_i$ is highly correlated with $v_i$ (the token's own value), wasting capacity on self-information already available via the residual connection.

**XSA** removes this redundancy:

$$z_i = y_i - \frac{y_i^\top v_i}{\|v_i\|^2} v_i$$

The paper's biggest empirical finding: **gains scale with sequence length**. Long video inputs (up to 100K tokens in LongVA) are the single largest sequence length in vision-language ML — this is where XSA should shine brightest.

## Hardware

**Designed for:** 1× H100 80GB (RunPod ~$3/h)  
**Training time:** 10-14 hours  
**Total cost:** ~$50-80 (end-to-end including eval)

Should also run on 1× A100 80GB with minor config changes. Won't fit on RTX 5090 for full FT due to 32GB VRAM — would require aggressive quantization.

## Status

Current phase: **Implementation** (code being written based on detailed plan in `docs/plans/2026-04-07-xsa-longva.md`)

## Citation

```bibtex
@article{zhai2026exclusive,
  title={Exclusive Self Attention},
  author={Zhai, Shuangfei},
  journal={arXiv preprint arXiv:2603.09078},
  year={2026}
}

@article{zhang2024longva,
  title={Long Context Transfer from Language to Vision},
  author={Zhang, Peiyuan and Zhang, Kaichen and Li, Bo and Zeng, Guangtao and Yang, Jingkang and Zhang, Yuanhan and Wang, Ziyue and Tan, Haoran and Li, Chunyuan and Liu, Ziwei},
  journal={arXiv preprint arXiv:2406.16852},
  year={2024}
}

@article{wu2024longvideobench,
  title={LongVideoBench: A Benchmark for Long-context Interleaved Video-Language Understanding},
  author={Wu, Haoning and Li, Dongxu and Chen, Bei and Li, Junnan},
  journal={arXiv preprint arXiv:2407.15754},
  year={2024}
}
```

## Acknowledgments

- [Exclusive Self Attention](https://arxiv.org/abs/2603.09078) by Shuangfei Zhai
- [LongVA](https://github.com/EvolvingLMMs-Lab/LongVA) by the LMMs-Lab team
- [LongVideoBench](https://github.com/longvideobench/LongVideoBench) by Wu et al.
- [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K) by Zhang et al.

## License

MIT — see [LICENSE](LICENSE)
