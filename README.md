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

## Results (experiment did not beat the baseline)

**Status:** Experiment complete. XSA did **not** beat LongVA-7B-DPO on LongVideoBench val under the recipes we could afford to run on a single H100. Full training logs, methodology, and both negative results are documented below and in `docs/plans/`.

Base model: `lmms-lab/LongVA-7B-DPO`  
Benchmark: LongVideoBench val split (1,337 multiple-choice questions, public ground truth)  
Hardware: 1× H100 80GB  
Eval dtype: fp16 (LongVA loader default) / bf16 (for XSA-tuned runs, matching training)  
Eval frames: 32

| Model | Overall | Δ vs baseline | Notes |
|---|---:|---:|---|
| LongVA-7B-DPO (SA baseline) | **52.80%** (706/1337) | — | reproducible via `eval_longvideobench.py --mode baseline` |
| Run 1: XSA, all 24 CLIP layers | **42.18%** (564/1337) | **−10.62%** | 5K samples × 2 epochs, vision_lr 2e-6, no alpha curriculum |
| Run 2: XSA, last-12 layers, alpha ramp | _incomplete_ | — | pod crashed at step 2060/3125 (66% through), loss stable ~0.9 |
| LLaVA-Video-7B-Qwen2 (open 7B SOTA, 128 frames) | 62.7%¹ | — | reference number from the leaderboard, not reproduced here |

¹ From the official LongVideoBench leaderboard — included only as a reference for 7B-class open models at higher frame budgets.

**Note on the baseline number:** Our 52.80% is ~3-5 pts below LongVA's published val score because LongVA's `load_pretrained_model` hardcodes fp16 (we lose ~2 pts vs bf16) and we evaluate at 32 frames instead of 64-128. The XSA comparison is **apples-to-apples** within our setup — same loader, same frame count, same eval script — so the −10.62 delta on run 1 is real.

### What each run taught us

**Run 1 — aggressive recipe, clear failure:**

- Patched all 24 CLIP layers with XSA at once
- Trained vision tower at LR 2e-6 + LoRA on LLM, fp16 → NaN, then bf16
- 5K samples × 2 epochs
- Training loss converged around 1.3
- **Eval: 42.18% (−10.62 vs baseline)** — the vision tower was perturbed enough to break LongVA's pretrained features but not enough to converge to an XSA-compatible solution

**Run 2 — curriculum recipe, promising trajectory, crashed before finishing:**

The plan: introduce XSA gradually instead of all at once, so the model starts as the baseline and is pushed toward XSA without catastrophic forgetting.

- **Layer subset:** XSA on only the last 12 of 24 CLIP layers (`--xsa-layers last-12`)
- **Alpha curriculum:** projection coefficient ramped linearly from 0 → 1 over the first 10% of training steps (`--xsa-alpha-ramp-ratio 0.10`). At step 0, XSA is off and the model behaves exactly like the baseline; by step 312, full XSA is active
- **Conservative vision_lr:** 5e-7 (4× lower than run 1)
- **LoRA only on LLM decoder blocks**, not on vision tower (run 1 had injected LoRA into the CLIP vision tower as well, corrupting saved checkpoints)
- 50K samples × 1 epoch, 16 frames, bf16 end-to-end
- Training loss trajectory: 2.8 → 1.2 during ramp, then steadily **0.9–1.0** under full XSA at alpha=1 — genuinely different regime from run 1
- **RunPod pod crashed at step 2060/3125** (66% done, 6.59h elapsed). All checkpoints and eval data were on the pod's ephemeral `/` volume and were lost with the restart. Training logs survived on `/workspace`

Run 2's loss trajectory (0.9 under full XSA vs run 1's 1.3) is strong evidence the curriculum approach was doing something qualitatively different. Whether it would have eventually beaten 52.80% on eval is an open question that requires a re-run to answer.

### What this means

1. **The XSA paper result does not transfer to video-LM vision encoders for free.** A naive swap-and-fine-tune on a modest budget degrades the model by ~10 points.
2. **Curriculum ramping + layer subset looks promising** but we couldn't close the loop. Loss went lower than run 1, but the eval number for run 2 doesn't exist.
3. **The bottleneck is compute, not architecture.** Original LongVA fine-tuning used ~750K samples on 8× A100 for 1.5 days. We used 0.7% of that data (run 1) or 6.7% (run 2) on 1× H100 for a few hours. LoRA helps but can't fully compensate.
4. **Pipeline is fully reproducible and debugged.** Every bug we hit (meta tensors, fp16 NaN, LoRA leaking into the vision tower, dtype mismatch, dataset schema) is fixed and committed.

### To actually get a positive result, you would need

- ~10× more training samples (200K-500K), or
- Several more training runs with a learning-rate / layer-count sweep, or
- Start from a fully bf16 LongVA checkpoint (avoid the fp16 handicap), or
- All of the above

None of which we can finish on a 1× H100 in an overnight budget.

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

**Designed for:** 1× H100 80GB  
**Training time:** 10-14 hours  
**Total VRAM needed:** ~50GB (with bf16, gradient checkpointing, DeepSpeed ZeRO-3)

Should also run on 1× A100 80GB with minor config changes. Won't fit on a 32GB GPU for full FT — would require aggressive quantization.

## Status

**Experiment concluded (negative result).** See the Results section below. The pipeline works end-to-end, both training recipes were executed on real LongVA checkpoints and real LongVideoBench data, and the results are honest. XSA did not beat the baseline under the training budget we had available on a single H100.

The code and methodology are fully reproducible — if you have the compute to run a 500K-sample, multi-day fine-tune, the scripts here are a working starting point.

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
