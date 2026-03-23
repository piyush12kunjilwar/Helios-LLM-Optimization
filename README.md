# Helios: Systematic LLM Inference Optimization

[![arXiv](https://img.shields.io/badge/arXiv-2026-red)](https://arxiv.org)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-orange)](https://pytorch.org)

> End-to-end empirical study of LLM inference optimization on NVIDIA L4.
> From naive FP32 to **190x throughput improvement** using vLLM PagedAttention.
> Evaluated on 317,486 real ShareGPT user prompts.

## Key Results

| Method | Latency | Speedup | Tok/s | Memory |
|--------|---------|---------|-------|--------|
| FP32 baseline | 5275ms | 1.00x | 18.2 | 12.35GB |
| FP16 | 4528ms | 1.17x | 21.4 | 6.18GB |
| INT8 batch=1 | 22288ms | 0.24x | 4.5 | 3.44GB |
| FP16+SDPA batch=16 | 338ms | 16.23x | 295.8 | 6.18GB |
| INT4 batch=16 | 904ms | 6.07x | 110.6 | 2.08GB |
| **vLLM continuous** | **28ms** | **190x** | **3472** | 5.80GB |

## Key Findings

**1. Serving strategy matters more than precision**
vLLM's PagedAttention + continuous batching achieves 190x speedup.
Static batching at FP16 achieves only 16x.
The serving architecture dominates all precision choices.

**2. INT8 at batch=1 is 4x SLOWER than FP32**
INT8 quantization degraded throughput by 4x at batch=1
due to dequantization overhead on L4 tensor cores.
This is a common production pitfall — measured and documented here.

**3. matmul/linear = 34% of CUDA time. Attention = 0.2%**
PyTorch profiling revealed the real bottleneck is linear layers,
not attention. This explains why Flash Attention alone gave minimal speedup.

**4. FP16 is essentially lossless**
WikiText-2 perplexity: FP32=16.26, FP16=16.26 (+0.02% degradation).
vLLM uses FP16 → 190x speedup at 0.02% quality cost.

**5. 99.5% cost reduction**
At 100M tokens/day on AWS p3.2xlarge:
- FP32 baseline: $4,662/day → $1.7M/year
- vLLM optimized: $24/day → $8,760/year
- Annual savings: $1,692,930

## Experimental Setup

| Component | Details |
|-----------|---------|
| Model | Qwen2.5-3B-Instruct (3.08B params) |
| Hardware | NVIDIA L4 GPU (23.7GB VRAM) |
| Dataset | ShareGPT 317,486 real user prompts |
| Accuracy | WikiText-2 perplexity benchmark |
| Serving | vLLM 0.18.0 + PagedAttention |
| Framework | PyTorch 2.10 + Transformers 4.57 |

## Results Files

| File | Description |
|------|-------------|
| `baseline_fp32.json` | FP32 ground truth (500 prompts) |
| `results_fp16.json` | FP16 half-precision |
| `results_int8.json` | INT8 bitsandbytes quantization |
| `results_sdpa.json` | PyTorch SDPA attention |
| `results_batched.json` | Batch size 1→16 scaling |
| `helios_final_results.json` | INT4 NF4 quantization |
| `results_vllm.json` | vLLM PagedAttention benchmark |
| `results_accuracy.json` | WikiText-2 perplexity tradeoff |
| `results_concurrency.json` | P50/P95/P99 under load |
| `results_profiling.json` | PyTorch profiler breakdown |
| `helios_complete_results.png` | Full visualization (9 plots) |

## Batch Scaling

Near-perfect linear throughput scaling observed:

| Batch Size | Tok/s | vs Baseline |
|-----------|-------|-------------|
| 1 | 19.4 | 1.07x |
| 2 | 38.2 | 2.09x |
| 4 | 75.2 | 4.13x |
| 8 | 150.1 | 8.23x |
| 16 | 295.8 | 16.23x |

## Concurrency Analysis

Static batching fails P99 SLA (5000ms) at ALL concurrency levels.
vLLM continuous batching: 28ms P99 at 320 concurrent requests.

| Users | P50 | P95 | P99 | SLA |
|-------|-----|-----|-----|-----|
| 1 | 5637ms | 5799ms | 6025ms | ❌ |
| 10 | 5725ms | 5832ms | 5832ms | ❌ |
| 50 | 5799ms | 6290ms | 22523ms | ❌ |
| vLLM 320 | ~28ms | ~28ms | ~28ms | ✅ |

## Tech Stack
```
PyTorch 2.10 · Transformers 4.57 · vLLM 0.18.0
bitsandbytes · CUDA 12.8 · NVIDIA L4 (23.7GB)
ShareGPT · WikiText-2 · Python 3.12
```

## Citation
```bibtex
@misc{kunjilwar2026helios,
  title={Helios: Systematic LLM Inference Optimization on 
         Consumer-Grade GPUs},
  author={Kunjilwar, Piyush},
  year={2026},
  note={Technical Report, Northeastern University}
}
```

## Author

**Piyush Kunjilwar**
MS Information Systems — Northeastern University (May 2026)
Applied ML Engineer | LLM Inference | GPU Optimization

[LinkedIn](https://linkedin.com/in/piyush-kunjilwar) ·
[GitHub](https://github.com/piyush12kunjilwar) ·
[Email](mailto:kunjilwar.p@northeastern.edu)
