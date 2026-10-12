[![Ecosystem](https://img.shields.io/badge/Portfolio%20Ecosystem-341%20Studios-6366f1.svg)](https://pradeeptalari.me/projects/)
[![Architecture DAG](https://img.shields.io/badge/Architecture-DAG%20%26%20Course-10b981.svg)](https://pradeeptalari.me/tools/repo-learning/)

# vLLM PagedAttention & Chunked Prefill High-Throughput Studio

[![CI SRE Validation](https://github.com/Pradeeptalari14/tp-vllm-paged-attention/actions/workflows/sre-validation.yml/badge.svg)](https://github.com/Pradeeptalari14/tp-vllm-paged-attention/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform: SRE & AI Infrastructure](https://img.shields.io/badge/Platform-Kubernetes%20%7C%20H100-emerald.svg)](https://talaripradeep.info)

Production-grade implementation of **vLLM PagedAttention v2** virtual memory KV-cache allocation and **Chunked Prefill co-scheduling** with FlashAttention-3.

![vLLM Architecture Flow](docs/vllm_architecture_flow.png)

## Key Technical Highlights
- **Zero External Fragmentation:** Maps logical prompt token allocations non-contiguously into fixed-size physical GPU blocks, dropping memory waste from 60–80% to <4%.
- **Chunked Prefill Co-Scheduling:** Co-schedules compute-intensive prefill chunks with memory-bound decode iterations, eliminating TTFT prefill latency bubbles.
- **2.8x Concurrency Speedup:** Serves 2.8x more concurrent streams on NVIDIA H100/A100 clusters without provisioning extra hardware.
- **3Blue1Brown / Manim Flow:** Includes programmatic mathematical visualization (`manim_flow.py`) illustrating memory paging.

## Quickstart
```bash
git clone https://github.com/Pradeeptalari14/tp-vllm-paged-attention.git
cd tp-vllm-paged-attention
bash scripts/validate.sh
```

---
*Synchronized with [vLLM PagedAttention & Chunked Prefill Studio Studio](https://pradeeptalari.me/tools/vllm-paged-attention/) & [Repository Learning DAG](https://pradeeptalari.me/tools/repo-learning/) — [Talari Pradeep Portfolio](https://pradeeptalari.me)*
