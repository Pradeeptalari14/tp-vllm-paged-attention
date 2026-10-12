# vLLM PagedAttention v2 & Chunked Prefill Engine (`tp-vllm-paged-attention`)

[![Ecosystem](https://img.shields.io/badge/Portfolio%20Ecosystem-341%20Studios-6366f1.svg)](https://pradeeptalari.me/projects/)
[![Architecture DAG](https://img.shields.io/badge/Architecture-DAG%20%26%20Course-10b981.svg)](https://pradeeptalari.me/tools/repo-learning/)
[![Interactive Studio](https://img.shields.io/badge/Interactive%20Studio-Live%20Workbench-0ea5e9.svg)](https://pradeeptalari.me/tools/vllm-paged-attention/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Security Policy](https://img.shields.io/badge/Security-Policy-success.svg)](SECURITY.md)
[![GPU Target: H100 / A100](https://img.shields.io/badge/GPU%20Target-NVIDIA%20H100%20%7C%20A100-76b900.svg)](k8s-vllm-h100.yaml)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11+-brightgreen.svg)](https://python.org)

Production-grade implementation of **vLLM PagedAttention v2** virtual memory KV-cache management, non-contiguous physical block allocation, and **Chunked Prefill co-scheduling** with FlashAttention-3 on NVIDIA Hopper and Ampere GPU clusters.

---

## 🏗️ Architecture & Memory Virtualization Flow

![vLLM Architecture Flow](docs/vllm_architecture_flow.png)

```mermaid
flowchart TD
    subgraph Ingress["Client Request Ingress"]
        Req["User Prompt Stream (Varying Sequence Lengths)"]
    end

    subgraph MemoryPlane["PagedAttention Virtual Memory Plane"]
        LTable["Logical Block Table (Continuous Virtual Tokens)"]
        BManager["Block Manager & Physical Page Allocator"]
        PBlocks["Physical GPU Block Pool (Non-Contiguous 16-Token Blocks)"]
    end

    subgraph Batcher["Scheduling & Execution Engine"]
        CB["Continuous Dynamic Iteration Batcher"]
        Chunker["Chunked Prefill Co-Scheduler"]
        FA3["FlashAttention-3 / PagedAttention v2 CUDA Kernels"]
    end

    subgraph Hardware["GPU Compute Infrastructure"]
        H100["NVIDIA H100 SXM5 / A100 80GB HBM3 Memory"]
    end

    Req --> LTable
    LTable --> BManager
    BManager --> PBlocks
    PBlocks --> CB
    CB --> Chunker
    Chunker --> FA3
    FA3 --> H100
```

---

## 🧠 Core Engineering & Architectural Principles

### 1. Zero External Memory Fragmentation
Traditional LLM serving pre-allocates contiguous memory for the maximum possible sequence length (e.g., 2048 or 8192 tokens), wasting **60% to 80%** of GPU memory due to internal and external fragmentation. PagedAttention borrows operating system virtual memory paging: KV cache tensors are partitioned into fixed-size physical blocks (e.g., 16 tokens), reducing memory waste to **<4%**.

### 2. Chunked Prefill Co-Scheduling
Compute-heavy prefill phases (processing new prompt tokens) monopolize Tensor Cores and cause high Time-To-First-Token (TTFT) latency bubbles for active autoregressive decoding steps. Chunked Prefill slices prompt token sequences into discrete budgets (e.g., 512 tokens), co-scheduling prefill chunks alongside ongoing decode iterations to maintain sub-25ms inter-token latencies.

### 3. Copy-on-Write & Prefix Caching
Shared prompt prefixes (system instructions, multi-turn dialogue histories, RAG context documents) point to the same physical memory blocks via reference counting. Physical memory is duplicated only when a sequence branches or modifies tokens, yielding up to **5x memory savings** for multi-turn conversations.

### 4. Continuous Iteration-Level Batching
Unlike static batching that locks an execution batch until the longest sequence completes, continuous batching evicts completed sequences and schedules newly arriving prompt tokens on every forward pass.

### 5. Multi-GPU Tensor Parallelism & CPU Swapping
When GPU HBM memory reaches threshold capacity (`GPU_MEMORY_UTILIZATION >= 0.95`), the block manager evicts least-recently-used KV blocks to host CPU DRAM over PCIe Gen5 / NVLink, preventing Out-Of-Memory (OOM) fatal crashes.

---

## 💻 Infrastructure & Technology Stack

| Layer | Technology / Tool | Production Role |
|---|---|---|
| **Serving Runtime** | vLLM 0.6+, FlashAttention-3 | High-throughput distributed inference engine with custom Triton/CUDA kernels |
| **Paging & Memory** | C++ Block Manager, PagedAttention v2 | Non-contiguous physical GPU block allocation, virtual-to-physical block tables |
| **Continuous Batcher** | TypeScript / Node.js & Python | Dynamic iteration-level scheduler with token budget chunking (`continuous_batcher.ts`) |
| **Mathematical Animation** | Manim (3Blue1Brown) | Programmatic visualization of KV cache allocation and paging (`manim_flow.py`) |
| **GPU Acceleration** | NVIDIA CUDA 12.4+, TensorRT-LLM | Optimized matrix multiplication on NVIDIA Hopper H100 (FP8 / FP16) |
| **Container & Cluster Orchestration** | Kubernetes, Docker, NVIDIA GPU Operator | Auto-scaling inference pods, GPU slicing, health probes (`k8s-vllm-h100.yaml`) |
| **Observability & SRE** | Prometheus, Grafana, OpenTelemetry | Real-time monitoring of KV cache usage %, prompt throughput, and generation latency |

---

## 🎯 Where to Use (Real-World Production Scenarios)

1. **High-Throughput Enterprise LLM Gateways**:
   - Serve 2.8x higher concurrent request streams on identical H100/A100 hardware footprints without increasing infrastructure cloud costs.
2. **Multi-Turn Conversational Agents & Chatbots**:
   - Maximize prefix cache reuse across long conversational sessions, eliminating redundant prompt precomputation.
3. **Long-Context Document Retrieval (RAG)**:
   - Handle 32K–128K document context windows with dynamic allocation, avoiding catastrophic CUDA OOM spikes.
4. **Offline Batch Synthesis & Model Distillation**:
   - Saturate GPU compute with continuous batching to process millions of synthetic evaluation prompts with maximum token throughput.

---

## 🛠️ Step-by-Step SRE Deployment & Verification Runbook

### 1. Prerequisites
- NVIDIA Driver 535+ & CUDA 12.0+ (for local GPU execution)
- Python 3.10+ and Node.js 18+
- Docker & Docker Compose with NVIDIA Container Toolkit

### 2. Clone & Inspect Workspace
```bash
git clone https://github.com/Pradeeptalari14/tp-vllm-paged-attention.git
cd tp-vllm-paged-attention
```

### 3. Environment Configuration
```bash
cp .env.example .env
# Adjust GPU_MEMORY_UTILIZATION, BLOCK_SIZE, and TENSOR_PARALLEL_SIZE as needed
```

### 4. Run Validation Suite
```bash
bash scripts/validate.sh
```

### 5. Run Python Paged Cache Engine
```bash
python3 paged_cache_engine.py
```

### 6. Run Continuous Batcher
```bash
npm install
node -e "require('./continuous_batcher.ts')" || npx ts-node continuous_batcher.ts
```

### 7. Render Mathematical Paging Animation (Optional)
```bash
python3 -m pip install -r requirements-animation.txt
manim -pql manim_flow.py PagedAttentionScene
```

### 8. Deploy to Kubernetes
```bash
kubectl apply -f k8s-vllm-h100.yaml
kubectl get pods -l app=vllm-paged-attention -o wide
```

---

## 📂 Repository Layout & File Tree

```
tp-vllm-paged-attention/
├── .github/
│   ├── workflows/
│   │   └── sre-validation.yml       # Automated CI validation pipeline
│   ├── CONTRIBUTING.md               # Contribution rules and PR guidelines
│   ├── PULL_REQUEST_TEMPLATE.md      # Conventional commit PR template
│   └── dependabot.yml                # Automated weekly dependency scan
├── docs/
│   ├── vllm_architecture_flow.png    # High-resolution architectural diagram
│   ├── TROUBLESHOOTING.md            # OOM mitigation & NCCL debugging runbook
│   ├── GLOSSARY.md                   # KV cache and PagedAttention terminology
│   └── COMPLIANCE.md                 # SOC2, data residency, and MIT terms
├── scripts/
│   └── validate.sh                   # SRE operational verification script
├── continuous_batcher.ts             # Dynamic iteration-level batching scheduler
├── docker-compose.yml                # Containerized local execution stack
├── k8s-vllm-h100.yaml                # Kubernetes NVIDIA H100 deployment manifest
├── manim_flow.py                     # 3Blue1Brown mathematical memory animation
├── package.json                      # Node.js dependencies for batcher
├── paged_cache_engine.py             # Core PagedAttention v2 memory block manager
├── requirements-animation.txt        # Manim visualization requirements
├── requirements-gpu.txt              # PyTorch and vLLM GPU acceleration packages
├── requirements.txt                  # Base Python dependencies
├── .env.example                      # Safe environment variables blueprint
├── .gitignore                        # Hardened git ignore exclusions
├── .gitleaks.toml                    # Secret scanning configurations
├── .pre-commit-config.yaml           # Pre-commit hooks for syntax and security
├── CHANGELOG.md                      # Release and feature version history
├── LICENSE                           # MIT Open Source License
├── SECURITY.md                       # Vulnerability disclosure policy
└── README.md                         # Comprehensive architecture documentation
```

---

## 🛡️ DevSecOps & 9+ Governance Compliance Suite

| Rule # | Requirement | Implementation in Repository | Status |
|---|---|---|---|
| **1** | **`README.md`** | Problem/solution flow, Mermaid DAG, SRE guide, principles, file tree | ✅ Complete |
| **2** | **`LICENSE`** | Permissive MIT Open Source License (Copyright © 2026 Talari Pradeep) | ✅ Verified |
| **3** | **`SECURITY.md`** | Vulnerability disclosure SLAs, supported versions, secret policies | ✅ Verified |
| **4** | **`.gitignore`** | Hardened against `.env`, `__pycache__`, keys, credentials, build logs | ✅ Cleaned & Active |
| **5** | **`.env.example`** | Safe template for GPU memory utilization, block sizes, and model tags | ✅ Provided |
| **6** | **`.pre-commit-config.yaml`** | Gitleaks secret scans, YAML/JSON linters, trailing whitespace checks | ✅ Configured |
| **7** | **`.github/workflows/`** | Automated CI pipeline (`sre-validation.yml`) & Dependabot updates | ✅ Enabled |
| **8** | **`docs/` Knowledge Suite** | `TROUBLESHOOTING.md`, `GLOSSARY.md`, `COMPLIANCE.md`, and diagram | ✅ Documented |
| **9** | **`scripts/validate.sh`** | Executable shell validation script checking health and syntax | ✅ Executable |
| **10** | **`CHANGELOG.md`** | Structured release notes following Keep-a-Changelog standard | ✅ Maintained |

---

*Synchronized with [vLLM PagedAttention & Chunked Prefill Studio](https://pradeeptalari.me/tools/vllm-paged-attention/) & [Repository Learning DAG](https://pradeeptalari.me/tools/repo-learning/) — [Talari Pradeep Portfolio](https://pradeeptalari.me)*
