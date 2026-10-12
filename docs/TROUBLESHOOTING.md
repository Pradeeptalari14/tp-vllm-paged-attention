# 🔧 Troubleshooting Guide — vLLM PagedAttention

## Issue 1: CUDA Out of Memory (OOM) during KV Cache Initialization
**Symptom:** `RuntimeError: CUDA out of memory. Tried to allocate X GiB`
**Resolution:**
- Lower `GPU_MEMORY_UTILIZATION` in `.env` from `0.95` to `0.85`.
- Reduce `MAX_NUM_SEQS` to limit concurrently handled request tokens.
- Enable CPU swap space via `SWAP_SPACE_GB=16`.

## Issue 2: Kernel Launch Failure on Multi-GPU Tensor Parallelism
**Symptom:** `NCCL error: unhandled system error`
**Resolution:**
- Verify NVLink / PCIe topology using `nvidia-smi topo -m`.
- Ensure `NCCL_DEBUG=INFO` is set to inspect interconnect handshake latency.
