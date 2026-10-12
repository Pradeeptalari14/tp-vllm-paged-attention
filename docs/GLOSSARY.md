# 📖 Glossary — vLLM PagedAttention

| Term | Definition |
|---|---|
| **PagedAttention** | Memory-efficient attention algorithm inspired by OS virtual memory paging, eliminating KV cache memory fragmentation. |
| **KV Cache** | Storage of Key and Value projection tensors across autoregressive decoding steps to prevent redundant matrix recomputations. |
| **Block Size** | The number of tokens grouped into a single non-contiguous physical GPU memory block (typically 16 or 32). |
| **Continuous Batching** | Dynamic iteration-level batching scheduling new prompt prefill tokens alongside ongoing decoding generations. |
