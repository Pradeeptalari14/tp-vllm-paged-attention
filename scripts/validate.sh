#!/usr/bin/env bash
set -eo pipefail

echo "=========================================================="
echo "⚡ Validating vLLM PagedAttention & Chunked Prefill SRE"
echo "=========================================================="

python3 -m py_compile paged_cache_engine.py
echo "✓ paged_cache_engine.py syntax validated."

python3 -m py_compile manim_flow.py
echo "✓ manim_flow.py syntax validated."

python3 -c "
from paged_cache_engine import PagedCacheManager
mgr = PagedCacheManager(total_blocks=128, block_size=16)
seq = mgr.allocate_sequence('test-req', list(range(35)))
assert len(seq) == 3, 'Block allocation failed'
assert mgr.get_fragmentation_ratio() < 0.35, 'Fragmentation ratio unexpected'
mgr.free_sequence('test-req')
assert len(mgr.free_blocks) == 128, 'Block leak detected'
print('✓ Unit verification passed: PagedCacheManager alloc/free clean.')
"

echo "=========================================================="
echo "✅ All SRE validation checks passed successfully!"
echo "=========================================================="
