"""
vLLM PagedAttention Virtual Memory KV-Cache Allocator
Manages physical GPU blocks in non-contiguous HBM3e without external fragmentation.
"""
from typing import Dict, List, Optional
import time

class PhysicalBlock:
    def __init__(self, block_number: int, block_size: int = 16):
        self.block_number = block_number
        self.block_size = block_size
        self.ref_count = 0
        self.tokens: List[int] = []

    def is_full(self) -> bool:
        return len(self.tokens) >= self.block_size

    def append_token(self, token_id: int):
        if self.is_full():
            raise OverflowError(f"Block {self.block_number} is full.")
        self.tokens.append(token_id)

class PagedCacheManager:
    def __init__(self, total_blocks: int = 8192, block_size: int = 16):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free_blocks: List[PhysicalBlock] = [PhysicalBlock(i, block_size) for i in range(total_blocks)]
        self.allocated_blocks: Dict[int, PhysicalBlock] = {}
        self.sequence_block_table: Dict[str, List[int]] = {}

    def allocate_sequence(self, seq_id: str, prompt_tokens: List[int]) -> List[int]:
        blocks_needed = (len(prompt_tokens) + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < blocks_needed:
            raise MemoryError("Out of VRAM: insufficient free blocks for PagedAttention.")

        block_indices = []
        for i in range(blocks_needed):
            block = self.free_blocks.pop(0)
            block.ref_count = 1
            chunk = prompt_tokens[i * self.block_size : (i + 1) * self.block_size]
            for t in chunk:
                block.append_token(t)
            self.allocated_blocks[block.block_number] = block
            block_indices.append(block.block_number)

        self.sequence_block_table[seq_id] = block_indices
        return block_indices

    def free_sequence(self, seq_id: str):
        if seq_id not in self.sequence_block_table:
            return
        for b_num in self.sequence_block_table[seq_id]:
            block = self.allocated_blocks.pop(b_num, None)
            if block:
                block.ref_count = 0
                block.tokens.clear()
                self.free_blocks.append(block)
        del self.sequence_block_table[seq_id]

    def get_fragmentation_ratio(self) -> float:
        # PagedAttention eliminates external fragmentation to <4%
        used_slots = sum(len(b.tokens) for b in self.allocated_blocks.values())
        allocated_capacity = len(self.allocated_blocks) * self.block_size
        if allocated_capacity == 0:
            return 0.0
        return round((allocated_capacity - used_slots) / allocated_capacity, 4)

if __name__ == "__main__":
    mgr = PagedCacheManager(total_blocks=1024, block_size=16)
    seq = mgr.allocate_sequence("request-001", list(range(42)))
    print(f"Allocated {len(seq)} blocks for 42 tokens. Fragmentation: {mgr.get_fragmentation_ratio() * 100:.1f}%")
