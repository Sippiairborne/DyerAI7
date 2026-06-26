# Copyright 2026 Matt Dyer / Dyer-Tech
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Seed the vector store with initial ML knowledge."""
from __future__ import annotations

import asyncio

from ai_engineer.core.llm import LLMClient
from ai_engineer.core.memory import MemorySystem


SEED_FACTS = [
    "Always set seeds for reproducibility: torch, numpy, random, transformers.",
    "Use bf16 instead of fp16 on Ampere+ GPUs to avoid loss-scaling issues.",
    "Apply gradient checkpointing for models >7B params to fit memory.",
    "Use Liger Kernel or FlashAttention 2 for 1.5-2x training speedup.",
    "For classification, AdamW with lr=2e-5 is a strong default for transformer fine-tunes.",
    "Always evaluate on a held-out test set that was never seen during training or validation.",
    "When fine-tuning LLMs, always include a chat template in the tokenizer.",
    "For RAG, chunk size of 512 tokens with 64-token overlap works well in most domains.",
    "Use loguru or structlog instead of print for production logging.",
    "Quantize to int8 for 2x speedup with <1% accuracy drop on most classification tasks.",
]


async def main() -> None:
    llm = LLMClient()
    memory = MemorySystem(llm)
    await memory.init()
    for fact in SEED_FACTS:
        await memory.remember(fact, kind="best_practice")
    print(f"Seeded {len(SEED_FACTS)} facts")
    await llm.close()
    await memory.close()


if __name__ == "__main__":
    asyncio.run(main())
