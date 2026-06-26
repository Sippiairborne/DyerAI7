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

import pytest

from ai_engineer.core.llm import LLMClient
from ai_engineer.core.memory import MemorySystem


@pytest.mark.asyncio
async def test_remember_and_recall() -> None:
    llm = LLMClient()
    m = MemorySystem(llm)
    await m.init()
    try:
        await m.remember("The capital of France is Paris.", kind="fact")
        await m.remember("Python is a programming language.", kind="fact")
        hits = await m.recall("What is the capital of France?", top_k=2)
        assert len(hits) >= 1
    finally:
        await llm.close()
        await m.close()
