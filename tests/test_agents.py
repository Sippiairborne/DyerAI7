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

from ai_engineer.agents.data_engineer import DataEngineerAgent
from ai_engineer.core.llm import LLMClient
from ai_engineer.core.memory import MemorySystem
from ai_engineer.tools.registry import ToolRegistry
from ai_engineer.tools.sandbox import Sandbox


@pytest.mark.asyncio
async def test_data_engineer_runs() -> None:
    llm = LLMClient()
    memory = MemorySystem(llm)
    await memory.init()
    sandbox = Sandbox(backend="local")
    agent = DataEngineerAgent(llm, memory, ToolRegistry(), sandbox)
    result = await agent.run(
        {
            "description": "Create a list of integers 1..5, save to /tmp/nums.json, print its sum.",
            "acceptance_criteria": ["file exists", "sum printed"],
        }
    )
    assert result["stdout"] or result["stderr"]
    await llm.close()
    await memory.close()
