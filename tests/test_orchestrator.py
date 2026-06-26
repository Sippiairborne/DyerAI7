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
from ai_engineer.core.orchestrator import Orchestrator
from ai_engineer.tools.registry import ToolRegistry
from ai_engineer.tools.sandbox import Sandbox


@pytest.mark.asyncio
async def test_run_goal_simple() -> None:
    llm = LLMClient()
    memory = MemorySystem(llm)
    await memory.init()
    sandbox = Sandbox(backend="local")
    tools = ToolRegistry()
    orch = Orchestrator(llm, memory, tools, sandbox)
    result = await orch.run_goal("Write a Python function that adds two numbers.", max_replans=0, max_task_retries=0)
    assert "success" in result
    await llm.close()
    await memory.close()
