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

"""Base agent class."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.core.memory import MemorySystem
from ai_engineer.tools.registry import ToolRegistry
from ai_engineer.tools.sandbox import Sandbox
from ai_engineer.utils.logging import get_logger
from ai_engineer.utils.prompts import ORCHESTRATOR_SYSTEM

logger = get_logger(__name__)


class BaseAgent(ABC):
    name: str = "base"
    system_prompt: str = ORCHESTRATOR_SYSTEM

    def __init__(
        self,
        llm: LLMClient,
        memory: MemorySystem,
        tools: ToolRegistry,
        sandbox: Sandbox,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.sandbox = sandbox

    @abstractmethod
    async def run(self, task: dict[str, Any]) -> dict[str, Any]:
        """Execute a task and return a result dict."""
        raise NotImplementedError

    def system_message(self) -> Message:
        return Message(role="system", content=self.system_prompt)
