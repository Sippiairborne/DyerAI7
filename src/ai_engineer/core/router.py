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

"""Model router for selecting the right LLM per task."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai_engineer.config import get_settings
from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

TaskType = Literal["planning", "code", "reflection", "embedding", "chat", "evaluation", "fast"]


@dataclass
class RoutingDecision:
    model: str
    temperature: float
    max_tokens: int
    rationale: str


class ModelRouter:
    """Routes tasks to the appropriate model variant."""

    def __init__(self, client: LLMClient) -> None:
        self.client = client
        s = get_settings()
        self.models = {
            "main": s.llm_model,
            "fast": s.llm_fast_model,
            "code": s.llm_code_model,
        }

    def route(self, task_type: TaskType, complexity: float = 0.5) -> RoutingDecision:
        if task_type == "code":
            return RoutingDecision(self.models["code"], 0.1, 8192, "Code task → code model")
        if task_type == "planning":
            return RoutingDecision(self.models["main"], 0.3, 4096, "Planning needs reasoning")
        if task_type == "reflection":
            return RoutingDecision(self.models["main"], 0.2, 2048, "Reflection needs care")
        if task_type == "evaluation":
            return RoutingDecision(self.models["main"], 0.1, 2048, "Evaluation needs care")
        if task_type == "fast" or complexity < 0.3:
            return RoutingDecision(self.models["fast"], 0.2, 1024, "Low complexity → fast model")
        return RoutingDecision(self.models["main"], 0.2, 4096, "Default → main model")

    async def complete(
        self,
        task_type: TaskType,
        messages: list[Message],
        complexity: float = 0.5,
        **kwargs: object,
    ) -> object:
        decision = self.route(task_type, complexity)
        logger.info(
            "routing.decision",
            task_type=task_type,
            complexity=complexity,
            model=decision.model,
            rationale=decision.rationale,
        )
        return await self.client.complete(
            messages=messages,
            model=decision.model,
            temperature=decision.temperature,
            max_tokens=decision.max_tokens,
            **kwargs,
        )
