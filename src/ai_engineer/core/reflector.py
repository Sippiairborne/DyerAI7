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

"""Self-reflection module — reviews work and suggests improvements."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.utils.logging import get_logger
from ai_engineer.utils.prompts import REFLECTOR_SYSTEM

logger = get_logger(__name__)


class ReflectionResult(BaseModel):
    issues: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)
    should_retry: bool = False
    new_approach: str = ""
    confidence: float = 0.5


class Reflector:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def reflect(
        self,
        task_description: str,
        action: str,
        observation: str,
        expected: list[str] | None = None,
    ) -> ReflectionResult:
        user_msg = (
            f"TASK: {task_description}\n\n"
            f"ACCEPTANCE CRITERIA:\n" + "\n".join(f"- {c}" for c in (expected or [])) + "\n\n"
            f"ACTION TAKEN:\n{action}\n\n"
            f"OBSERVATION / RESULT:\n{observation}"
        )
        try:
            return await self.llm.structured(
                messages=[
                    Message(role="system", content=REFLECTOR_SYSTEM),
                    Message(role="user", content=user_msg),
                ],
                schema=ReflectionResult,
                temperature=0.2,
            )
        except Exception:
            return ReflectionResult(should_retry=False, confidence=0.5)
