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

"""Reflexion (Shinn et al. 2023) — verbal reinforcement learning via self-reflection memory."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReflexionMemory:
    reflections: list[str] = field(default_factory=list)


@dataclass
class ReflexionResult:
    final_answer: str
    attempts: list[dict]
    reflections: list[str]
    success: bool


class Reflexion:
    """Agent reflects on failures and stores insights for future attempts."""

    def __init__(self, llm: LLMClient, max_attempts: int = 5) -> None:
        self.llm = llm
        self.max_attempts = max_attempts
        self.memory = ReflexionMemory()

    async def attempt(self, task: str, action_fn, success_fn) -> ReflexionResult:
        attempts: list[dict] = []
        for i in range(self.max_attempts):
            ctx = "\n".join(f"- {r}" for r in self.memory.reflections) if self.memory.reflections else "None"
            attempt = await action_fn(task, ctx)
            ok, obs = success_fn(attempt)
            attempts.append({"attempt": i, "action": attempt, "success": ok, "observation": obs})
            if ok:
                return ReflexionResult(final_answer=attempt, attempts=attempts, reflections=self.memory.reflections, success=True)
            reflection = await self._reflect(task, attempt, obs)
            self.memory.reflections.append(reflection)
        return ReflexionResult(final_answer=attempts[-1]["action"] if attempts else "", attempts=attempts, reflections=self.memory.reflections, success=False)

    async def _reflect(self, task: str, attempt: str, observation: str) -> str:
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="Reflect on this failure and provide an improved strategy for next attempt."),
                Message(role="user", content=f"Task: {task}\nAttempt: {attempt}\nOutcome: {observation}\n\nReflection:"),
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return resp.content
