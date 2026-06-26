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

"""Self-Refine: iterative refinement with self-feedback."""
from __future__ import annotations

import time
from dataclasses import dataclass

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RefineResult:
    final_output: str
    iterations: int
    history: list[dict]


class SelfRefine:
    """Self-Refine (Madaan et al. 2023) — iterative generate → feedback → refine."""

    def __init__(self, llm: LLMClient, max_iter: int = 5, min_improvement: float = 0.05) -> None:
        self.llm = llm
        self.max_iter = max_iter
        self.min_improvement = min_improvement

    async def refine(self, initial: str, task_context: str = "") -> RefineResult:
        history: list[dict] = []
        current = initial
        prev_score = await self._score(current, task_context)
        for i in range(self.max_iter):
            feedback = await self._feedback(current, task_context)
            refined = await self._apply(current, feedback, task_context)
            new_score = await self._score(refined, task_context)
            history.append({"iter": i, "feedback": feedback, "score_before": prev_score, "score_after": new_score, "output": refined})
            logger.info("self_refine.iter", iter=i, prev=prev_score, new=new_score)
            if new_score - prev_score < self.min_improvement:
                break
            current = refined
            prev_score = new_score
        return RefineResult(final_output=current, iterations=len(history), history=history)

    async def _feedback(self, output: str, context: str) -> str:
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="You are a rigorous critic. Identify specific weaknesses, errors, and concrete improvements."),
                Message(role="user", content=f"Context: {context}\n\nOutput to critique:\n{output}\n\nConcrete feedback:"),
            ],
            temperature=0.3,
            max_tokens=600,
        )
        return resp.content

    async def _apply(self, output: str, feedback: str, context: str) -> str:
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="Apply the feedback to produce a refined version."),
                Message(role="user", content=f"Context: {context}\n\nOriginal:\n{output}\n\nFeedback:\n{feedback}\n\nRefined:"),
            ],
            temperature=0.4,
            max_tokens=2048,
        )
        return resp.content

    async def _score(self, output: str, context: str) -> float:
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="Rate the quality of this output on a 0-1 scale. Respond with just a number."),
                Message(role="user", content=f"Context: {context}\n\nOutput:\n{output}\n\nScore:"),
            ],
            temperature=0.0,
            max_tokens=10,
        )
        import re
        m = re.search(r"(\d+\.?\d*)", resp.content)
        return float(m.group(1)) if m else 0.5
