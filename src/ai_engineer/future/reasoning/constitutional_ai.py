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

"""Constitutional AI (Bai et al. 2022) — self-critique and revision against principles."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

DEFAULT_PRINCIPLES = [
    "Please choose the response that is least harmful, racist, sexist, socially biased, or unethical.",
    "Please choose the response that is most helpful, informative, and accurate.",
    "Please choose the response that is least false, deceptive, or misleading.",
    "Please choose the response that is most respectful, polite, and considerate.",
    "Please choose the response that is least likely to violate laws or enable wrongdoing.",
]


@dataclass
class ConstitutionPrinciple:
    text: str
    weight: float = 1.0


@dataclass
class CAITrajectory:
    initial: str
    critiques: list[str] = field(default_factory=list)
    revisions: list[str] = field(default_factory=list)
    final: str = ""
    iterations: int = 0
    elapsed_s: float = 0.0


class ConstitutionalAI:
    """Constitutional AI with iterative self-critique and revision."""

    def __init__(self, llm: LLMClient, principles: list[str] | None = None, max_iterations: int = 3) -> None:
        self.llm = llm
        self.principles = [ConstitutionPrinciple(p) for p in (principles or DEFAULT_PRINCIPLES)]
        self.max_iterations = max_iterations

    async def train(self, prompt: str, response: str, critique_samples: int = 1) -> CAITrajectory:
        start = time.time()
        traj = CAITrajectory(initial=response)
        current = response
        for it in range(self.max_iterations):
            for principle in self.principles:
                critique = await self._critique(prompt, current, principle.text)
                traj.critiques.append(critique)
                revised = await self._revise(prompt, current, critique)
                traj.revisions.append(revised)
                current = revised
            traj.iterations = it + 1
        traj.final = current
        traj.elapsed_s = time.time() - start
        return traj

    async def rlaif_preferences(self, prompt: str, response_a: str, response_b: str) -> tuple[str, str]:
        """Generate preference pairs for RLAIF training data."""
        preferences: list[tuple[str, str]] = []
        for principle in self.principles:
            for a, b in [(response_a, response_b), (response_b, response_a)]:
                critique_a = await self._critique(prompt, a, principle.text)
                critique_b = await self._critique(prompt, b, principle.text)
                pref_resp = await self.llm.complete(
                    messages=[
                        Message(role="system", content=f"Apply this principle: {principle.text}\nBased on the critiques, which is better: A or B?"),
                        Message(role="user", content=f"Prompt: {prompt}\n\nA: {a}\nA critique: {critique_a}\n\nB: {b}\nB critique: {critique_b}\n\nChoice (A or B):"),
                    ],
                    temperature=0.0,
                    max_tokens=10,
                )
                choice = "A" if "A" in pref_resp.content.upper() and "B" not in pref_resp.content.upper() else "B"
                chosen = a if choice == "A" else b
                rejected = b if choice == "A" else a
                preferences.append((chosen, rejected))
        # Majority vote
        from collections import Counter
        votes = Counter()
        for chosen, _ in preferences:
            if chosen == response_a:
                votes["A"] += 1
            else:
                votes["B"] += 1
        winner = response_a if votes["A"] > votes["B"] else response_b
        loser = response_b if winner == response_a else response_a
        return winner, loser

    async def _critique(self, prompt: str, response: str, principle: str) -> str:
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content=f"You critique a response based on this principle: {principle}\n\nList specific issues."),
                Message(role="user", content=f"Prompt: {prompt}\n\nResponse: {response}\n\nCritique:"),
            ],
            temperature=0.3,
            max_tokens=400,
        )
        return resp.content

    async def _revise(self, prompt: str, response: str, critique: str) -> str:
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="Revise the response to address the critique while preserving helpful content."),
                Message(role="user", content=f"Prompt: {prompt}\n\nResponse: {response}\n\nCritique: {critique}\n\nRevised:"),
            ],
            temperature=0.4,
            max_tokens=2048,
        )
        return resp.content
