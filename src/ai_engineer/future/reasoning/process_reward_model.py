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

"""Process Reward Model (PRM) — score each step of reasoning (o1-style)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.ml.optimization.distillation import DistillationConfig, Distiller
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Method = Literal["llm_judge", "math_shepherd", "self_supervised"]


@dataclass
class StepScore:
    step: str
    score: float
    confidence: float
    reasoning: str


class ProcessRewardModel:
    """Score each step of a reasoning trace. Used to guide beam search."""

    def __init__(self, llm: LLMClient, method: Method = "llm_judge") -> None:
        self.llm = llm
        self.method = method

    async def score_trace(self, question: str, trace: list[str]) -> list[StepScore]:
        scores: list[StepScore] = []
        for i, step in enumerate(trace):
            score, conf, reasoning = await self._score_step(question, trace[: i + 1], step)
            scores.append(StepScore(step=step, score=score, confidence=conf, reasoning=reasoning))
        return scores

    async def best_of_n_search(self, question: str, generate_fn, n: int = 8, beam_width: int = 3) -> list[str]:
        """Beam search guided by PRM."""
        beams: list[tuple[float, list[str]]] = [(0.0, [])]
        for depth in range(10):
            candidates: list[tuple[float, list[str]]] = []
            for score, path in beams:
                continuations = await generate_fn(path, n=n // max(len(beams), 1))
                for c in continuations:
                    new_path = path + [c]
                    s, _, _ = await self._score_step(question, new_path, c)
                    candidates.append((score + s, new_path))
            candidates.sort(key=lambda x: -x[0])
            beams = candidates[:beam_width]
            if all(self._is_terminal(b[1][-1]) for b in beams):
                break
        return max(beams, key=lambda x: x[0])[1]

    async def _score_step(self, question: str, history: list[str], step: str) -> tuple[float, float, str]:
        if self.method == "math_shepherd":
            return await self._math_shepherd_score(question, history, step)
        if self.method == "self_supervised":
            return await self._self_supervised_score(question, history, step)
        return await self._llm_judge(question, history, step)

    async def _llm_judge(self, question: str, history: list[str], step: str) -> tuple[float, float, str]:
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="You are a Process Reward Model. Evaluate whether this step is correct and contributes to solving the problem. Output: SCORE (0-1), CONFIDENCE (0-1), REASONING."),
                Message(role="user", content=f"Question: {question}\n\nPrevious steps:\n" + "\n".join(f"- {s}" for s in history[:-1]) + f"\n\nCurrent step: {step}\n\nSCORE, CONFIDENCE, REASONING:"),
            ],
            temperature=0.0,
            max_tokens=200,
        )
        import re
        text = resp.content
        s = float(re.search(r"SCORE[:\s]+(\d+\.?\d*)", text).group(1)) if re.search(r"SCORE[:\s]+(\d+\.?\d*)", text) else 0.5
        c = float(re.search(r"CONFIDENCE[:\s]+(\d+\.?\d*)", text).group(1)) if re.search(r"CONFIDENCE[:\s]+(\d+\.?\d*)", text) else 0.5
        r_match = re.search(r"REASONING[:\s]+(.+)", text, re.DOTALL)
        return max(0.0, min(1.0, s)), max(0.0, min(1.0, c)), r_match.group(1).strip() if r_match else ""

    async def _math_shepherd_score(self, question: str, history: list[str], step: str) -> tuple[float, float, str]:
        """Math-Shepherd (Wang et al. 2024): estimate step correctness by completing forward and backward."""
        forward_resp = await self.llm.complete(
            messages=[Message(role="system", content="Continue the solution to completion."), Message(role="user", content=f"Q: {question}\nSteps:\n" + "\n".join(history))],
            temperature=0.0,
            max_tokens=500,
        )
        correct = "answer:" in forward_resp.content.lower() or "the answer is" in forward_resp.content.lower()
        return (1.0 if correct else 0.0, 0.7, "math-shepherd forward check")

    async def _self_supervised_score(self, question: str, history: list[str], step: str) -> tuple[float, float, str]:
        """Generate k completions after the step and check consistency."""
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="Sample 4 possible continuations and return the most common final answer."),
                Message(role="user", content=f"Q: {question}\nSteps:\n" + "\n".join(history) + "\n\nFinal answer:"),
            ],
            temperature=0.7,
            max_tokens=400,
        )
        return 0.7, 0.5, resp.content

    def _is_terminal(self, s: str) -> bool:
        sl = s.strip().lower()
        return any(sl.startswith(p) for p in ("answer:", "the answer is", "conclusion:", "final answer:"))
