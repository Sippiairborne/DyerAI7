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

"""LLM-as-Judge (Zheng et al. 2023, MT-Bench)."""
from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass
from pathlib import Path

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class JudgmentResult:
    score: float
    reasoning: str
    criteria_scores: dict[str, float]
    raw: str


class LLMJudge:
    """Pairwise and pointwise judging with multiple LLM judges."""

    def __init__(self, llm: LLMClient, n_judges: int = 3, position_swap: bool = True) -> None:
        self.llm = llm
        self.n_judges = n_judges
        self.position_swap = position_swap

    async def judge_pairwise(self, prompt: str, response_a: str, response_b: str, reference: str = "") -> dict:
        """Pairwise comparison with position debiasing."""
        votes = {"A": 0, "B": 0, "tie": 0}
        judgments = []
        orderings = [(response_a, response_b, "A"), (response_b, response_a, "B")] if self.position_swap else [(response_a, response_b, "A")]
        for _ in range(self.n_judges):
            for first, second, label in orderings:
                j = await self._pairwise_judgment(prompt, first, second, reference)
                if j["winner"] == "tie":
                    votes["tie"] += 1
                elif j["winner"] == "first":
                    if label == "A":
                        votes["A"] += 1
                    else:
                        votes["B"] += 1
                else:
                    if label == "A":
                        votes["B"] += 1
                    else:
                        votes["A"] += 1
                judgments.append(j)
        winner = "A" if votes["A"] > votes["B"] else "B" if votes["B"] > votes["A"] else "tie"
        return {"winner": winner, "votes": votes, "judgments": judgments}

    async def judge_pointwise(self, prompt: str, response: str, rubric: str = "", scale: int = 10) -> JudgmentResult:
        criteria = ["accuracy", "relevance", "clarity", "depth", "creativity"]
        criteria_scores: dict[str, float] = {}
        reasoning_parts = []
        for c in criteria:
            score_resp = await self.llm.complete(
                messages=[
                    Message(role="system", content=f"You are a rigorous judge scoring {c} on 1-{scale}."),
                    Message(role="user", content=f"Task: {prompt}\nRubric: {rubric}\n\nResponse:\n{response}\n\n{c} score (1-{scale}):"),
                ],
                temperature=0.0,
                max_tokens=20,
            )
            import re
            m = re.search(r"(\d+(?:\.\d+)?)", score_resp.content)
            s = float(m.group(1)) if m else scale / 2
            criteria_scores[c] = s
            reasoning_parts.append(f"{c}: {s}")
        avg = sum(criteria_scores.values()) / len(criteria_scores)
        return JudgmentResult(score=avg / scale, reasoning="\n".join(reasoning_parts), criteria_scores=criteria_scores, raw=response)

    async def _pairwise_judgment(self, prompt: str, a: str, b: str, reference: str = "") -> dict:
        ref_block = f"\nReference answer: {reference}" if reference else ""
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="You compare two responses. Pick A, B, or tie. Provide reasoning."),
                Message(role="user", content=f"Task: {prompt}{ref_block}\n\nResponse A:\n{a}\n\nResponse B:\n{b}\n\nVerdict (A/B/tie):"),
            ],
            temperature=0.0,
            max_tokens=300,
        )
        text = resp.content.upper()
        winner = "tie"
        if "VERDICT: A" in text or text.strip().startswith("A") or "RESPONSE A" in text and "RESPONSE B" not in text:
            winner = "first"
        elif "VERDICT: B" in text or text.strip().startswith("B"):
            winner = "second"
        return {"winner": winner, "reasoning": resp.content}


class MTBenchEval:
    """MT-Bench style multi-turn benchmark."""

    def __init__(self, llm: LLMClient, judge: LLMJudge) -> None:
        self.llm = llm
        self.judge = judge
        self.categories = ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"]

    async def evaluate_model(self, model_fn, questions: list[dict]) -> dict:
        results = []
        for q in questions:
            turns = q.get("turns", [q["question"]])
            conversation: list[Message] = [Message(role="user", content=turns[0])]
            responses = []
            for turn in turns:
                resp = await model_fn(conversation)
                responses.append(resp)
                conversation.append(Message(role="assistant", content=resp))
                if turn != turns[-1]:
                    conversation.append(Message(role="user", content=turns[turns.index(turn) + 1]))
            # Judge
            judgment = await self.judge.judge_pointwise(turns[0], responses[0], scale=10)
            results.append({"category": q.get("category", "general"), "question_id": q.get("question_id"), "score": judgment.score, "criteria": judgment.criteria_scores})
        # Aggregate
        per_category = {}
        for cat in self.categories:
            scores = [r["score"] for r in results if r["category"] == cat]
            if scores:
                per_category[cat] = {"mean": sum(scores) / len(scores), "n": len(scores)}
        overall = sum(r["score"] for r in results) / max(len(results), 1)
        return {"overall": overall, "per_category": per_category, "results": results}
