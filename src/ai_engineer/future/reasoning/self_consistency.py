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

"""Self-Consistency (Wang et al. 2022) — sample multiple paths and majority-vote."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from ai_engineer.core.llm import LLMClient, Message


@dataclass
class ConsistencyResult:
    final_answer: str
    samples: list[str]
    votes: dict[str, int]
    agreement_rate: float


class SelfConsistency:
    """Sample diverse reasoning paths, then majority-vote on final answers."""

    def __init__(self, llm: LLMClient, n_samples: int = 7, temperature: float = 0.7) -> None:
        self.llm = llm
        self.n_samples = n_samples
        self.temperature = temperature

    async def solve(self, problem: str, answer_extractor: str = r"(?:answer is|=|:)\s*([^\n.]+)") -> ConsistencyResult:
        samples: list[str] = []
        for _ in range(self.n_samples):
            resp = await self.llm.complete(
                messages=[
                    Message(role="system", content="Think step by step. End with 'The answer is X.'"),
                    Message(role="user", content=problem),
                ],
                temperature=self.temperature,
                max_tokens=1024,
            )
            samples.append(resp.content)
        answers: list[str] = []
        for s in samples:
            m = re.search(answer_extractor, s, re.IGNORECASE)
            answers.append(m.group(1).strip() if m else s.strip().split("\n")[-1])
        votes = Counter(answers)
        final = votes.most_common(1)[0][0] if votes else ""
        return ConsistencyResult(final_answer=final, samples=samples, votes=dict(votes), agreement_rate=votes.most_common(1)[0][1] / max(len(samples), 1))
