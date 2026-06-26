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

"""Constitutional synthetic data — generate, critique, revise using principles."""
from __future__ import annotations

import json
from dataclasses import dataclass

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.future.reasoning.constitutional_ai import ConstitutionalAI
from ai_engineer.utils.logging import get_logger


@dataclass
class ConstitutionalSample:
    prompt: str
    chosen: str
    rejected: str
    principle: str


class ConstitutionalDataGenerator:
    """Generate preference data using Constitutional AI."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.cai = ConstitutionalAI(llm)

    async def generate_pairs(self, prompts: list[str]) -> list[ConstitutionalSample]:
        out: list[ConstitutionalSample] = []
        for p in prompts:
            # Generate two candidates
            a = await self.llm.complete(messages=[Message(role="user", content=p)], temperature=0.7, max_tokens=512)
            b = await self.llm.complete(messages=[Message(role="user", content=p)], temperature=0.9, max_tokens=512)
            for principle in self.cai.principles:
                winner, loser = await self.cai.rlaif_preferences(p, a.content, b.content)
                out.append(ConstitutionalSample(prompt=p, chosen=winner, rejected=loser, principle=principle.text))
        return out

    def export_dpo(self, samples: list[ConstitutionalSample], path: str) -> None:
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps({"prompt": s.prompt, "chosen": s.chosen, "rejected": s.rejected}) + "\n")
