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

"""Self-Instruct (Wang et al. 2022) — bootstrap training data from a seed set."""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SyntheticInstruction:
    instruction: str
    input: str
    output: str
    source: str = "self_instruct"


class SelfInstruct:
    """Generate synthetic instruction-tuning data from a small seed set."""

    def __init__(self, llm: LLMClient, n_per_round: int = 8, num_rounds: int = 10, dedup_threshold: float = 0.7) -> None:
        self.llm = llm
        self.n_per_round = n_per_round
        self.num_rounds = num_rounds
        self.dedup_threshold = dedup_threshold

    async def generate(
        self,
        seed_instructions: list[str],
        target_count: int = 1000,
    ) -> list[SyntheticInstruction]:
        pool = list(seed_instructions)
        results: list[SyntheticInstruction] = []
        for _ in range(self.num_rounds):
            if len(results) >= target_count:
                break
            new_insts = await self._sample_batch(pool)
            for inst in new_insts:
                if self._is_unique(inst, results):
                    output = await self._generate_output(inst)
                    results.append(SyntheticInstruction(instruction=inst, input="", output=output))
                    pool.append(inst)
            logger.info("self_instruct.round", pool_size=len(pool), results=len(results))
        return results

    async def _sample_batch(self, pool: list[str]) -> list[str]:
        sample = random.sample(pool, min(6, len(pool)))
        prompt = "You are generating new task instructions.\n\nExamples:\n" + "\n".join(f"- {s}" for s in sample) + f"\n\nGenerate {self.n_per_round} new and distinct tasks. Each on a new line, prefixed 'TASK:'."
        resp = await self.llm.complete(messages=[Message(role="user", content=prompt)], temperature=1.0, max_tokens=1024)
        out = []
        for line in resp.content.splitlines():
            if "TASK:" in line:
                out.append(line.split("TASK:", 1)[1].strip())
        return out

    async def _generate_output(self, instruction: str) -> str:
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content="You are a helpful assistant. Provide a thorough, accurate response to the task."),
                Message(role="user", content=instruction),
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        return resp.content

    def _is_unique(self, instruction: str, existing: list[SyntheticInstruction]) -> bool:
        from difflib import SequenceMatcher
        for e in existing:
            ratio = SequenceMatcher(None, instruction.lower(), e.instruction.lower()).ratio()
            if ratio > self.dedup_threshold:
                return False
        return True

    def export(self, instructions: list[SyntheticInstruction], path: str | Path) -> None:
        with Path(path).open("w") as f:
            for i in instructions:
                f.write(json.dumps({"instruction": i.instruction, "input": i.input, "output": i.output}) + "\n")
