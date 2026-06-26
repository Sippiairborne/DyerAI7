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

"""Evol-Instruct (WizardLM) — evolve instructions through depth/breadth mutation."""
from __future__ import annotations

import random
from dataclasses import dataclass

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.future.synthetic.self_instruct import SelfInstruct, SyntheticInstruction
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Mutation = Literal := __import__("typing").Literal["deepen", "concretize", "increase_reasoning", "add_constraints", "complicate_input", "broaden"]  # type: ignore


@dataclass
class EvolResult:
    original: str
    evolved: str
    mutation: str


class EvolInstruct:
    """Evol-Instruct pipeline: mutate instructions to increase complexity."""

    MUTATIONS: list[str] = ["deepen", "concretize", "increase_reasoning", "add_constraints", "complicate_input", "broaden"]

    def __init__(self, llm: LLMClient, evolution_rate: float = 0.5, max_attempts: int = 3) -> None:
        self.llm = llm
        self.evolution_rate = evolution_rate
        self.max_attempts = max_attempts

    async def evolve(self, instructions: list[SyntheticInstruction]) -> list[SyntheticInstruction]:
        out: list[SyntheticInstruction] = []
        for inst in instructions:
            if random.random() > self.evolution_rate:
                out.append(inst)
                continue
            mutation = random.choice(self.MUTATIONS)
            evolved = await self._mutate(inst.instruction, mutation)
            if evolved and evolved != inst.instruction:
                output = await self._generate_output(evolved)
                out.append(SyntheticInstruction(instruction=evolved, input=inst.input, output=output, source=f"evol_{mutation}"))
            else:
                out.append(inst)
        return out

    async def _mutate(self, instruction: str, mutation: str) -> str:
        prompts = {
            "deepen": f"Add a deeper layer of complexity or sophistication: {instruction}",
            "concretize": f"Make this instruction more specific and concrete with examples or constraints: {instruction}",
            "increase_reasoning": f"Rewrite to require multi-step reasoning: {instruction}",
            "add_constraints": f"Add meaningful constraints (e.g., length, format, audience): {instruction}",
            "complicate_input": f"Make the input context more realistic and detailed: {instruction}",
            "broaden": f"Generalize this to apply to a broader domain: {instruction}",
        }
        for _ in range(self.max_attempts):
            resp = await self.llm.complete(
                messages=[Message(role="system", content="Rewrite the instruction. Output ONLY the new instruction."), Message(role="user", content=prompts[mutation])],
                temperature=0.7,
                max_tokens=512,
            )
            new = resp.content.strip().strip('"').strip("'")
            if new and len(new) > 10 and new.lower() != instruction.lower():
                return new
        return instruction

    async def _generate_output(self, instruction: str) -> str:
        resp = await self.llm.complete(
            messages=[Message(role="system", content="Provide a thorough response."), Message(role="user", content=instruction)],
            temperature=0.7,
            max_tokens=1024,
        )
        return resp.content
