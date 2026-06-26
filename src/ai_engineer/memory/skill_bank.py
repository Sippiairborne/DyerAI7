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

"""Reusable skill bank — patterns extracted from successful runs."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ai_engineer.config import get_settings
from ai_engineer.memory.vector_store import VectorStore
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Skill:
    id: str
    name: str
    description: str
    code_template: str
    success_count: int = 0
    failure_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total else 0.0


class SkillBank:
    """Stores and retrieves reusable skills/patterns."""

    def __init__(self, vector_store: VectorStore) -> None:
        self.vs = vector_store
        self.collection = "skills"
        self._skills: dict[str, Skill] = {}

    async def init(self) -> None:
        # Use the same Qdrant collection with namespace, or separate collection
        # For simplicity, we keep a separate collection created in init of vector_store
        pass

    async def add(self, skill: Skill) -> None:
        self._skills[skill.id] = skill
        await self.vs.add(
            texts=[f"{skill.name}\n{skill.description}\n{skill.code_template}"],
            metadata=[{"type": "skill", "skill_id": skill.id, "tags": skill.tags}],
            ids=[skill.id],
        )
        logger.info("skill.added", skill_id=skill.id, name=skill.name)

    async def find_relevant(self, query: str, top_k: int = 3) -> list[Skill]:
        results = await self.vs.search(query, top_k=top_k, filter_=[{"key": "type", "match": {"value": "skill"}}])
        return [self._skills[r["payload"]["skill_id"]] for r in results if r["payload"]["skill_id"] in self._skills]

    async def record_outcome(self, skill_id: str, success: bool) -> None:
        if skill_id in self._skills:
            s = self._skills[skill_id]
            if success:
                s.success_count += 1
            else:
                s.failure_count += 1
            s.last_used = time.time()
