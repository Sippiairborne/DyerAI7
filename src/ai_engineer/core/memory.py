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

"""Unified memory system combining vector, graph, skill, and trajectory stores."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_engineer.core.llm import LLMClient
from ai_engineer.memory.graph_store import GraphStore
from ai_engineer.memory.skill_bank import Skill, SkillBank
from ai_engineer.memory.trajectory_store import Trajectory, TrajectoryStore
from ai_engineer.memory.vector_store import VectorStore
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MemoryHit:
    text: str
    score: float
    source: str  # "vector" | "graph" | "skill"
    metadata: dict[str, Any]


class MemorySystem:
    """The unified memory layer of the AI engineer."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.vector = VectorStore(llm)
        self.graph = GraphStore()
        self.skills = SkillBank(self.vector)
        self.trajectories = TrajectoryStore()

    async def init(self) -> None:
        await self.vector.init()
        await self.graph.init()
        logger.info("memory.initialized")

    async def close(self) -> None:
        await self.vector.close()
        await self.graph.close()

    async def remember(
        self,
        text: str,
        kind: str = "fact",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        meta = {"kind": kind, **(metadata or {})}
        ids = await self.vector.add([text], metadata=[meta])
        return ids[0]

    async def recall(self, query: str, top_k: int = 5, kind: str | None = None) -> list[MemoryHit]:
        filt = [{"key": "kind", "match": {"value": kind}}] if kind else None
        results = await self.vector.search(query, top_k=top_k, filter_=filt)
        return [MemoryHit(text=str(r["payload"].get("text", "")), score=r["score"], source="vector", metadata=r["payload"]) for r in results]

    async def learn_skill(self, name: str, description: str, template: str, tags: list[str] | None = None) -> Skill:
        import uuid

        skill = Skill(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            code_template=template,
            tags=tags or [],
        )
        await self.skills.add(skill)
        return skill
