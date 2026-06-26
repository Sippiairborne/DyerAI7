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

"""Task planner — decomposes a goal into an executable DAG."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.utils.errors import PlanningError
from ai_engineer.utils.logging import get_logger
from ai_engineer.utils.prompts import PLANNER_SYSTEM

logger = get_logger(__name__)


class PlannedTask(BaseModel):
    id: str
    title: str
    agent: str
    description: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    estimated_minutes: int = 30
    tools_required: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    goal: str
    rationale: str
    tasks: list[PlannedTask]


@dataclass
class TaskNode:
    id: str
    title: str
    agent: str
    description: str
    acceptance_criteria: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    estimated_minutes: int = 30
    tools_required: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | running | done | failed | blocked
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    retries: int = 0


class Planner:
    """Decomposes high-level goals into executable task graphs."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def plan(self, goal: str, context: str = "") -> Plan:
        user_msg = f"GOAL:\n{goal}\n\nCONTEXT (relevant past work, preferences, constraints):\n{context or 'None'}"
        try:
            return await self.llm.structured(
                messages=[
                    Message(role="system", content=PLANNER_SYSTEM),
                    Message(role="user", content=user_msg),
                ],
                schema=Plan,
                temperature=0.3,
            )
        except Exception as e:
            raise PlanningError(f"Planning failed: {e}") from e

    def build_dag(self, plan: Plan) -> dict[str, TaskNode]:
        """Convert a Plan into an in-memory DAG of TaskNodes."""
        nodes: dict[str, TaskNode] = {}
        for t in plan.tasks:
            nodes[t.id] = TaskNode(
                id=t.id,
                title=t.title,
                agent=t.agent,
                description=t.description,
                acceptance_criteria=t.acceptance_criteria,
                depends_on=t.depends_on,
                estimated_minutes=t.estimated_minutes,
                tools_required=t.tools_required,
            )
        return nodes

    def ready_tasks(self, nodes: dict[str, TaskNode]) -> list[TaskNode]:
        return [
            n
            for n in nodes.values()
            if n.status == "pending"
            and all(nodes[d].status == "done" for d in n.depends_on if d in nodes)
        ]

    def is_complete(self, nodes: dict[str, TaskNode]) -> bool:
        return all(n.status in ("done", "failed") for n in nodes.values())

    def has_failures(self, nodes: dict[str, TaskNode]) -> bool:
        return any(n.status == "failed" for n in nodes.values())

    async def replan(
        self,
        goal: str,
        previous_plan: Plan,
        failed_tasks: list[TaskNode],
        observations: str,
    ) -> Plan:
        """Replan after a failure."""
        user_msg = (
            f"GOAL: {goal}\n\n"
            f"PREVIOUS PLAN:\n{json.dumps(previous_plan.model_dump(), indent=2)}\n\n"
            f"FAILED TASKS:\n{json.dumps([{'id': t.id, 'title': t.title, 'error': t.error} for t in failed_tasks], indent=2)}\n\n"
            f"OBSERVATIONS:\n{observations}\n\n"
            "Produce a revised plan that addresses the failures."
        )
        return await self.llm.structured(
            messages=[Message(role="system", content=PLANNER_SYSTEM), Message(role="user", content=user_msg)],
            schema=Plan,
            temperature=0.4,
        )
