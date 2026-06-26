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

"""Agent Mesh Coordinator — orchestrate many specialized agents."""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.future.agent_mesh.a2a_protocol import A2AMessage, A2AProtocol
from ai_engineer.future.agent_mesh.shared_memory import SharedBlackboard
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AgentSpec:
    name: str
    role: str
    system_prompt: str
    tools: list[str] = field(default_factory=list)


@dataclass
class MeshTask:
    id: str
    description: str
    status: str = "pending"
    assigned_to: str | None = None
    result: Any = None
    history: list[dict] = field(default_factory=list)


class AgentMeshCoordinator:
    """Coordinate a mesh of agents via shared blackboard + A2A messages."""

    def __init__(self, llm: LLMClient, agents: list[AgentSpec]) -> None:
        self.llm = llm
        self.agents = {a.name: a for a in agents}
        self.blackboard = SharedBlackboard()
        self.a2a = A2AProtocol()
        self.tasks: dict[str, MeshTask] = {}

    async def solve(self, goal: str) -> dict:
        # Decompose into subtasks via the mesh
        self.blackboard.write("goal", goal)
        # Initial decomposition
        subtasks = await self._decompose(goal)
        for st in subtasks:
            self.tasks[st["id"]] = MeshTask(id=st["id"], description=st["description"])
        # Iterative assignment
        while not all(t.status in ("done", "failed") for t in self.tasks.values()):
            await self._step()
        return {"goal": goal, "results": {t.id: t.result for t in self.tasks.values()}}

    async def _decompose(self, goal: str) -> list[dict]:
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content=f"Available agents: {list(self.agents.keys())}. Decompose the goal into 3-7 subtasks. Output JSON: [{{'id': 't1', 'description': '...', 'agent': '...'}}, ...]"),
                Message(role="user", content=goal),
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        import json
        try:
            text = resp.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
            return json.loads(text)
        except Exception:
            return [{"id": "t1", "description": goal, "agent": list(self.agents.keys())[0]}]

    async def _step(self) -> None:
        # Pick next pending task
        next_task = next((t for t in self.tasks.values() if t.status == "pending"), None)
        if not next_task:
            return
        # Choose best agent via blackboard
        chosen = await self._select_agent(next_task)
        next_task.assigned_to = chosen
        next_task.status = "running"
        try:
            result = await self._execute_agent(chosen, next_task.description)
            next_task.result = result
            next_task.status = "done"
            # Broadcast via A2A
            await self.a2a.broadcast(A2AMessage(
                sender=chosen, recipient="*",
                kind="task_complete",
                payload={"task_id": next_task.id, "result": str(result)[:500]},
            ))
            self.blackboard.write(f"task:{next_task.id}", result)
        except Exception as e:
            next_task.status = "failed"
            next_task.result = str(e)
            logger.warning("mesh.task_failed", task=next_task.id, error=str(e))

    async def _select_agent(self, task: MeshTask) -> str:
        # Use blackboard's past performance to pick best agent
        scores = defaultdict(float)
        for name in self.agents:
            success_rate = self.blackboard.get(f"agent:{name}:success_rate", 0.5)
            scores[name] = success_rate
        # Soft pick top
        best = max(scores.items(), key=lambda x: x[1])
        return best[0]

    async def _execute_agent(self, agent_name: str, task_desc: str) -> Any:
        spec = self.agents[agent_name]
        ctx = self.blackboard.snapshot()
        resp = await self.llm.complete(
            messages=[
                Message(role="system", content=spec.system_prompt + "\n\nShared context:\n" + ctx),
                Message(role="user", content=task_desc),
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        # Update blackboard with success
        old = self.blackboard.get(f"agent:{agent_name}:success_rate", 0.5)
        self.blackboard.write(f"agent:{agent_name}:success_rate", old * 0.9 + 0.1)
        return resp.content
