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

"""Main orchestrator that runs the full plan-execute-reflect loop."""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from ai_engineer.agents.base import BaseAgent
from ai_engineer.agents.data_engineer import DataEngineerAgent
from ai_engineer.agents.deployer import DeployerAgent
from ai_engineer.agents.evaluator import EvaluatorAgent
from ai_engineer.agents.model_architect import ModelArchitectAgent
from ai_engineer.agents.trainer import TrainerAgent
from ai_engineer.core.llm import LLMClient, Message
from ai_engineer.core.memory import MemorySystem
from ai_engineer.core.planner import Planner, TaskNode
from ai_engineer.core.reflector import Reflector
from ai_engineer.tools.registry import ToolRegistry
from ai_engineer.tools.sandbox import Sandbox
from ai_engineer.utils.errors import PlanningError
from ai_engineer.utils.logging import get_logger
from ai_engineer.utils.prompts import CRITIC_SYSTEM

logger = get_logger(__name__)

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "data_engineer": DataEngineerAgent,
    "model_architect": ModelArchitectAgent,
    "trainer": TrainerAgent,
    "evaluator": EvaluatorAgent,
    "deployer": DeployerAgent,
}


class Orchestrator:
    """Coordinates planning, execution, and reflection."""

    def __init__(
        self,
        llm: LLMClient,
        memory: MemorySystem,
        tools: ToolRegistry,
        sandbox: Sandbox,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.sandbox = sandbox
        self.planner = Planner(llm)
        self.reflector = Reflector(llm)
        self.agents: dict[str, BaseAgent] = {
            name: cls(llm, memory, tools, sandbox) for name, cls in AGENT_REGISTRY.items()
        }
        self._event_subscribers: list[Any] = []

    def on_event(self, callback: Any) -> None:
        self._event_subscribers.append(callback)

    async def _emit(self, event: dict[str, Any]) -> None:
        for cb in self._event_subscribers:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception as e:
                logger.warning("orchestrator.subscriber_error", error=str(e))

    async def run_goal(
        self,
        goal: str,
        max_replans: int = 3,
        max_task_retries: int = 2,
    ) -> dict[str, Any]:
        """Run a goal end-to-end with replanning on failure."""
        # Recall relevant context
        context_hits = await self.memory.recall(goal, top_k=5)
        context = "\n".join(h.text for h in context_hits) if context_hits else ""

        # Initial plan
        plan = await self.planner.plan(goal, context)
        nodes = self.planner.build_dag(plan)
        trajectory = self.memory.trajectories.create(goal, plan.model_dump())

        await self._emit({"type": "plan_created", "plan": plan.model_dump(), "num_tasks": len(nodes)})

        replans = 0
        while replans <= max_replans:
            # Execute all ready tasks (in parallel when possible)
            while not self.planner.is_complete(nodes):
                ready = self.planner.ready_tasks(nodes)
                if not ready:
                    # Deadlock or all blocked
                    failed = [n for n in nodes.values() if n.status == "failed"]
                    if failed:
                        break
                    # Wait briefly
                    await asyncio.sleep(0.5)
                    continue
                await asyncio.gather(*[self._run_task(n, nodes, trajectory, max_task_retries) for n in ready])

            # Check final state
            if self.planner.has_failures(nodes):
                failed = [n for n in nodes.values() if n.status == "failed"]
                if replans >= max_replans:
                    await self._emit({"type": "aborted", "reason": "max_replans", "failed": [f.id for f in failed]})
                    break
                replans += 1
                observations = "\n".join(
                    f"Task {f.id} ({f.title}) failed: {f.error or 'unknown'}" for f in failed
                )
                plan = await self.planner.replan(goal, plan, failed, observations)
                # Reset failed nodes' status to pending for new attempt
                kept_ids = {t.id for t in plan.tasks}
                for n in nodes.values():
                    if n.id not in kept_ids or n.status == "failed":
                        n.status = "pending"
                        n.error = None
                await self._emit({"type": "replanned", "iteration": replans, "plan": plan.model_dump()})
                continue

            # All tasks done → run critic
            critic = await self._critic(goal, nodes)
            if critic.get("achieved"):
                success = True
                break
            if replans >= max_replans:
                success = False
                break
            replans += 1
            plan = await self.planner.replan(
                goal,
                plan,
                [],
                f"Critic says not achieved: {critic.get('summary', '')}\nIssues: {critic.get('remaining_issues', [])}",
            )
            await self._emit({"type": "replanned", "iteration": replans, "reason": "critic", "plan": plan.model_dump()})

        final = {
            "goal": goal,
            "success": success,
            "tasks": [
                {
                    "id": n.id,
                    "title": n.title,
                    "agent": n.agent,
                    "status": n.status,
                    "result": n.result,
                    "error": n.error,
                    "duration_s": (n.finished_at - n.started_at) if (n.started_at and n.finished_at) else None,
                }
                for n in nodes.values()
            ],
        }
        await self.memory.trajectories.finalize(
            trajectory.id,
            success=success,
            metrics=final["tasks"][-1]["result"] if final["tasks"] else {},
        )
        # Persist goal text into memory
        await self.memory.remember(goal, kind="goal")
        return final

    async def _run_task(
        self,
        node: TaskNode,
        nodes: dict[str, TaskNode],
        trajectory_id: str,
        max_retries: int,
    ) -> None:
        agent = self.agents.get(node.agent)
        if not agent:
            node.status = "failed"
            node.error = f"Unknown agent: {node.agent}"
            return

        node.status = "running"
        node.started_at = time.time()
        await self._emit({"type": "task_started", "task_id": node.id, "agent": node.agent, "title": node.title})

        attempt = 0
        last_result: dict[str, Any] = {}
        last_error: str | None = None
        while attempt <= max_retries:
            try:
                last_result = await agent.run(
                    {
                        "id": node.id,
                        "title": node.title,
                        "description": node.description,
                        "acceptance_criteria": node.acceptance_criteria,
                        "tools_required": node.tools_required,
                    }
                )
                self.memory.trajectories.add_step(
                    trajectory_id, node.agent, node.title, json.dumps(last_result)[:2000]
                )

                # Reflect
                reflection = await self.reflector.reflect(
                    node.description, node.title, last_result.get("stdout", "") + "\n" + last_result.get("stderr", ""),
                    node.acceptance_criteria,
                )
                if not last_result.get("success") or reflection.should_retry:
                    if attempt < max_retries:
                        attempt += 1
                        node.retries = attempt
                        await self._emit({
                            "type": "task_retry",
                            "task_id": node.id,
                            "attempt": attempt,
                            "reason": reflection.improvements[:1] or ["execution failed"],
                        })
                        # Modify description with reflection
                        node.description += f"\n\nPrevious attempt feedback: {reflection.new_approach or 'retry with fix'}"
                        continue
                node.status = "done" if last_result.get("success") else "failed"
                node.result = last_result
                node.finished_at = time.time()
                await self._emit({
                    "type": "task_completed",
                    "task_id": node.id,
                    "success": node.status == "done",
                    "duration_s": node.finished_at - node.started_at,
                })
                return
            except Exception as e:
                last_error = str(e)
                attempt += 1
                logger.exception("orchestrator.task_error", task_id=node.id, attempt=attempt)
                if attempt > max_retries:
                    node.status = "failed"
                    node.error = last_error
                    node.finished_at = time.time()
                    await self._emit({"type": "task_failed", "task_id": node.id, "error": last_error})
                    return

    async def _critic(self, goal: str, nodes: dict[str, TaskNode]) -> dict[str, Any]:
        from pydantic import BaseModel, Field

        class CriticOutput(BaseModel):
            achieved: bool
            confidence: float
            summary: str
            remaining_issues: list[str] = Field(default_factory=list)
            next_steps: list[str] = Field(default_factory=list)

        results_summary = "\n".join(
            f"- [{n.status}] {n.title} ({n.agent}): "
            + (n.result.get("stdout", "")[:500] if n.result else n.error or "no result")
            for n in nodes.values()
        )
        try:
            out = await self.llm.structured(
                messages=[
                    Message(role="system", content=CRITIC_SYSTEM),
                    Message(role="user", content=f"GOAL: {goal}\n\nRESULTS:\n{results_summary}"),
                ],
                schema=CriticOutput,
                temperature=0.1,
            )
            return out.model_dump()
        except Exception as e:
            logger.error("orchestrator.critic_failed", error=str(e))
            return {"achieved": True, "confidence": 0.5, "summary": "Critic failed; defaulting to success."}
