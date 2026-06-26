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

"""Task routes — submit and track AI engineering jobs."""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ai_engineer.api.deps import get_state
from ai_engineer.api.schemas import TaskRequest, TaskResponse

router = APIRouter(prefix="/api/tasks", tags=["tasks"])


@router.post("", response_model=TaskResponse)
async def submit_task(req: TaskRequest, background: BackgroundTasks) -> TaskResponse:
    state = get_state()
    if not state.orchestrator:
        raise HTTPException(503, "Orchestrator not initialized")
    task_id = str(uuid.uuid4())
    state.tasks[task_id] = {
        "id": task_id,
        "goal": req.goal,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "finished_at": None,
        "result": None,
        "error": None,
    }

    async def run() -> None:
        state.tasks[task_id]["status"] = "running"
        state.tasks[task_id]["started_at"] = datetime.now(timezone.utc).isoformat()
        # Subscribe to events and persist
        async def on_event(event: dict) -> None:
            state.tasks[task_id].setdefault("events", []).append(event)

        state.orchestrator.on_event(on_event)
        try:
            result = await state.orchestrator.run_goal(
                req.goal, max_replans=req.max_replans, max_task_retries=req.max_task_retries
            )
            state.tasks[task_id]["result"] = result
            state.tasks[task_id]["status"] = "done" if result.get("success") else "failed"
        except Exception as e:
            state.tasks[task_id]["error"] = str(e)
            state.tasks[task_id]["status"] = "failed"
        finally:
            state.tasks[task_id]["finished_at"] = datetime.now(timezone.utc).isoformat()

    background.add_task(run)
    return TaskResponse(**state.tasks[task_id])


@router.get("", response_model=list[TaskResponse])
async def list_tasks() -> list[TaskResponse]:
    state = get_state()
    return [TaskResponse(**t) for t in state.tasks.values()]


@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str) -> TaskResponse:
    state = get_state()
    t = state.tasks.get(task_id)
    if not t:
        raise HTTPException(404, "Task not found")
    return TaskResponse(**t)


@router.get("/{task_id}/events")
async def get_task_events(task_id: str) -> list[dict]:
    state = get_state()
    t = state.tasks.get(task_id)
    if not t:
        raise HTTPException(404, "Task not found")
    return t.get("events", [])
