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

"""WebSocket for streaming task events to the UI."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ai_engineer.api.deps import get_state

router = APIRouter()


@router.websocket("/ws/tasks/{task_id}")
async def task_stream(websocket: WebSocket, task_id: str) -> None:
    await websocket.accept()
    state = get_state()
    orchestrator = state.orchestrator
    if not orchestrator:
        await websocket.close(code=1011)
        return

    queue: asyncio.Queue = asyncio.Queue()
    state.tasks.setdefault(task_id, {"events": []})

    async def on_event(event: dict) -> None:
        await queue.put(event)

    orchestrator.on_event(on_event)
    try:
        # Send existing events
        for e in state.tasks[task_id].get("events", []):
            await websocket.send_text(json.dumps(e))
        # Stream new
        last_idx = len(state.tasks[task_id].get("events", []))
        while True:
            events = state.tasks[task_id].get("events", [])
            while last_idx < len(events):
                await websocket.send_text(json.dumps(events[last_idx]))
                last_idx += 1
            t = state.tasks.get(task_id, {})
            if t.get("status") in ("done", "failed") and last_idx >= len(events):
                await websocket.send_text(json.dumps({"type": "stream_end", "status": t["status"]}))
                break
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        pass
