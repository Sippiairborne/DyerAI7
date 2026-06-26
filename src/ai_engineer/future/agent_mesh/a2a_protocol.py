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

"""A2A (Agent-to-Agent) protocol for inter-agent communication."""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Literal

Kind = Literal := __import__("typing").Literal["request", "response", "broadcast", "task_complete", "error", "consensus"]  # type: ignore


@dataclass
class A2AMessage:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = "*"
    kind: str = "request"
    payload: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    correlation_id: str | None = None


class A2AProtocol:
    """Asynchronous message bus for agents."""

    def __init__(self) -> None:
        self.subscribers: dict[str, list] = {}  # recipient -> [callbacks]
        self.history: list[A2AMessage] = []

    def subscribe(self, recipient: str, callback) -> None:
        self.subscribers.setdefault(recipient, []).append(callback)

    async def send(self, msg: A2AMessage) -> None:
        self.history.append(msg)
        if msg.recipient == "*":
            targets = [cb for cbs in self.subscribers.values() for cb in cbs]
        else:
            targets = self.subscribers.get(msg.recipient, [])
        for cb in targets:
            try:
                r = cb(msg)
                if asyncio.iscoroutine(r):
                    await r
            except Exception:
                pass

    async def broadcast(self, msg: A2AMessage) -> None:
        msg.recipient = "*"
        await self.send(msg)

    async def request(self, sender: str, recipient: str, payload: dict, timeout: float = 30.0) -> A2AMessage | None:
        req = A2AMessage(sender=sender, recipient=recipient, kind="request", payload=payload)
        response_future: asyncio.Future = asyncio.Future()

        def waiter(msg: A2AMessage) -> None:
            if msg.correlation_id == req.id and not response_future.done():
                response_future.set_result(msg)

        self.subscribe(recipient, waiter)
        await self.send(req)
        try:
            return await asyncio.wait_for(response_future, timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def consensus(self, proposals: dict[str, Any]) -> Any:
        """Simple majority vote consensus across agent proposals."""
        from collections import Counter
        return Counter(str(v) for v in proposals.values()).most_common(1)[0][0]
