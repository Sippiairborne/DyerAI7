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

"""Shared blackboard memory for multi-agent systems."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BlackboardEntry:
    key: str
    value: Any
    written_by: str
    timestamp: float = field(default_factory=time.time)
    read_count: int = 0


class SharedBlackboard:
    """Blackboard pattern: shared memory across agents."""

    def __init__(self, namespace: str = "default") -> None:
        self.namespace = namespace
        self.entries: dict[str, BlackboardEntry] = {}

    def write(self, key: str, value: Any, writer: str = "system") -> None:
        self.entries[key] = BlackboardEntry(key=key, value=value, written_by=writer)

    def read(self, key: str) -> Any:
        if key in self.entries:
            self.entries[key].read_count += 1
            return self.entries[key].value
        return None

    def get(self, key: str, default: Any = None) -> Any:
        return self.read(key) if key in self.entries else default

    def has(self, key: str) -> bool:
        return key in self.entries

    def snapshot(self) -> str:
        lines = []
        for e in self.entries.values():
            lines.append(f"[{e.written_by}] {e.key}: {str(e.value)[:200]}")
        return "\n".join(lines)

    def keys(self) -> list[str]:
        return list(self.entries.keys())
