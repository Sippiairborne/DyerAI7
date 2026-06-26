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

"""Shared FastAPI dependencies."""
from __future__ import annotations

from functools import lru_cache

from ai_engineer.core.llm import LLMClient
from ai_engineer.core.memory import MemorySystem
from ai_engineer.core.orchestrator import Orchestrator
from ai_engineer.tools.registry import ToolRegistry
from ai_engineer.tools.sandbox import Sandbox
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


class AppState:
    llm: LLMClient | None = None
    memory: MemorySystem | None = None
    tools: ToolRegistry | None = None
    sandbox: Sandbox | None = None
    orchestrator: Orchestrator | None = None
    tasks: dict[str, dict] = {}


@lru_cache(maxsize=1)
def get_state() -> AppState:
    return AppState()
