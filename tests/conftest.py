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

"""Pytest fixtures."""
from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

os.environ.setdefault("LLM_BASE_URL", "http://localhost:8000/v1")
os.environ.setdefault("SANDBOX_BACKEND", "local")

from ai_engineer.core.llm import LLMClient  # noqa: E402
from ai_engineer.core.memory import MemorySystem  # noqa: E402
from ai_engineer.tools.registry import ToolRegistry  # noqa: E402
from ai_engineer.tools.sandbox import Sandbox  # noqa: E402


@pytest.fixture(scope="session")
def event_loop() -> AsyncIterator[asyncio.AbstractEventLoop]:
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def llm() -> AsyncIterator[LLMClient]:
    c = LLMClient()
    yield c
    await c.close()


@pytest_asyncio.fixture
async def sandbox() -> AsyncIterator[Sandbox]:
    s = Sandbox(backend="local")
    yield s


@pytest.fixture
def tools() -> ToolRegistry:
    return ToolRegistry()
