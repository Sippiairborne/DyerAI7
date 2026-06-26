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

"""Background tasks for the CI loop."""
from __future__ import annotations

import asyncio

from ai_engineer.ci.workflows import CIWorkflow
from ai_engineer.core.llm import LLMClient
from ai_engineer.core.memory import MemorySystem
from ai_engineer.tools.sandbox import Sandbox
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


async def ci_loop(memory: MemorySystem, llm: LLMClient, sandbox: Sandbox, interval_minutes: int = 60) -> None:
    """Periodic CI loop: pattern extraction, fine-tuning."""
    workflow = CIWorkflow(memory, llm, sandbox)
    while True:
        try:
            result = await workflow.run_cycle()
            logger.info("ci.cycle_complete", **result)
        except Exception as e:
            logger.error("ci.cycle_failed", error=str(e))
        await asyncio.sleep(interval_minutes * 60)
