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

"""Data Engineer Agent."""
from __future__ import annotations

from typing import Any

from ai_engineer.agents.base import BaseAgent
from ai_engineer.core.llm import Message
from ai_engineer.tools.sandbox import Sandbox
from ai_engineer.utils.logging import get_logger
from ai_engineer.utils.prompts import DATA_ENGINEER_SYSTEM

logger = get_logger(__name__)


class DataEngineerAgent(BaseAgent):
    name = "data_engineer"
    system_prompt = DATA_ENGINEER_SYSTEM

    async def run(self, task: dict[str, Any]) -> dict[str, Any]:
        description = task.get("description", "")
        criteria = task.get("acceptance_criteria", [])

        # Plan
        plan_resp = await self.llm.complete(
            messages=[
                self.system_message(),
                Message(role="user", content=f"TASK: {description}\n\nCRITERIA: {criteria}"),
            ],
            temperature=0.2,
        )
        plan_text = plan_resp.content

        # Generate and execute code
        code_resp = await self.llm.complete(
            messages=[
                self.system_message(),
                Message(
                    role="user",
                    content=(
                        "Write a single self-contained Python script that fulfills this task. "
                        "Print clear progress messages and final summary. Save datasets to /workspace/datasets/.\n\n"
                        f"TASK: {description}\n\nPLAN: {plan_text}"
                    ),
                ),
            ],
            temperature=0.1,
            max_tokens=4096,
        )

        result = await self.sandbox.execute(code_resp.content, timeout=1200)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "artifacts": result.artifacts,
            "plan": plan_text,
            "success": result.exit_code == 0,
        }
