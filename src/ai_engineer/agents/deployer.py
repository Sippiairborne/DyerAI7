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

"""Deployer Agent."""
from __future__ import annotations

from typing import Any

from ai_engineer.agents.base import BaseAgent
from ai_engineer.core.llm import Message
from ai_engineer.utils.prompts import DEPLOYER_SYSTEM


class DeployerAgent(BaseAgent):
    name = "deployer"
    system_prompt = DEPLOYER_SYSTEM

    async def run(self, task: dict[str, Any]) -> dict[str, Any]:
        description = task.get("description", "")

        code_resp = await self.llm.complete(
            messages=[
                self.system_message(),
                Message(
                    role="user",
                    content=(
                        "Produce a serving setup: Dockerfile, model card, and run commands. "
                        "Use vLLM or TGI. Save all files to /workspace/deploy/.\n\n"
                        f"TASK: {description}"
                    ),
                ),
            ],
            temperature=0.1,
            max_tokens=6000,
        )
        result = await self.sandbox.execute(code_resp.content, timeout=600)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "artifacts": result.artifacts,
            "success": result.exit_code == 0,
        }
