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

"""Evaluator Agent."""
from __future__ import annotations

from typing import Any

from ai_engineer.agents.base import BaseAgent
from ai_engineer.core.llm import Message
from ai_engineer.utils.prompts import EVALUATOR_SYSTEM


class EvaluatorAgent(BaseAgent):
    name = "evaluator"
    system_prompt = EVALUATOR_SYSTEM

    async def run(self, task: dict[str, Any]) -> dict[str, Any]:
        description = task.get("description", "")

        code_resp = await self.llm.complete(
            messages=[
                self.system_message(),
                Message(
                    role="user",
                    content=(
                        "Write a self-contained evaluation script. Compute primary metric, "
                        "per-slice metrics, and confidence intervals. Print a JSON report to stdout. "
                        f"TASK: {description}"
                    ),
                ),
            ],
            temperature=0.1,
            max_tokens=6000,
        )
        result = await self.sandbox.execute(code_resp.content, timeout=1800)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "success": result.exit_code == 0,
        }
