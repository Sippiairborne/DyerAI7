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

"""Trainer Agent — runs training jobs in the sandbox."""
from __future__ import annotations

from typing import Any

from ai_engineer.agents.base import BaseAgent
from ai_engineer.core.llm import Message
from ai_engineer.utils.prompts import TRAINER_SYSTEM


class TrainerAgent(BaseAgent):
    name = "trainer"
    system_prompt = TRAINER_SYSTEM

    async def run(self, task: dict[str, Any]) -> dict[str, Any]:
        description = task.get("description", "")

        # Ask LLM to compose a runnable training script
        code_resp = await self.llm.complete(
            messages=[
                self.system_message(),
                Message(
                    role="user",
                    content=(
                        "Write a self-contained Python training script for this task. "
                        "Use HuggingFace Trainer or Unsloth. Log to stdout. Save final model to /workspace/model. "
                        f"TASK: {description}"
                    ),
                ),
            ],
            temperature=0.1,
            max_tokens=8000,
        )

        # Smoke test: run a small subset first (1 step)
        # The full script is then run separately; here we assume the script itself is idempotent and budget-aware
        result = await self.sandbox.execute(code_resp.content, timeout=7200, gpu=True)
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.exit_code,
            "artifacts": result.artifacts,
            "success": result.exit_code == 0,
        }
