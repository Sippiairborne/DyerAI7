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

"""Model Architect Agent."""
from __future__ import annotations

from typing import Any

from ai_engineer.agents.base import BaseAgent
from ai_engineer.core.llm import Message
from ai_engineer.utils.prompts import MODEL_ARCHITECT_SYSTEM


class ModelArchitectAgent(BaseAgent):
    name = "model_architect"
    system_prompt = MODEL_ARCHITECT_SYSTEM

    async def run(self, task: dict[str, Any]) -> dict[str, Any]:
        description = task.get("description", "")

        resp = await self.llm.complete(
            messages=[
                self.system_message(),
                Message(
                    role="user",
                    content=(
                        "Produce a complete, runnable training config and architecture justification. "
                        f"TASK: {description}"
                    ),
                ),
            ],
            temperature=0.2,
            max_tokens=6000,
        )
        return {"design": resp.content, "success": True}
