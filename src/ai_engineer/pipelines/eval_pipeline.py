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

"""Evaluation pipeline."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ai_engineer.tools.sandbox import Sandbox
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EvalResult:
    metrics: dict[str, float] = field(default_factory=dict)
    per_slice: dict[str, dict[str, float]] = field(default_factory=dict)
    failure_cases: list[dict[str, Any]] = field(default_factory=list)
    raw: str = ""


class EvalPipeline:
    def __init__(self, sandbox: Sandbox) -> None:
        self.sandbox = sandbox

    async def run(self, eval_script: str, output_dir: str, timeout: int = 3600) -> EvalResult:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        result = await self.sandbox.execute(eval_script, timeout=timeout)
        out_path = Path(output_dir) / "eval.json"
        parsed: dict[str, Any] = {}
        if out_path.exists():
            try:
                parsed = json.loads(out_path.read_text())
            except Exception:
                pass
        return EvalResult(
            metrics={k: float(v) for k, v in parsed.get("metrics", {}).items() if isinstance(v, (int, float))},
            per_slice=parsed.get("per_slice", {}),
            failure_cases=parsed.get("failure_cases", []),
            raw=result.stdout,
        )
