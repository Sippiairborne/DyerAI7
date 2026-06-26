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

"""Stores full task trajectories for replay and fine-tuning."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ai_engineer.config import get_settings
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrajectoryStep:
    agent: str
    action: str
    observation: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Trajectory:
    id: str
    goal: str
    plan: dict[str, Any]
    steps: list[TrajectoryStep] = field(default_factory=list)
    success: bool = False
    final_metrics: dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_training_example(self) -> dict[str, str]:
        """Convert to {prompt, response} for SFT."""
        prompt = f"GOAL: {self.goal}\n\nPLAN:\n{json.dumps(self.plan, indent=2)}"
        response_lines = []
        for s in self.steps:
            response_lines.append(f"## {s.agent} → {s.action}\n{s.observation}\n")
        response = "\n".join(response_lines)
        if self.final_metrics:
            response += f"\n\nFINAL METRICS:\n{json.dumps(self.final_metrics, indent=2)}"
        return {"prompt": prompt, "response": response}


class TrajectoryStore:
    """Persists full run trajectories for analysis and self-improvement."""

    def __init__(self) -> None:
        s = get_settings()
        self.base = Path(s.artifacts_dir) / "trajectories"
        self.base.mkdir(parents=True, exist_ok=True)
        self._index_path = self.base / "_index.jsonl"
        self._cache: dict[str, Trajectory] = {}

    def create(self, goal: str, plan: dict[str, Any]) -> Trajectory:
        t = Trajectory(id=str(uuid.uuid4()), goal=goal, plan=plan)
        self._cache[t.id] = t
        return t

    def add_step(self, trajectory_id: str, agent: str, action: str, observation: str) -> None:
        if trajectory_id in self._cache:
            self._cache[trajectory_id].steps.append(
                TrajectoryStep(agent=agent, action=action, observation=observation)
            )

    async def finalize(self, trajectory_id: str, success: bool, metrics: dict[str, Any]) -> None:
        t = self._cache.get(trajectory_id)
        if not t:
            return
        t.success = success
        t.final_metrics = metrics
        t.finished_at = time.time()
        path = self.base / f"{trajectory_id}.json"
        path.write_text(json.dumps(asdict(t), indent=2))
        with self._index_path.open("a") as f:
            f.write(json.dumps({"id": t.id, "goal": t.goal, "success": t.success}) + "\n")
        logger.info("trajectory.saved", id=t.id, success=success)

    async def successful(self, limit: int = 1000) -> list[Trajectory]:
        results: list[Trajectory] = []
        if not self._index_path.exists():
            return results
        with self._index_path.open() as f:
            for line in f:
                rec = json.loads(line)
                if not rec.get("success"):
                    continue
                p = self.base / f"{rec['id']}.json"
                if p.exists():
                    results.append(Trajectory(**json.loads(p.read_text())))
                if len(results) >= limit:
                    break
        return results

    async def export_training_data(self, output_path: Path) -> int:
        """Export successful trajectories as JSONL training data."""
        trajs = await self.successful()
        count = 0
        with Path(output_path).open("w") as f:
            for t in trajs:
                ex = t.to_training_example()
                f.write(json.dumps(ex) + "\n")
                count += 1
        logger.info("trajectory.exported", count=count, path=str(output_path))
        return count
