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

"""Canary deployment — gradual traffic shift."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CanaryConfig:
    new_model_url: str
    old_model_url: str
    proxy_port: int = 9000
    initial_traffic: float = 0.05
    ramp_steps: int = 10
    step_duration_minutes: int = 30
    success_metric: str = "latency_p95"
    success_threshold: float = 1.2  # new must be ≤ 1.2x old
    rollback_on_failure: bool = True


@dataclass
class CanaryStatus:
    step: int
    current_traffic: float
    success_metric_new: float
    success_metric_old: float
    ratio: float
    is_healthy: bool
    message: str


class CanaryDeployer:
    def __init__(self) -> None:
        self.status: list[CanaryStatus] = []

    def ramp(self, config: CanaryConfig) -> list[CanaryStatus]:
        from ai_engineer.ml.monitoring.performance import PerformanceMonitor
        for step in range(config.ramp_steps):
            traffic = config.initial_traffic + (1 - config.initial_traffic) * (step / max(config.ramp_steps - 1, 1))
            time.sleep(config.step_duration_minutes * 60)
            # Sample metrics — in real impl this would query the proxy
            ratio = 1.0  # placeholder
            healthy = ratio <= config.success_threshold
            status = CanaryStatus(step=step, current_traffic=traffic, success_metric_new=100, success_metric_old=100, ratio=ratio, is_healthy=healthy, message="OK" if healthy else "Regression")
            self.status.append(status)
            if not healthy and config.rollback_on_failure:
                status.message = "ROLLBACK"
                logger.warning("canary.rollback", step=step, ratio=ratio)
                break
        return self.status
