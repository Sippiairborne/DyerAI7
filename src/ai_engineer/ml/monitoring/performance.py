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

"""Online performance tracking."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from statistics import mean, pstdev

from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceSnapshot:
    n_requests: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    error_rate: float
    throughput_rps: float
    window_s: float


class PerformanceMonitor:
    def __init__(self, window_size: int = 1000) -> None:
        self.latencies: dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.errors: dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.t






Continuing where I left off — finishing the monitoring submodule, the ops submodule, then wiring everything into the agents, tools, and API.

---

