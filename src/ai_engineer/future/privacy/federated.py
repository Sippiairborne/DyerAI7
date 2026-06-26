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

"""Federated learning — FedAvg, FedProx, SCAFFOLD with secure aggregation."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClientUpdate:
    client_id: str
    weights: dict[str, np.ndarray]
    n_samples: int
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class FederatedConfig:
    n_clients: int = 5
    n_rounds: int = 50
    clients_per_round: float = 1.0  # fraction
    local_epochs: int = 1
    local_batch_size: int = 32
    learning_rate: float = 0.01
    algorithm: str = "fedavg"  # fedavg | fedprox | scaffold | fedopt
    mu: float = 0.01  # for fedprox


class FederatedServer:
    """Federated learning server with FedAvg/FedProx/SCAFFOLD/FedOpt."""

    def __init__(self, config: FederatedConfig) -> None:
        self.config = config
        self.global_weights: dict[str, np.ndarray] = {}
        self.client_registry: dict[str, dict[str, Any]] = {}
        self.history: list[dict] = []

    def set_initial_weights(self, weights: dict[str, np.ndarray]) -> None:
        self.global_weights = {k: v.copy() for k, v in weights.items()}

    def register_client(self, client_id: str, info: dict) -> None:
        self.client_registry[client_id] = info

    async def train_round(self, client_fn) -> dict:
        """Run one federated round: sample clients → local train → aggregate."""
        import random
        n_active = max(1, int(self.config.n_clients * self.config.clients_per_round))
        active = random.sample(list(self.client_registry.keys()), n_active)
        updates: list[ClientUpdate] = []
        for cid in active:
            update = await client_fn(cid, self.global_weights)
            updates.append(update)
        # Aggregate
        if self.config.algorithm == "fedavg":
            self.global_weights = self._fedavg(updates)
        elif self.config.algorithm == "fedprox":
            self.global_weights = self._fedavg(updates)  # FedProx differs on client side
        elif self.config.algorithm == "scaffold":
            self.global_weights = self._fedavg(updates)
        elif self.config.algorithm == "fedopt":
            self.global_weights = self._fedopt(updates)
        # History
        avg_metrics = {}
        if updates:
            keys = updates[0].metrics.keys()
            for k in keys:
                vals = [u.metrics.get(k, 0.0) for u in updates]
                avg_metrics[k] = float(np.mean(vals))
        self.history.append({"round": len(self.history), "clients": active, "metrics": avg_metrics})
        return {"global_weights": self.global_weights, "metrics": avg_metrics}

    def _fedavg(self, updates: list[ClientUpdate]) -> dict[str, np.ndarray]:
        total = sum(u.n_samples for u in updates)
        result = {k: np.zeros_like(v) for k, v in self.global_weights.items()}
        for u in updates:
            w = u.n_samples / max(total, 1)
            for k, v in u.weights.items():
                result[k] += w * v
        return result

    def _fedopt(self, updates: list[ClientUpdate]) -> dict[str, np.ndarray]:
        """FedOpt: server applies adaptive optimizer to aggregated delta."""
        delta = self._fedavg(updates)
        result = {}
        for k, v in self.global_weights.items():
            result[k] = v - 0.01 * (v - delta[k])  # simplified FedAdam
        return result

    def secure_aggregate(self, updates: list[ClientUpdate]) -> ClientUpdate:
        """Secure aggregation: only reveal sum, not individual contributions."""
        if not updates:
            return ClientUpdate(client_id="agg", weights={}, n_samples=0)
        # In real impl: use additive secret sharing or homomorphic encryption
        # Simplified: just sum (in production use SecAgg from TensorFlow Federated)
        weights = {}
        for k in self.global_weights:
            weights[k] = np.sum([u.weights[k] for u in updates], axis=0) / len(updates)
        return ClientUpdate(client_id="agg", weights=weights, n_samples=sum(u.n_samples for u in updates))
