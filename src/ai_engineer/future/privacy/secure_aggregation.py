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

"""Secure aggregation with additive secret sharing."""
from __future__ import annotations

import secrets

import numpy as np


class SecureAggregator:
    """Simplified additive secret sharing for federated averaging."""

    def __init__(self, n_shares: int = 3, modulus: int = 2**31 - 1) -> None:
        self.n_shares = n_shares
        self.modulus = modulus

    def split(self, value: np.ndarray) -> list[np.ndarray]:
        shares = [np.random.randint(0, self.modulus, size=value.shape, dtype=np.int64) for _ in range(self.n_shares - 1)]
        last = (value.astype(np.int64) - sum(shares)) % self.modulus
        shares.append(last)
        return shares

    def combine(self, shares: list[np.ndarray]) -> np.ndarray:
        return (sum(shares) % self.modulus).astype(np.float64) / self.modulus

    def aggregate(self, updates: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Aggregate model updates with secure sharing."""
        keys = updates[0].keys()
        result: dict[str, np.ndarray] = {}
        for k in keys:
            shares_list = [self.split(u[k]) for u in updates]
            summed_shares = [sum(s[i] for s in shares_list) for i in range(self.n_shares)]
            result[k] = self.combine(summed_shares) / len(updates)
        return result
