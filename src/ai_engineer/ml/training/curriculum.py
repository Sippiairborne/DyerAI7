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

"""Curriculum learning: easy-to-hard sample scheduling."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class CurriculumSampler(Sampler):
    """Sample easy examples first, then hard ones as training progresses."""

    def __init__(self, difficulty: np.ndarray, num_samples: int, pacing: str = "linear") -> None:
        self.difficulty = np.asarray(difficulty)
        self.num_samples = num_samples
        self.pacing = pacing

    def __iter__(self):
        n = len(self.difficulty)
        order = np.argsort(self.difficulty)
        if self.pacing == "linear":
            frac = 1.0
        else:
            frac = 0.5
        cut = max(int(n * frac), 1)
        current = order[:cut]
        rest = order[cut:]
        np.random.shuffle(current)
        for i in current:
            yield int(i)
        np.random.shuffle(rest)
        for i in rest:
            yield int(i)

    def __len__(self) -> int:
        return self.num_samples
