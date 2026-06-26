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

"""Slice discovery: automatically find underperforming subgroups."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SliceResult:
    description: str
    n: int
    metric: float
    overall: float
    gap: float


class SliceFinder:
    """Find worst-performing slices using decision tree leaves."""

    def find_slices(self, X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, max_slices: int = 20, min_size: int = 20) -> list[SliceResult]:
        from sklearn.tree import DecisionTreeClassifier
        correct = (y_pred == y_true).astype(int)
        overall = float(correct.mean())
        try:
            tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=min_size).fit(X, correct)
        except Exception:
            return []
        leaves = tree.apply(X)
        results: list[SliceResult] = []
        for leaf_id in np.unique(leaves):
            mask = leaves == leaf_id
            n = int(mask.sum())
            if n < min_size:
                continue
            metric = float(correct[mask].mean())
            path = self._path_to(tree, leaf_id, X.shape[1])
            results.append(SliceResult(
                description=path,
                n=n,
                metric=metric,
                overall=overall,
                gap=metric - overall,
            ))
        results.sort(key=lambda r: r.gap)
        return results[:max_slices]

    def _path_to(self, tree, leaf_id: int, n_features: int) -> str:
        from sklearn.tree import _tree
        t = tree.tree_
        node = leaf_id
        path = []
        while t.feature[node] != _tree.TREE_UNDEFINED:
            f = t.feature[node]
            threshold = t.threshold[node]
            path.append(f"f{f} {'<=' if f == 0 else '>'} {threshold:.3f}")
            node = t.children_left[node] if t.feature[node] != _tree.TREE_UNDEFINED else t.children_left[node]
        return " AND ".join(path[:5]) or "root"
