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

"""Data and model lineage tracker."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ai_engineer.config import get_settings
from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LineageNode:
    id: str
    kind: str  # dataset | model | run | feature | deployment
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class LineageEdge:
    source: str
    target: str
    relation: str  # derived_from | trained_on | deployed_to | produced
    metadata: dict[str, Any] = field(default_factory=dict)


class LineageTracker:
    """Tracks relationships between datasets, models, runs, and deployments."""

    def __init__(self, base_dir: str | None = None) -> None:
        s = get_settings()
        self.base = Path(base_dir or s.artifacts_dir) / "lineage"
        self.base.mkdir(parents=True, exist_ok=True)
        self._nodes: dict[str, LineageNode] = {}
        self._edges: list[LineageEdge] = []
        self._load()

    def _load(self) -> None:
        nodes_path = self.base / "nodes.json"
        edges_path = self.base / "edges.json"
        if nodes_path.exists():
            for n in json.loads(nodes_path.read_text()):
                self._nodes[n["id"]] = LineageNode(**n)
        if edges_path.exists():
            for e in json.loads(edges_path.read_text()):
                self._edges.append(LineageEdge(**e))

    def _save(self) -> None:
        (self.base / "nodes.json").write_text(json.dumps([asdict(n) for n in self._nodes.values()], indent=2, default=str))
        (self.base / "edges.json").write_text(json.dumps([asdict(e) for e in self._edges], indent=2, default=str))

    def add_node(self, kind: str, name: str, metadata: dict | None = None, node_id: str | None = None) -> str:
        nid = node_id or f"{kind}_{uuid.uuid4().hex[:8]}"
        self._nodes[nid] = LineageNode(id=nid, kind=kind, name=name, metadata=metadata or {})
        self._save()
        return nid

    def add_edge(self, source: str, target: str, relation: str, metadata: dict | None = None) -> None:
        if source not in self._nodes or target not in self._nodes:
            raise AIEngineerError(f"Unknown node: {source} or {target}")
        self._edges.append(LineageEdge(source=source, target=target, relation=relation, metadata=metadata or {}))
        self._save()

    def link_dataset_to_model(self, dataset_id: str, model_id: str, run_id: str | None = None) -> None:
        self.add_edge(dataset_id, model_id, "trained_on", {"run_id": run_id} if run_id else {})

    def link_model_to_deployment(self, model_id: str, deployment_id: str, stage: str) -> None:
        self.add_edge(model_id, deployment_id, "deployed_to", {"stage": stage})

    def link_features(self, feature_id: str, dataset_id: str) -> None:
        self.add_edge(feature_id, dataset_id, "produced")

    def ancestors(self, node_id: str) -> list[LineageNode]:
        """All upstream nodes."""
        visited: set[str] = set()
        result: list[LineageNode] = []
        stack = [node_id]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for e in self._edges:
                if e.target == cur and e.source not in visited:
                    stack.append(e.source)
                    if e.source in self._nodes:
                        result.append(self._nodes[e.source])
        return result

    def descendants(self, node_id: str) -> list[LineageNode]:
        """All downstream nodes."""
        visited: set[str] = set()
        result: list[LineageNode] = []
        stack = [node_id]
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for e in self._edges:
                if e.source == cur and e.target not in visited:
                    stack.append(e.target)
                    if e.target in self._nodes:
                        result.append(self._nodes[e.target])
        return result

    def impact_analysis(self, node_id: str) -> dict:
        descendants = self.descendants(node_id)
        deployments = [n for n in descendants if n.kind == "deployment"]
        models = [n for n in descendants if n.kind == "model"]
        return {
            "node": self._nodes[node_id].__dict__,
            "downstream_count": len(descendants),
            "affected_models": [m.name for m in models],
            "affected_deployments": [d.name for d in deployments],
        }

    def export_dot(self) -> str:
        """Export lineage as Graphviz DOT format."""
        lines = ["digraph lineage { rankdir=LR;"]
        for n in self._nodes.values():
            color = {"dataset": "lightblue", "model": "lightgreen", "run": "yellow", "feature": "orange", "deployment": "lightcoral"}.get(n.kind, "white")
            lines.append(f'  "{n.id}" [label="{n.name}\\n({n.kind})", style=filled, fillcolor={color}];')
        for e in self._edges:
            lines.append(f'  "{e.source}" -> "{e.target}" [label="{e.relation}"];')
        lines.append("}")
        return "\n".join(lines)
