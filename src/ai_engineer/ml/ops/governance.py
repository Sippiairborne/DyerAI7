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

"""Model cards, governance, access control, audit log."""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ai_engineer.config import get_settings
from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelCard:
    name: str
    version: str
    description: str
    intended_use: str
    limitations: str
    ethical_considerations: str
    training_data: str
    metrics: dict[str, float]
    hyperparameters: dict[str, Any]
    authors: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    citation: str = ""
    license: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: list[str] = field(default_factory=list)


@dataclass
class AuditEvent:
    actor: str
    action: str
    resource: str
    timestamp: float
    metadata: dict[str, Any] = field(default_factory=dict)


class GovernanceManager:
    """Manage model cards, audit log, and access control."""

    def __init__(self, base_dir: str | None = None) -> None:
        s = get_settings()
        self.base = Path(base_dir or s.artifacts_dir) / "governance"
        self.base.mkdir(parents=True, exist_ok=True)
        self.cards: dict[str, ModelCard] = {}
        self.audit_log: list[AuditEvent] = []
        self._acl: dict[str, set[str]] = {}  # resource -> {roles}
        self._load()

    def _load(self) -> None:
        cards_path = self.base / "cards.json"
        audit_path = self.base / "audit.jsonl"
        acl_path = self.base / "acl.json"
        if cards_path.exists():
            for k, v in json.loads(cards_path.read_text()).items():
                self.cards[k] = ModelCard(**v)
        if audit_path.exists():
            with audit_path.open() as f:
                for line in f:
                    self.audit_log.append(AuditEvent(**json.loads(line)))
        if acl_path.exists():
            for k, v in json.loads(acl_path.read_text()).items():
                self._acl[k] = set(v)

    def _save_cards(self) -> None:
        (self.base / "cards.json").write_text(json.dumps({k: asdict(v) for k, v in self.cards.items()}, indent=2, default=str))

    def _save_audit(self) -> None:
        with (self.base / "audit.jsonl").open("a") as f:
            for e in self.audit_log:
                f.write(json.dumps(asdict(e)) + "\n")

    def create_card(self, card: ModelCard) -> None:
        self.cards[f"{card.name}:{card.version}"] = card
        self._save_cards()
        self.log("create_card", f"{card.name}:{card.version}")

    def get_card(self, name: str, version: str) -> ModelCard:
        return self.cards[f"{name}:{version}"]

    def render_card_markdown(self, card: ModelCard) -> str:
        lines = [
            f"# Model Card: {card.name} ({card.version})",
            f"**Created:** {card.created_at}",
            f"**Authors:** {', '.join(card.authors) or 'N/A'}",
            f"**License:** {card.license or 'N/A'}",
            "",
            "## Description",
            card.description,
            "",
            "## Intended Use",
            card.intended_use,
            "",
            "## Limitations",
            card.limitations,
            "",
            "## Ethical Considerations",
            card.ethical_considerations,
            "",
            "## Training Data",
            card.training_data,
            "",
            "## Metrics",
        ]
        for k, v in card.metrics.items():
            lines.append(f"- **{k}**: {v:.4f}")
        if card.hyperparameters:
            lines += ["", "## Hyperparameters", "```json", json.dumps(card.hyperparameters, indent=2, default=str), "```"]
        return "\n".join(lines)

    def log(self, action: str, resource: str, actor: str = "system", metadata: dict | None = None) -> None:
        ev = AuditEvent(actor=actor, action=action, resource=resource, timestamp=time.time(), metadata=metadata or {})
        self.audit_log.append(ev)
        self._save_audit()
        logger.info("governance.audit", actor=actor, action=action, resource=resource)

    def set_acl(self, resource: str, roles: list[str]) -> None:
        self._acl[resource] = set(roles)
        (self.base / "acl.json").write_text(json.dumps({k: list(v) for k, v in self._acl.items()}))

    def check_permission(self, resource: str, role: str) -> bool:
        return role in self._acl.get(resource, set()) or "admin" in self._acl.get(resource, set())
