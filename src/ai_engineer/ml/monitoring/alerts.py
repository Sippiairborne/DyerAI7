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

"""Alerting rules and notification."""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Literal

from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Channel = Literal["log", "webhook", "email", "slack", "pagerduty"]


@dataclass
class AlertRule:
    name: str
    metric: str  # latency_p95 | error_rate | drift_score | fairness_gap | calibration_ece
    comparator: str  # gt | lt | gte | lte
    threshold: float
    window_minutes: int = 5
    severity: Literal["info", "warning", "critical"] = "warning"
    channels: list[Channel] = field(default_factory=lambda: ["log"])
    cooldown_minutes: int = 30
    enabled: bool = True


@dataclass
class AlertEvent:
    rule: str
    severity: str
    message: str
    metric_value: float
    threshold: float
    timestamp: float
    metadata: dict = field(default_factory=dict)


class AlertManager:
    def __init__(self) -> None:
        self.rules: dict[str, AlertRule] = {}
        self._last_fired: dict[str, float] = {}
        self.events: list[AlertEvent] = []

    def add_rule(self, rule: AlertRule) -> None:
        self.rules[rule.name] = rule

    def evaluate(self, metrics: dict[str, float]) -> list[AlertEvent]:
        fired: list[AlertEvent] = []
        now = time.time()
        for name, rule in self.rules.items():
            if not rule.enabled or name not in metrics:
                continue
            v = metrics[name]
            triggered = self._check(v, rule.comparator, rule.threshold)
            if not triggered:
                continue
            last = self._last_fired.get(name, 0)
            if now - last < rule.cooldown_minutes * 60:
                continue
            self._last_fired[name] = now
            event = AlertEvent(
                rule=name,
                severity=rule.severity,
                message=f"{name} {rule.comparator} {rule.threshold} (actual {v:.4f})",
                metric_value=float(v),
                threshold=rule.threshold,
                timestamp=now,
                metadata={"window_minutes": rule.window_minutes},
            )
            self.events.append(event)
            self._dispatch(event, rule)
            fired.append(event)
        return fired

    def _check(self, value: float, op: str, threshold: float) -> bool:
        return {
            "gt": value > threshold,
            "lt": value < threshold,
            "gte": value >= threshold,
            "lte": value <= threshold,
        }[op]

    def _dispatch(self, event: AlertEvent, rule: AlertRule) -> None:
        for ch in rule.channels:
            if ch == "log":
                getattr(logger, "warning" if event.severity != "critical" else "error")(
                    "alert.fired", rule=event.rule, severity=event.severity, value=event.metric_value, threshold=event.threshold
                )
            elif ch == "webhook":
                self._send_webhook(event)
            elif ch == "slack":
                self._send_slack(event)
            elif ch == "email":
                self._send_email(event)
            elif ch == "pagerduty":
                self._send_pagerduty(event)

    def _send_webhook(self, event: AlertEvent) -> None:
        import httpx
        # Default webhook from env
        import os
        url = os.environ.get("ALERT_WEBHOOK_URL")
        if not url:
            return
        try:
            httpx.post(url, json=asdict(event), timeout=5)
        except Exception as e:
            logger.warning("alert.webhook_failed", error=str(e))

    def _send_slack(self, event: AlertEvent) -> None:
        import httpx
        import os
        url = os.environ.get("SLACK_WEBHOOK_URL")
        if not url:
            return
        color = {"info": "#36a64f", "warning": "#daa038", "critical": "#d00000"}[event.severity]
        payload = {
            "attachments": [{
                "color": color,
                "title": f"AI Engineer Alert: {event.rule}",
                "text": event.message,
                "fields": [
                    {"title": "Severity", "value": event.severity, "short": True},
                    {"title": "Value", "value": f"{event.metric_value:.4f}", "short": True},
                ],
            }]
        }
        try:
            httpx.post(url, json=payload, timeout=5)
        except Exception as e:
            logger.warning("alert.slack_failed", error=str(e))

    def _send_email(self, event: AlertEvent) -> None:
        import os
        import smtplib
        from email.mime.text import MIMEText

        host = os.environ.get("SMTP_HOST")
        if not host:
            return
        msg = MIMEText(json.dumps(asdict(event), indent=2))
        msg["Subject"] = f"[AI Engineer] {event.severity}: {event.rule}"
        msg["From"] = os.environ.get("SMTP_FROM", "ai-engineer@localhost")
        msg["To"] = os.environ.get("ALERT_EMAIL_TO", "")
        try:
            with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", 587))) as s:
                s.starttls()
                s.login(os.environ.get("SMTP_USER", ""), os.environ.get("SMTP_PASS", ""))
                s.send_message(msg)
        except Exception as e:
            logger.warning("alert.email_failed", error=str(e))

    def _send_pagerduty(self, event: AlertEvent) -> None:
        import httpx
        import os
        key = os.environ.get("PAGERDUTY_KEY")
        if not key:
            return
        try:
            httpx.post(
                "https://events.pagerduty.com/v2/enqueue",
                json={
                    "routing_key": key,
                    "event_action": "trigger",
                    "payload": {
                        "summary": event.message,
                        "severity": event.severity,
                        "source": "ai-engineer",
                    },
                },
                timeout=5,
            )
        except Exception as e:
            logger.warning("alert.pagerduty_failed", error=str(e))
