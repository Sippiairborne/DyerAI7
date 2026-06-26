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

"""Structured logging configuration."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from ai_engineer.config import get_settings


def _add_severity(_, __, event_dict: EventDict) -> EventDict:
    event_dict["severity"] = event_dict.get("level", "info").upper()
    return event_dict


def configure_logging() -> None:
    settings = get_settings()
    log_file = Path(settings.logs_dir) / "ai_engineer.log"

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        _add_severity,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=shared_processors
        + [structlog.dev.ConsoleRenderer(colors=False)],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)

    # Quiet noisy libs
    for name in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str | None = None, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name, **initial_values)
