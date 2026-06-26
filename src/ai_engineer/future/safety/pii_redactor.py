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

"""PII detection and redaction."""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PIIMatch:
    kind: str
    value: str
    start: int
    end: int


class PIIRedactor:
    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "phone": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "ipv6": r"\b(?:[A-F0-9]{1,4}:){7}[A-F0-9]{1,4}\b",
        "date_of_birth": r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b",
        "url": r"https?://[^\s]+",
        "address": r"\b\d+\s+[A-Z][a-z]+\s+(Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b",
        "name": r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b",
    }

    def detect(self, text: str) -> list[PIIMatch]:
        matches: list[PIIMatch] = []
        for kind, pattern in self.PATTERNS.items():
            for m in re.finditer(pattern, text):
                matches.append(PIIMatch(kind=kind, value=m.group(), start=m.start(), end=m.end()))
        matches.sort(key=lambda x: x.start)
        return matches

    def redact(self, text: str, replacement: str = "[REDACTED]") -> str:
        for kind, pattern in self.PATTERNS.items():
            text = re.sub(pattern, f"{replacement}_{kind.upper()}", text)
        return text

    def detect_and_classify(self, text: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for kind, pattern in self.PATTERNS.items():
            counts[kind] = len(re.findall(pattern, text))
        return counts
