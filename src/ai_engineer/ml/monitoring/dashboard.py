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

"""Monitoring dashboard (Grafana JSON + Streamlit page)."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ai_engineer.ml.monitoring.drift import DriftResult
from ai_engineer.ml.monitoring.performance import PerformanceMonitor, PerformanceSnapshot


def build_grafana_dashboard(
    title: str = "AI Engineer Monitoring",
    panels: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a Grafana dashboard JSON."""
    return {
        "title": title,
        "schemaVersion": 38,
        "version": 1,
        "refresh": "30s",
        "panels": panels or [
            {
                "type": "timeseries",
                "title": "Latency p95",
                "targets": [{"expr": 'ai_engineer_latency_ms{quantile="0.95"}'}],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
            },
            {
                "type": "stat",
                "title": "Error Rate",
                "targets": [{"expr": "ai_engineer_error_rate"}],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
            },
            {
                "type": "timeseries",
                "title": "Throughput (rps)",
                "targets": [{"expr": "ai_engineer_throughput_rps"}],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
            },
            {
                "type": "stat",
                "title": "Total Requests",
                "targets": [{"expr": "ai_engineer_requests_total"}],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
            },
        ],
    }


def save_dashboard(dashboard: dict[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dashboard, indent=2))
    return out


def render_streamlit_dashboard(monitor: PerformanceMonitor, model_name: str) -> None:
    """Streamlit rendering helper — call from a streamlit script."""
    try:
        import streamlit as st
        import plotly.graph_objects as go
    except ImportError:
        return
    s = monitor.snapshot(model_name)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("p50 (ms)", f"{s.latency_p50_ms:.1f}")
    c2.metric("p95 (ms)", f"{s.latency_p95_ms:.1f}")
    c3.metric("Error rate", f"{s.error_rate * 100:.2f}%")
    c4.metric("Throughput", f"{s.throughput_rps:.1f} rps")
    fig = go.Figure()
    lats = sorted(monitor.latencies[model_name])
    if lats:
        fig.add_trace(go.Scatter(y=lats, mode="lines", name="latency"))
        st.plotly_chart(fig)
