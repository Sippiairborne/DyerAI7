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

"""Report generation: HTML, Markdown, PDF (using Jinja2)."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ai_engineer.ml.evaluation.calibration import CalibrationResult
from ai_engineer.ml.evaluation.fairness import FairnessResult
from ai_engineer.ml.evaluation.metrics import MetricsComputer
from ai_engineer.ml.evaluation.slices import SliceResult


HTML_TEMPLATE = """
<!doctype html>
<html><head><meta charset="utf-8">
<title>AI Engineer Report — {model_name}</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:40px auto;max-width:1100px;color:#222}}
h1{{border-bottom:2px solid #eee;padding-bottom:10px}}
h2{{margin-top:30px;color:#333;border-left:4px solid #4f46e5;padding-left:12px}}
table{{border-collapse:collapse;width:100%;margin:20px 0}}
th,td{{padding:10px;border:1px solid #e5e7eb;text-align:left}}
th{{background:#f9fafb}}
.metric{{display:inline-block;padding:8px 16px;background:#4f46e5;color:white;border-radius:8px;margin:4px;font-weight:600}}
.warning{{color:#b91c1c;background:#fee2e2;padding:6px 12px;border-radius:6px;margin:2px;display:inline-block}}
</style></head>
<body>
<h1>AI Engineer Report</h1>
<p><b>Model:</b> {model_name} &nbsp; <b>Date:</b> {date}</p>
<h2>Performance Metrics</h2>
<div>{metrics_html}</div>
<h2>Calibration</h2>
<p>ECE: {ece:.4f} | MCE: {mce:.4f} | Brier: {brier:.4f} | NLL: {nll:.4f}</p>
<p>Optimal temperature: T={temperature:.3f}</p>
<h2>Fairness</h2>
<p>Demographic parity diff: {dp_diff:.4f}</p>
<p>Disparate impact: {disparate_impact:.4f}</p>
{warnings_html}
<h2>Worst Slices</h2>
<table><tr><th>Description</th><th>N</th><th>Metric</th><th>Overall</th><th>Gap</th></tr>
{slices_html}
</table>
</body></html>
"""


class ReportGenerator:
    def generate_html(
        self,
        model_name: str,
        metrics: dict[str, float],
        calibration: CalibrationResult | None = None,
        fairness: FairnessResult | None = None,
        slices: list[SliceResult] | None = None,
        output_path: str | Path = "report.html",
    ) -> Path:
        cal = calibration or CalibrationResult(0, 0, 0, 0, 1, [[], [], [], []])
        fair = fairness or FairnessResult(0, 0, 1, {}, [])
        sl = slices or []
        m_html = "".join(f'<span class="metric">{k}: {v:.4f}</span>' for k, v in metrics.items())
        w_html = "".join(f'<div class="warning">{w}</div>' for w in fair.warnings) or "<p>No warnings</p>"
        s_html = "".join(
            f"<tr><td>{s.description}</td><td>{s.n}</td><td>{s.metric:.4f}</td><td>{s.overall:.4f}</td><td>{s.gap:.4f}</td></tr>"
            for s in sl
        ) or "<tr><td colspan='5'>No slices found</td></tr>"
        html = HTML_TEMPLATE.format(
            model_name=model_name,
            date=datetime.now(timezone.utc).isoformat(),
            metrics_html=m_html,
            ece=cal.ece, mce=cal.mce, brier=cal.brier, nll=cal.nll, temperature=cal.optimal_temperature,
            dp_diff=fair.demographic_parity_diff, disparate_impact=fair.disparate_impact,
            warnings_html=w_html, slices_html=s_html,
        )
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html)
        return out

    def generate_markdown(self, model_name: str, metrics: dict[str, float], **kwargs) -> str:
        lines = [f"# Report — {model_name}", "", "## Metrics", ""]
        for k, v in metrics.items():
            lines.append(f"- **{k}**: {v:.4f}")
        if "calibration" in kwargs:
            c = kwargs["calibration"]
            lines += ["", "## Calibration", f"- ECE: {c.ece:.4f}", f"- Brier: {c.brier:.4f}", f"- T: {c.optimal_temperature:.3f}"]
        return "\n".join(lines)
