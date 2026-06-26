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

"""ML CI/CD pipelines — automated testing, validation, retraining, redeployment."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ai_engineer.config import get_settings
from ai_engineer.ml.evaluation.metrics import MetricsComputer
from ai_engineer.ml.models.registry import ModelRegistry
from ai_engineer.ml.monitoring.drift import DriftDetector
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineStage:
    name: str
    fn: Callable
    required: bool = True
    retries: int = 1
    on_failure: str = "fail"  # fail | warn | skip


@dataclass
class PipelineResult:
    success: bool
    stages: list[dict[str, Any]]
    started_at: float
    finished_at: float
    triggered_by: str

    def to_markdown(self) -> str:
        lines = [f"# ML Pipeline Run", f"**Success:** {self.success}", f"**Triggered by:** {self.triggered_by}", ""]
        for s in self.stages:
            mark = "✅" if s["success"] else "❌"
            lines.append(f"- {mark} {s['name']} ({s['duration_s']:.2f}s)")
            if s.get("error"):
                lines.append(f"  - Error: {s['error']}")
        return "\n".join(lines)


class MLPipelineCI:
    """Compose and execute ML CI/CD pipelines."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()
        self.pipelines: dict[str, list[PipelineStage]] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        self.pipelines["retrain"] = [
            PipelineStage("check_drift", self._check_drift, required=False, on_failure="warn"),
            PipelineStage("validate_data", self._validate_data, required=True),
            PipelineStage("train", self._train, required=True),
            PipelineStage("evaluate", self._evaluate, required=True),
            PipelineStage("compare_to_production", self._compare, required=True),
            PipelineStage("register", self._register, required=False, on_failure="warn"),
        ]
        self.pipelines["validate"] = [
            PipelineStage("check_drift", self._check_drift, required=False, on_failure="warn"),
            PipelineStage("evaluate", self._evaluate, required=True),
        ]
        self.pipelines["deploy"] = [
            PipelineStage("evaluate", self._evaluate, required=True),
            PipelineStage("smoke_test", self._smoke_test, required=True),
            PipelineStage("transition_production", self._transition_production, required=True),
        ]

    async def run(self, pipeline_name: str, context: dict, triggered_by: str = "manual") -> PipelineResult:
        stages = self.pipelines.get(pipeline_name, [])
        if not stages:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")
        start = time.time()
        results: list[dict[str, Any]] = []
        overall_success = True
        for stage in stages:
            t0 = time.time()
            success = True
            error = None
            output: Any = None
            for attempt in range(stage.retries):
                try:
                    output = await self._run_stage(stage, context) if asyncio.iscoroutinefunction(stage.fn) else stage.fn(context)
                    success = True
                    break
                except Exception as e:
                    error = str(e)
                    success = False
                    logger.warning("ci.stage_failed", stage=stage.name, attempt=attempt, error=error)
            if not success:
                if stage.on_failure == "fail" and stage.required:
                    overall_success = False
                if stage.on_failure == "warn":
                    logger.warning("ci.stage_warn", stage=stage.name)
            results.append({
                "name": stage.name,
                "success": success,
                "duration_s": time.time() - t0,
                "error": error,
                "output": output if isinstance(output, (dict, list, str, int, float, bool, type(None))) else str(output),
            })
            if not success and stage.on_failure == "fail" and stage.required:
                break
        return PipelineResult(
            success=overall_success,
            stages=results,
            started_at=start,
            finished_at=time.time(),
            triggered_by=triggered_by,
        )

    def add_pipeline(self, name: str, stages: list[PipelineStage]) -> None:
        self.pipelines[name] = stages

    # === Default stage implementations ===
    async def _run_stage(self, stage: PipelineStage, context: dict) -> Any:
        result = stage.fn(context)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def _check_drift(self, context: dict) -> dict:
        if "reference_data" not in context or "current_data" not in context:
            return {"skipped": "no data"}
        det = DriftDetector(method="ks")
        result = det.detect(context["reference_data"], context["current_data"], context.get("feature_names"))
        if result.drift_detected:
            logger.warning("ci.drift_detected", scores=result.feature_scores)
        return {"drift_detected": result.drift_detected, "scores": result.feature_scores}

    def _validate_data(self, context: dict) -> dict:
        if "data_path" not in context:
            return {"skipped": "no data path"}
        from ai_engineer.ml.data.validator import auto_infer_expectations
        import pandas as pd
        df = pd.read_csv(context["data_path"], nrows=10_000)
        v = auto_infer_expectations(df, target=context.get("target"))
        result = v.validate(df)
        if not result.success:
            raise ValueError(f"Data validation failed: {result.expectations}")
        return {"expectations": len(result.expectations), "passed": result.success}

    def _train(self, context: dict) -> dict:
        if "train_fn" not in context:
            return {"skipped": "no train_fn"}
        return context["train_fn"](context)

    def _evaluate(self, context: dict) -> dict:
        if "model_path" not in context or "X_test" not in context or "y_test" not in context:
            return {"skipped": "missing inputs"}
        import joblib
        m = joblib.load(Path(context["model_path"]) / "model.pkl")
        y_pred = m.predict(context["X_test"])
        y_score = m.predict_proba(context["X_test"]) if hasattr(m, "predict_proba") else None
        mc = MetricsComputer(task=context.get("task", "classification"))
        metrics = mc.compute(context["y_test"], y_pred, y_score)
        return {"metrics": metrics}

    def _compare(self, context: dict) -> dict:
        if "model_name" not in context or "candidate_metrics" not in context:
            return {"skipped": "missing"}
        try:
            prod = self.registry.get(context["model_name"], stage="production")
        except Exception:
            return {"skipped": "no production model"}
        cand = context["candidate_metrics"]
        primary = "roc_auc" if "roc_auc" in cand else ("f1" if "f1" in cand else next(iter(cand)))
        if cand.get(primary, 0) >= prod.metrics.get(primary, 0):
            return {"promote": True, "candidate": cand.get(primary), "production": prod.metrics.get(primary)}
        return {"promote": False, "candidate": cand.get(primary), "production": prod.metrics.get(primary)}

    def _register(self, context: dict) -> dict:
        if not all(k in context for k in ("model_name", "model_path", "metrics")):
            return {"skipped": "missing"}
        rm = self.registry.register(
            name=context["model_name"],
            path=context["model_path"],
            metrics=context["metrics"],
            params=context.get("params", {}),
            tags=context.get("tags", {}),
            description=context.get("description", ""),
        )
        return {"version": rm.version}

    def _smoke_test(self, context: dict) -> dict:
        if "model_path" not in context:
            return {"skipped": "no model"}
        return {"status": "ok"}

    def _transition_production(self, context: dict) -> dict:
        if not all(k in context for k in ("model_name", "version")):
            return {"skipped": "missing"}
        rm = self.registry.transition(context["model_name"], context["version"], "production")
        return {"version": rm.version, "stage": rm.stage}
