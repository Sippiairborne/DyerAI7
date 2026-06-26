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

"""Experiment tracking integration (W&B + MLflow)."""
from __future__ import annotations

from ai_engineer.config import get_settings
from ai_engineer.tools.registry import ToolRegistry, tool

_registry = ToolRegistry()


@tool(
    name="log_metrics",
    description="Log metrics to W&B and MLflow. Pass metrics as a JSON object string.",
)
def log_metrics(project: str, run_name: str, metrics_json: str) -> str:
    import json
    import mlflow
    import wandb

    settings = get_settings()
    metrics = json.loads(metrics_json)

    # W&B
    if settings.wandb_api_key:
        wandb.init(project=project or settings.wandb_project, name=run_name, reinit=True)
        wandb.log(metrics)
        wandb.finish()

    # MLflow
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(project or settings.wandb_project)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics(metrics)

    return f"Logged {list(metrics.keys())} to {project}/{run_name}"
