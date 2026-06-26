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

"""Deep learning trainer for custom PyTorch models."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ai_engineer.ml.models.registry import ModelRegistry
from ai_engineer.ml.training.trainer import UnifiedTrainer, TrainingConfig
from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DeepTrainingResult:
    model_path: str
    metrics: dict[str, float]
    history: list[dict[str, float]] = field(default_factory=list)
    training_time_s: float = 0.0
    best_epoch: int = 0
    best_metric: float = 0.0


class DeepModelTrainer:
    """Train any custom nn.Module on tabular tensors."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def train(
        self,
        model: nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        config: TrainingConfig | None = None,
        task: str = "classification",
        register_name: str | None = None,
        output_dir: str | None = None,
    ) -> DeepTrainingResult:
        config = config or TrainingConfig()
        output_dir = output_dir or f"/tmp/deep_{int(time.time())}"
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
        model = model.to(device)

        Xt = torch.tensor(X_train, dtype=torch.float32)
        yt = torch.tensor(y_train, dtype=torch.long if task == "classification" else torch.float32)
        train_ds = TensorDataset(Xt, yt)
        train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)

        if X_val is not None:
            Xv = torch.tensor(X_val, dtype=torch.float32)
            yv = torch.tensor(y_val, dtype=torch.long if task == "classification" else torch.float32)
            val_dl = DataLoader(TensorDataset(Xv, yv), batch_size=config.batch_size * 2, num_workers=2, pin_memory=True)
        else:
            val_dl = None

        from ai_engineer.ml.training.trainer import UnifiedTrainer
        trainer = UnifiedTrainer(model=model, config=config, task=task, device=device, output_dir=out_path)
        history = trainer.fit(train_dl, val_dl)
        best = max(history, key=lambda h: h.get("val_metric", 0.0) if val_dl else h.get("train_loss", 0.0))

        # Save
        torch.save(model.state_dict(), out_path / "model.pt")
        (out_path / "config.json").write_text(json.dumps(config.__dict__, indent=2, default=str))
        (out_path / "history.json").write_text(json.dumps(history, indent=2))
        (out_path / "task.txt").write_text(task)

        metrics = {
            "best_val_metric": float(best.get("val_metric", 0.0)) if val_dl else float(best.get("train_loss", 0.0)),
            "best_epoch": int(best.get("epoch", 0)),
            "final_train_loss": float(history[-1].get("train_loss", 0.0)) if history else 0.0,
        }

        if register_name:
            rm = self.registry.register(
                name=register_name,
                path=out_path,
                metrics=metrics,
                params=config.__dict__,
                tags={"task": task, "framework": "pytorch"},
                description="Custom deep model",
            )

        return DeepTrainingResult(
            model_path=str(out_path / "model.pt"),
            metrics=metrics,
            history=history,
            training_time_s=sum(h.get("time_s", 0) for h in history),
            best_epoch=int(best.get("epoch", 0)),
            best_metric=float(best.get("val_metric", 0.0)) if val_dl else float(best.get("train_loss", 0.0)),
        )
