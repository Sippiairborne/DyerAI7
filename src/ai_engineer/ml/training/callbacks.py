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

"""Training callbacks: early stopping, checkpointing, W&B, MLflow, EMA, SWA, gradient clipping."""
from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ai_engineer.ml.training.trainer import UnifiedTrainer

from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


class Callback:
    def on_train_start(self, trainer: "UnifiedTrainer") -> None: pass
    def on_train_end(self, trainer: "UnifiedTrainer") -> None: pass
    def on_epoch_end(self, trainer: "UnifiedTrainer", record: dict) -> None: pass
    def on_step_end(self, trainer: "UnifiedTrainer", record: dict) -> None: pass


class EarlyStopping(Callback):
    def __init__(self, patience: int = 5, metric: str = "val_metric", mode: str = "max", min_delta: float = 0.0) -> None:
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0

    def on_epoch_end(self, trainer, record):
        v = record.get(self.metric)
        if v is None:
            return
        improved = (v > self.best + self.min_delta) if self.mode == "max" else (v < self.best - self.min_delta)
        if improved:
            self.best = v
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                trainer.global_step = float("inf")  # Hack to break out
                logger.info("callback.early_stop", metric=self.metric, best=self.best)


class ModelCheckpoint(Callback):
    def __init__(self, dirpath: str, metric: str = "val_metric", mode: str = "max", save_top_k: int = 3) -> None:
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.metric = metric
        self.mode = mode
        self.save_top_k = save_top_k
        self.saved: list[tuple[float, str]] = []

    def on_epoch_end(self, trainer, record):
        v = record.get(self.metric)
        if v is None:
            return
        path = self.dirpath / f"epoch{record.get('epoch', 0)}_v{v:.4f}.pt"
        torch.save({"state_dict": trainer.model.state_dict(), "epoch": record.get("epoch"), "metric": v}, path)
        self.saved.append((v, str(path)))
        self.saved.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        while len(self.saved) > self.save_top_k:
            _, p = self.saved.pop()
            try:
                Path(p).unlink()
            except Exception:
                pass


class WandBLogger(Callback):
    def __init__(self, project: str, run_name: str | None = None, config: dict | None = None) -> None:
        self.project = project
        self.run_name = run_name
        self.config = config or {}
        self._run = None

    def on_train_start(self, trainer):
        try:
            import wandb
            self._run = wandb.init(project=self.project, name=self.run_name, config=self.config, reinit=True)
        except Exception as e:
            logger.warning("callback.wandb_init_failed", error=str(e))

    def on_step_end(self, trainer, record):
        if self._run:
            try:
                import wandb
                wandb.log(record, step=trainer.global_step)
            except Exception:
                pass

    def on_epoch_end(self, trainer, record):
        if self._run:
            try:
                import wandb
                wandb.log(record, step=trainer.global_step)
            except Exception:
                pass

    def on_train_end(self, trainer):
        if self._run:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass


class MLflowLogger(Callback):
    def __init__(self, tracking_uri: str, experiment: str, run_name: str | None = None) -> None:
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self.run_name = run_name
        self._run = None

    def on_train_start(self, trainer):
        try:
            import mlflow
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment)
            self._run = mlflow.start_run(run_name=self.run_name)
        except Exception as e:
            logger.warning("callback.mlflow_init_failed", error=str(e))

    def on_epoch_end(self, trainer, record):
        if self._run:
            try:
                import mlflow
                mlflow.log_metrics({k: float(v) for k, v in record.items() if isinstance(v, (int, float))}, step=record.get("epoch", 0))
            except Exception:
                pass

    def on_train_end(self, trainer):
        if self._run:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass


class GradientClipping(Callback):
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm

    def on_step_end(self, trainer, record):
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), self.max_norm)


class EMA(Callback):
    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}

    def on_step_end(self, trainer, record):
        for n, p in trainer.model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = self.decay * self.shadow.get(n, p.detach().clone()) + (1 - self.decay) * p.detach().clone()

    def on_train_end(self, trainer):
        for n, p in trainer.model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n])


class SWA(Callback):
    def __init__(self, swa_start: int = 5, swa_freq: int = 1):
        self.swa_start = swa_start
        self.swa_freq = swa_freq
        self.swa_model = None
        self.epoch = 0

    def on_epoch_end(self, trainer, record):
        if record.get("epoch", 0) >= self.swa_start:
            if self.swa_model is None:
                from torch.optim.swa_utils import AveragedModel
                self.swa_model = AveragedModel(trainer.model)
            else:
                self.swa_model.update_parameters(trainer.model)
        self.epoch += 1
