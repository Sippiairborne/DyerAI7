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

"""Unified training loop with mixed precision, accumulation, EMA, etc."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ai_engineer.ml.training.callbacks import Callback
from ai_engineer.ml.training.optimizers import build_optimizer
from ai_engineer.ml.training.schedulers import build_scheduler
from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adamw"  # adam | adamw | sgd | lamb | lion | adafactor | sophia
    scheduler: str = "cosine"  # cosine | linear | constant | onecycle | cosine_warm_restarts | polynomial
    warmup_steps: int = 0
    warmup_ratio: float = 0.03
    grad_accum_steps: int = 1
    max_grad_norm: float = 1.0
    use_amp: bool = True
    use_gpu: bool = True
    log_every: int = 10
    eval_every: int = 0  # 0 = end of epoch
    save_every: int = 0  # 0 = end of training
    early_stopping_patience: int = 0
    ema_decay: float = 0.0  # 0 = off
    seed: int = 42
    mixed_precision_dtype: str = "bf16"  # bf16 | fp16
    gradient_checkpointing: bool = False
    deterministic: bool = False


class UnifiedTrainer:
    """One trainer to rule them all. Works for any nn.Module + DataLoader."""

    def __init__(self, model: nn.Module, config: TrainingConfig, task: str = "classification", device: torch.device | None = None, output_dir: Path | None = None, callbacks: list[Callback] | None = None) -> None:
        self.model = model
        self.config = config
        self.task = task
        self.device = device or torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")
        self.output_dir = output_dir or Path(f"/tmp/run_{int(time.time())}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.callbacks = callbacks or []
        self.global_step = 0
        self.ema = None
        if config.ema_decay > 0:
            from torch.optim.swa_utils import AveragedModel
            self.ema = AveragedModel(self.model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(config.ema_decay))
        self._set_seed()
        self.model.to(self.device)
        if config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def _set_seed(self) -> None:
        import random
        import numpy as np
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        if self.config.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _build_loss(self) -> nn.Module:
        if self.task == "classification":
            return nn.CrossEntropyLoss()
        if self.task == "regression":
            return nn.MSELoss()
        if self.task == "binary":
            return nn.BCEWithLogitsLoss()
        if self.task == "multilabel":
            return nn.BCEWithLogitsLoss()
        raise AIEngineerError(f"Unknown task: {self.task}")

    def fit(self, train_dl: DataLoader, val_dl: DataLoader | None = None) -> list[dict[str, Any]]:
        config = self.config
        loss_fn = self._build_loss()
        opt = build_optimizer(config.optimizer, self.model.parameters(), config.learning_rate, config.weight_decay)
        steps_per_epoch = max(len(train_dl) // config.grad_accum_steps, 1)
        total_steps = steps_per_epoch * config.num_epochs
        sch = build_scheduler(config.scheduler, opt, total_steps, config.warmup_steps, config.warmup_ratio)
        amp_dtype = torch.bfloat16 if config.mixed_precision_dtype == "bf16" else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp and amp_dtype == torch.float16)

        history: list[dict[str, Any]] = []
        best_val = float("-inf")
        patience_left = config.early_stopping_patience
        for cb in self.callbacks:
            cb.on_train_start(self)

        for epoch in range(config.num_epochs):
            self.model.train()
            t0 = time.time()
            ep_losses = []
            opt.zero_grad()
            for step, batch in enumerate(train_dl):
                batch = self._to_device(batch)
                with torch.amp.autocast("cuda", enabled=config.use_amp, dtype=amp_dtype):
                    out = self._forward(batch)
                    loss = self._compute_loss(loss_fn, out, batch)
                    loss = loss / config.grad_accum_steps
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step + 1) % config.grad_accum_steps == 0:
                    if config.max_grad_norm > 0:
                        if scaler.is_enabled():
                            scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.max_grad_norm)
                    if scaler.is_enabled():
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad()
                    if sch is not None:
                        sch.step()
                    if self.ema is not None:
                        self.ema.update_parameters(self.model)
                    self.global_step += 1
                    for cb in self.callbacks:
                        cb.on_step_end(self, {"loss": float(loss.item() * config.grad_accum_steps), "lr": opt.param_groups[0]["lr"]})
                    if config.log_every and self.global_step % config.log_every == 0:
                        logger.info("train.step", step=self.global_step, loss=float(loss.item() * config.grad_accum_steps))
                ep_losses.append(float(loss.item() * config.grad_accum_steps))

            train_loss = sum(ep_losses) / max(len(ep_losses), 1)
            ep_time = time.time() - t0

            val_metrics: dict[str, float] = {}
            if val_dl is not None and (config.eval_every == 0 or (epoch + 1) % config.eval_every == 0):
                val_metrics = self.evaluate(val_dl)
                if val_metrics.get("val_metric", 0.0) > best_val:
                    best_val = val_metrics["val_metric"]
                    patience_left = config.early_stopping_patience
                else:
                    patience_left -= 1
            elif val_dl is None:
                val_metrics = {"val_metric": -train_loss}

            epoch_record = {"epoch": epoch, "train_loss": train_loss, "time_s": ep_time, **val_metrics}
            history.append(epoch_record)
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch_record)
            logger.info("train.epoch", **epoch_record)
            if config.early_stopping_patience and patience_left <= 0:
                logger.info("train.early_stop", epoch=epoch)
                break

        for cb in self.callbacks:
            cb.on_train_end(self)
        return history

    @torch.no_grad()
    def evaluate(self, val_dl: DataLoader) -> dict[str, float]:
        self.model.eval()
        loss_fn = self._build_loss()
        losses: list[float] = []
        all_logits: list[torch.Tensor] = []
        all_y: list[torch.Tensor] = []
        amp_dtype = torch.bfloat16 if self.config.mixed_precision_dtype == "bf16" else torch.float16
        for batch in val_dl:
            batch = self._to_device(batch)
            with torch.amp.autocast("cuda", enabled=self.config.use_amp, dtype=amp_dtype):
                out = self._forward(batch)
                loss = self._compute_loss(loss_fn, out, batch)
            losses.append(float(loss.item()))
            all_logits.append(out.detach().float())
            all_y.append(batch[1] if isinstance(batch, (tuple, list)) else batch["labels"])
        from ai_engineer.ml.evaluation.metrics import MetricsComputer
        mc = MetricsComputer(task=self.task)
        metrics = mc.compute(torch.cat(all_y).cpu().numpy(), torch.cat(all_logits).cpu().numpy())
        metrics["val_loss"] = float(sum(losses) / max(len(losses), 1))
        primary = mc.primary_metric(metrics)
        metrics["val_metric"] = float(primary)
        return metrics

    def _to_device(self, batch: Any) -> Any:
        if isinstance(batch, dict):
            return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if isinstance(batch, (tuple, list)):
            return [v.to(self.device) if torch.is_tensor(v) else v for v in batch]
        return batch.to(self.device)

    def _forward(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, dict):
            if "pixel_values" in batch:
                return self.model(pixel_values=batch["pixel_values"])
            if "input_ids" in batch:
                return self.model(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
            x = batch.get("x", batch.get("features"))
            return self.model(x)
        x = batch[0]
        return self.model(x)

    def _compute_loss(self, loss_fn: nn.Module, out: torch.Tensor, batch: Any) -> torch.Tensor:
        if isinstance(batch, dict):
            y = batch.get("y", batch.get("labels", batch.get("target")))
        else:
            y = batch[1]
        if self.task in ("binary", "multilabel") and y.dtype == torch.long:
            y = y.float()
        return loss_fn(out, y)
