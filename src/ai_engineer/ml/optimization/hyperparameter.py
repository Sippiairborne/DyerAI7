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

"""Hyperparameter optimization with Optuna / Ray Tune / Hyperopt."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np

from ai_engineer.utils.errors import AIEngineerError
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Backend = Literal["optuna", "ray", "hyperopt"]


@dataclass
class TrialResult:
    number: int
    params: dict[str, Any]
    value: float
    state: str
    duration_s: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TuningResult:
    best_params: dict[str, Any]
    best_value: float
    trials: list[TrialResult]
    study_path: str
    n_trials: int
    total_time_s: float


class HyperparameterTuner:
    def __init__(self, backend: Backend = "optuna") -> None:
        self.backend = backend

    def optimize(
        self,
        objective: Callable[[dict[str, Any]], float],
        search_space: dict[str, Any],
        n_trials: int = 50,
        direction: str = "maximize",
        study_name: str = "study",
        storage: str | None = None,
        pruner: bool = True,
        timeout: int | None = None,
        n_jobs: int = 1,
    ) -> TuningResult:
        start = time.time()
        if self.backend == "optuna":
            return self._optuna(objective, search_space, n_trials, direction, study_name, storage, pruner, timeout, n_jobs, start)
        if self.backend == "ray":
            return self._ray(objective, search_space, n_trials, direction, study_name, timeout, start)
        if self.backend == "hyperopt":
            return self._hyperopt(objective, search_space, n_trials, study_name, start)
        raise AIEngineerError(f"Unknown backend: {self.backend}")

    def _optuna(self, objective, search_space, n_trials, direction, study_name, storage, pruner, timeout, n_jobs, start) -> TuningResult:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(n_startup_trials=min(10, n_trials // 5), multivariate=True)
        study = optuna.create_study(direction=direction, study_name=study_name, storage=storage, sampler=sampler, load_if_exists=True)
        if pruner:
            study = optuna.create_study(direction=direction, study_name=study_name, sampler=sampler, pruner=optuna.pruners.MedianPruner())
        trials: list[TrialResult] = []

        def optuna_objective(trial: "optuna.Trial") -> float:
            params = self._sample_optuna(trial, search_space)
            t0 = time.time()
            try:
                value = objective(params)
            except Exception as e:
                logger.warning("tuner.trial_failed", error=str(e))
                raise optuna.exceptions.TrialPruned()
            trials.append(TrialResult(number=trial.number, params=params, value=float(value), state="complete", duration_s=time.time() - t0))
            return float(value)

        study.optimize(optuna_objective, n_trials=n_trials, timeout=timeout, n_jobs=n_jobs, show_progress_bar=False)
        study_path = f"/tmp/{study_name}.db"
        return TuningResult(
            best_params=study.best_params,
            best_value=float(study.best_value),
            trials=trials,
            study_path=study_path,
            n_trials=len(study.trials),
            total_time_s=time.time() - start,
        )

    def _sample_optuna(self, trial, space: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in space.items():
            if isinstance(v, dict):
                kind = v.get("type")
                if kind == "int":
                    out[k] = trial.suggest_int(k, v["low"], v["high"], step=v.get("step", 1), log=v.get("log", False))
                elif kind == "uniform":
                    out[k] = trial.suggest_float(k, v["low"], v["high"], log=v.get("log", False))
                elif kind == "categorical":
                    out[k] = trial.suggest_categorical(k, v["choices"])
                elif kind == "loguniform":
                    out[k] = trial.suggest_float(k, v["low"], v["high"], log=True)
            else:
                out[k] = v
        return out

    def _ray(self, objective, space, n_trials, direction, name, timeout, start) -> TuningResult:
        from ray import tune
        config = {}
        for k, v in space.items():
            if isinstance(v, dict):
                if v.get("type") == "int":
                    config[k] = tune.randint(v["low"], v["high"])
                elif v.get("type") == "uniform":
                    config[k] = tune.uniform(v["low"], v["high"])
                elif v.get("type") == "categorical":
                    config[k] = tune.choice(v["choices"])
                elif v.get("type") == "loguniform":
                    config[k] = tune.loguniform(v["low"], v["high"])
            else:
                config[k] = v
        results = []
        def trainable(config):
            try:
                score = objective(config)
                tune.report({"score": score})
            except Exception:
                tune.report({"score": -1e9 if direction == "maximize" else 1e9})
        analysis = tune.run(trainable, config=config, num_samples=n_trials, metric="score", mode=direction, name=name, time_budget_s=timeout)
        best = analysis.get_best_config(metric="score", mode=direction)
        return TuningResult(best_params=best, best_value=float(analysis.get_best_trial().last_result["score"]), trials=[], study_path=str(analysis.get_best_logdir()), n_trials=n_trials, total_time_s=time.time() - start)

    def _hyperopt(self, objective, space, n_trials, name, start) -> TuningResult:
        from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
        sp = {}
        for k, v in space.items():
            if isinstance(v, dict):
                if v.get("type") == "int":
                    sp[k] = hp.quniform(k, v["low"], v["high"], 1)
                elif v.get("type") == "uniform":
                    sp[k] = hp.uniform(k, v["low"], v["high"])
                elif v.get("type") == "loguniform":
                    sp[k] = hp.loguniform(k, np.log(v["low"]), np.log(v["high"]))
                elif v.get("type") == "categorical":
                    sp[k] = hp.choice(k, v["choices"])
            else:
                sp[k] = v
        trials = Trials()
        best = fmin(lambda p: {"loss": -objective(p) if True else objective(p), "status": STATUS_OK}, space=sp, algo=tpe.suggest, max_evals=n_trials, trials=trials)
        return TuningResult(best_params=best, best_value=-min(t["result"]["loss"] for t in trials.trials), trials=[], study_path="", n_trials=n_trials, total_time_s=time.time() - start)
