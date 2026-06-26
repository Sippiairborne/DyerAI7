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

"""Model trainers for all modalities and tasks."""
from ai_engineer.ml.models.registry import ModelRegistry, RegisteredModel
from ai_engineer.ml.models.classical import ClassicalTrainer
from ai_engineer.ml.models.deep import DeepModelTrainer
from ai_engineer.ml.models.llm import LLMTrainer
from ai_engineer.ml.models.vision import VisionTrainer
from ai_engineer.ml.models.audio import AudioTrainer
from ai_engineer.ml.models.tabular import TabularDLTrainer
from ai_engineer.ml.models.time_series import TimeSeriesTrainer
from ai_engineer.ml.models.graph import GNNTrainer
from ai_engineer.ml.models.recommendation import RecSysTrainer
from ai_engineer.ml.models.rl import RLTrainer
from ai_engineer.ml.models.ensemble import EnsembleBuilder

__all__ = [
    "ModelRegistry", "RegisteredModel",
    "ClassicalTrainer", "DeepModelTrainer", "LLMTrainer",
    "VisionTrainer", "AudioTrainer", "TabularDLTrainer",
    "TimeSeriesTrainer", "GNNTrainer", "RecSysTrainer", "RLTrainer",
    "EnsembleBuilder",
]
