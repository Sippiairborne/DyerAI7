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

"""Full ML capabilities."""
from ai_engineer.ml.data.profiler import DataProfiler, ProfileReport
from ai_engineer.ml.data.cleaner import DataCleaner, CleaningReport
from ai_engineer.ml.data.validator import DataValidator, ValidationReport
from ai_engineer.ml.data.augmenter import DataAugmenter
from ai_engineer.ml.data.synthesizer import DataSynthesizer
from ai_engineer.ml.data.labeler import DataLabeler
from ai_engineer.ml.data.splitter import DataSplitter
from ai_engineer.ml.data.balancer import DataBalancer
from ai_engineer.ml.features.engineer import FeatureEngineer
from ai_engineer.ml.features.selector import FeatureSelector
from ai_engineer.ml.features.scaler import FeatureScaler
from ai_engineer.ml.features.store import FeatureStore
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
from ai_engineer.ml.training.trainer import UnifiedTrainer, TrainingConfig
from ai_engineer.ml.training.distributed import DistributedLauncher
from ai_engineer.ml.training.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    WandBLogger,
    MLflowLogger,
    GradientClipping,
    EMA,
    SWA,
)
from ai_engineer.ml.evaluation.metrics import MetricsComputer
from ai_engineer.ml.evaluation.statistical import StatisticalTester
from ai_engineer.ml.evaluation.calibration import CalibrationAnalyzer
from ai_engineer.ml.evaluation.fairness import FairnessAuditor
from ai_engineer.ml.evaluation.robustness import RobustnessTester
from ai_engineer.ml.evaluation.slices import SliceFinder
from ai_engineer.ml.evaluation.reports import ReportGenerator
from ai_engineer.ml.interpretation.shap_explain import SHAPExplainer
from ai_engineer.ml.interpretation.lime_explain import LIMEExplainer
from ai_engineer.ml.interpretation.attention import AttentionVisualizer
from ai_engineer.ml.interpretation.integrated_grad import IntegratedGradients
from ai_engineer.ml.interpretation.counterfactual import CounterfactualExplorer
from ai_engineer.ml.optimization.hyperparameter import HyperparameterTuner
from ai_engineer.ml.optimization.nas import NAS
from ai_engineer.ml.optimization.pruning import Pruner
from ai_engineer.ml.optimization.quantization import Quantizer
from ai_engineer.ml.optimization.distillation import Distiller
from ai_engineer.ml.optimization.compile import ModelCompiler
from ai_engineer.ml.optimization.peft import PEFTFactory
from ai_engineer.ml.deployment.exporter import ModelExporter
from ai_engineer.ml.deployment.server import ServingDeployer
from ai_engineer.ml.deployment.canary import CanaryDeployer
from ai_engineer.ml.deployment.ab_test import ABTestFramework
from ai_engineer.ml.monitoring.drift import DriftDetector
from ai_engineer.ml.monitoring.performance import PerformanceMonitor
from ai_engineer.ml.monitoring.alerts import AlertManager
from ai_engineer.ml.ops.lineage import LineageTracker
from ai_engineer.ml.ops.governance import ModelCard, GovernanceManager
from ai_engineer.ml.ops.ci import MLPipelineCI

__all__ = [name for name in dir() if not name.startswith("_")]
