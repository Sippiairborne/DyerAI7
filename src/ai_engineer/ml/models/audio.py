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

"""Audio model training: ASR, TTS, classification, separation."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ai_engineer.ml.models.registry import ModelRegistry
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Task = Literal["asr", "tts", "classification", "separation"]


@dataclass
class AudioConfig:
    task: Task = "asr"
    model_name: str = "openai/whisper-small"  # for ASR
    language: str = "en"
    num_epochs: int = 3
    batch_size: int = 8
    learning_rate: float = 1e-5
    output_dir: str = ""
    use_lora: bool = True


@dataclass
class AudioResult:
    output_dir: str
    metrics: dict[str, float] = field(default_factory=dict)


class AudioTrainer:
    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def train_asr(self, config: AudioConfig, dataset_path: str, register_name: str | None = None) -> AudioResult:
        if not config.output_dir:
            config.output_dir = f"/tmp/asr_{int(time.time())}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        script = f"""
import os
os.environ['HF_HOME'] = '/data/hf'
from datasets import load_from_disk, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model
import torch

processor = WhisperProcessor.from_pretrained('{config.model_name}', language='{config.language}', task='transcribe')
model = WhisperForConditionalGeneration.from_pretrained('{config.model_name}')
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language='{config.language}', task='transcribe')

dataset = load_from_disk('{dataset_path}').cast_column('audio', Audio(sampling_rate=16000))

def prep(batch):
    audio = batch['audio']
    batch['input_features'] = processor.feature_extractor(audio['array'], sampling_rate=16000).input_features[0]
    batch['labels'] = processor.tokenizer(batch['text']).input_ids
    return batch

dataset = dataset.map(prep, remove_columns=dataset.column_names)

if {str(config.use_lora).lower()}:
    cfg = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj'], task_type='SEQ_2_SEQ_LM')
    model = get_peft_model(model, cfg)

args = Seq2SeqTrainingArguments(
    output_dir='{config.output_dir}', num_train_epochs={config.num_epochs},
    per_device_train_batch_size={config.batch_size}, learning_rate={config.learning_rate},
    bf16=True, evaluation_strategy='steps', eval_steps=200, save_steps=200, report_to='none', predict_with_generate=True,
)
trainer = Seq2SeqTrainer(args=args, model=model, train_dataset=dataset, eval_dataset=dataset, tokenizer=processor.feature_extractor)
trainer.train()
trainer.save_model('{config.output_dir}')
print('ASR_COMPLETE')
"""
        Path(config.output_dir, "train.py").write_text(script)
        Path(config.output_dir, "config.json").write_text(json.dumps(config.__dict__, indent=2, default=str))
        return AudioResult(output_dir=config.output_dir, metrics={"script_path": f"{config.output_dir}/train.py"})

    def train_tts(self, config: AudioConfig, dataset_path: str, register_name: str | None = None) -> AudioResult:
        if not config.output_dir:
            config.output_dir = f"/tmp/tts_{int(time.time())}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        script = f"""
import os
os.environ['HF_HOME'] = '/data/hf'
from transformers import VitsModel, VitsTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk, Audio

model = VitsModel.from_pretrained('facebook/mms-tts-{config.language}')
tok = VitsTokenizer.from_pretrained('facebook/mms-tts-{config.language}')
ds = load_from_disk('{dataset_path}').cast_column('audio', Audio(sampling_rate=16000))

def prep(b):
    a = b['audio']
    inputs = tok(text=b['text'], return_tensors='pt')
    b['input_ids'] = inputs.input_ids[0]
    b['waveform'] = a['array']
    return b

ds = ds.map(prep, remove_columns=ds.column_names)
args = TrainingArguments(output_dir='{config.output_dir}', num_train_epochs={config.num_epochs},
    per_device_train_batch_size={config.batch_size}, learning_rate={config.learning_rate}, report_to='none')
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
trainer.save_model('{config.output_dir}')
print('TTS_COMPLETE')
"""
        Path(config.output_dir, "train.py").write_text(script)
        return AudioResult(output_dir=config.output_dir)

    def train_classification(self, config: AudioConfig, dataset_path: str, num_labels: int, register_name: str | None = None) -> AudioResult:
        if not config.output_dir:
            config.output_dir = f"/tmp/audio_cls_{int(time.time())}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        script = f"""
import os
os.environ['HF_HOME'] = '/data/hf'
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, TrainingArguments, Trainer
from datasets import load_from_disk, Audio

model = AutoModelForAudioClassification.from_pretrained('facebook/wav2vec2-base', num_labels={num_labels})
fe = AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base')
ds = load_from_disk('{dataset_path}').cast_column('audio', Audio(sampling_rate=16000))

def prep(b):
    a = b['audio']
    b['input_values'] = fe(a['array'], sampling_rate=16000).input_values[0]
    return b

ds = ds.map(prep, remove_columns=['audio'])
args = TrainingArguments(output_dir='{config.output_dir}', num_train_epochs={config.num_epochs},
    per_device_train_batch_size={config.batch_size}, learning_rate={config.learning_rate}, report_to='none')
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
trainer.save_model('{config.output_dir}')
print('AUDIO_CLS_COMPLETE')
"""
        Path(config.output_dir, "train.py").write_text(script)
        return AudioResult(output_dir=config.output_dir)
