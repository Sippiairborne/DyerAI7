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

"""Vision model training: classification, detection, segmentation, diffusion fine-tuning."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from ai_engineer.ml.models.registry import ModelRegistry
from ai_engineer.utils.logging import get_logger

logger = get_logger(__name__)

Task = Literal["classification", "detection", "segmentation", "diffusion"]


@dataclass
class VisionConfig:
    task: Task = "classification"
    backbone: str = "resnet50"  # resnet50, vit_base_patch16_224, swin_tiny, convnext_tiny, ssd, faster_rcnn, mask_rcnn, stable_diffusion
    num_classes: int = 1000
    pretrained: bool = True
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    image_size: int = 224
    use_amp: bool = True
    output_dir: str = ""


@dataclass
class VisionResult:
    output_dir: str
    metrics: dict[str, float] = field(default_factory=dict)
    training_time_s: float = 0.0


class VisionTrainer:
    """Train vision models with timm + torchvision + HuggingFace."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry()

    def train_classification(self, config: VisionConfig, train_dir: str, val_dir: str, register_name: str | None = None) -> VisionResult:
        if not config.output_dir:
            config.output_dir = f"/tmp/vision_cls_{int(time.time())}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        script = f"""
import os, json, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_tf = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)) if False else transforms.Resize(config.image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tf = transforms.Compose([
    transforms.Resize(config.image_size),
    transforms.CenterCrop(config.image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder('{train_dir}', transform=train_tf)
val_ds = datasets.ImageFolder('{val_dir}', transform=val_tf)
train_dl = DataLoader(train_ds, batch_size={config.batch_size}, shuffle=True, num_workers=4, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size={config.batch_size}, shuffle=False, num_workers=4, pin_memory=True)

model = timm.create_model('{config.backbone}', pretrained={str(config.pretrained)}, num_classes={config.num_classes})
model = model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr={config.learning_rate}, weight_decay={config.weight_decay})
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max={config.num_epochs} * len(train_dl))
scaler = torch.amp.GradScaler('cuda', enabled={str(config.use_amp).lower()})
crit = nn.CrossEntropyLoss()

best_acc = 0.0
history = []
for epoch in range({config.num_epochs}):
    model.train()
    t0 = time.time()
    losses = []
    for x, y in train_dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad()
        with torch.amp.autocast('cuda', enabled={str(config.use_amp).lower()}):
            out = model(x)
            loss = crit(out, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sch.step()
        losses.append(loss.item())
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', enabled={str(config.use_amp).lower()}):
                out = model(x)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    acc = correct / max(total, 1)
    history.append({{'epoch': epoch, 'train_loss': float(sum(losses)/len(losses)), 'val_acc': acc, 'time_s': time.time() - t0}})
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), '{config.output_dir}/best.pt')

torch.save(model.state_dict(), '{config.output_dir}/final.pt')
with open('{config.output_dir}/history.json', 'w') as f:
    json.dump(history, f, indent=2)
print(f'VISION_CLS_BEST_ACC {{best_acc:.4f}}')
"""
        Path(config.output_dir, "train.py").write_text(script)
        Path(config.output_dir, "config.json").write_text(json.dumps(config.__dict__, indent=2, default=str))
        return VisionResult(output_dir=config.output_dir, metrics={"script_path": f"{config.output_dir}/train.py"})

    def train_detection(self, config: VisionConfig, train_coco: str, val_coco: str, register_name: str | None = None) -> VisionResult:
        if not config.output_dir:
            config.output_dir = f"/tmp/vision_det_{int(time.time())}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        script = f"""
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets.coco import CocoDetection
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = fasterrcnn_resnet50_fpn(pretrained=True, num_classes={config.num_classes}).to(device)
opt = torch.optim.AdamW(model.parameters(), lr={config.learning_rate}, weight_decay={config.weight_decay})

train_ds = CocoDetection('{train_coco}', annFile='{train_coco}/annotations.json')
val_ds = CocoDetection('{val_coco}', annFile='{val_coco}/annotations.json')

def collate(batch):
    return tuple(zip(*batch))

train_dl = DataLoader(train_ds, batch_size={config.batch_size}, shuffle=True, num_workers=4, collate_fn=collate)

for epoch in range({config.num_epochs}):
    model.train()
    total = 0.0
    for imgs, targets in train_dl:
        imgs = [img.to(device) for img in imgs]
        targets = [{{k: v.to(device) for k, v in t.items()}} for t in targets]
        loss_dict = model(imgs, targets)
        loss = sum(l for l in loss_dict.values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    print(f'epoch {{epoch}} det_loss {{total / max(len(train_dl), 1):.4f}}')
torch.save(model.state_dict(), '{config.output_dir}/detection.pt')
print('VISION_DET_COMPLETE')
"""
        Path(config.output_dir, "train.py").write_text(script)
        return VisionResult(output_dir=config.output_dir)

    def train_segmentation(self, config: VisionConfig, train_dir: str, val_dir: str, register_name: str | None = None) -> VisionResult:
        if not config.output_dir:
            config.output_dir = f"/tmp/vision_seg_{int(time.time())}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        script = f"""
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os, json, time

class SegDataset(Dataset):
    def __init__(self, root, size=224):
        self.root = root
        self.size = size
        self.ids = sorted(os.listdir(os.path.join(root, 'images')))
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        img = Image.open(os.path.join(self.root, 'images', self.ids[i])).convert('RGB').resize((self.size, self.size))
        mask = Image.open(os.path.join(self.root, 'masks', self.ids[i])).resize((self.size, self.size))
        import numpy as np
        return torch.tensor(np.array(img).transpose(2, 0, 1) / 255., dtype=torch.float32), torch.tensor(np.array(mask), dtype=torch.long)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smp.Unet(encoder_name='resnet50', encoder_weights='imagenet', classes={config.num_classes}).to(device)
opt = torch.optim.AdamW(model.parameters(), lr={config.learning_rate})
crit = smp.losses.DiceLoss(mode='multiclass')
train_dl = DataLoader(SegDataset('{train_dir}', {config.image_size}), batch_size={config.batch_size}, shuffle=True, num_workers=4)
for epoch in range({config.num_epochs}):
    model.train()
    losses = []
    for x, y in train_dl:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = crit(out, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f'epoch {{epoch}} seg_loss {{sum(losses)/max(len(losses),1):.4f}}')
torch.save(model.state_dict(), '{config.output_dir}/segmentation.pt')
print('VISION_SEG_COMPLETE')
"""
        Path(config.output_dir, "train.py").write_text(script)
        return VisionResult(output_dir=config.output_dir)

    def fine_tune_diffusion(self, config: VisionConfig, image_dir: str, instance_prompt: str, register_name: str | None = None) -> VisionResult:
        if not config.output_dir:
            config.output_dir = f"/tmp/diffusion_{int(time.time())}"
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        script = f"""
import os
os.environ['HF_HOME'] = '/data/hf'
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.loaders import LoraLoaderMixin
from peft import LoraConfig, get_peft_model
import torch

pipe = StableDiffusionPipeline.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5', torch_dtype=torch.float16).to('cuda')
unet = pipe.unet
text_encoder = pipe.text_encoder
unet.requires_grad_(False)
text_encoder.requires_grad_(False)
unet_lora = LoraConfig(r=16, lora_alpha=32, target_modules=['to_q', 'to_v'])
unet = get_peft_model(unet, unet_lora)
text_lora = LoraConfig(r=16, lora_alpha=32, target_modules=['q_proj', 'v_proj'])
text_encoder = get_peft_model(text_encoder, text_lora)

from PIL import Image
import torchvision.transforms as T
tf = T.Compose([T.Resize((512, 512)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)])
import os
imgs = []
for f in os.listdir('{image_dir}')[:200]:
    imgs.append(tf(Image.open(os.path.join('{image_dir}', f)).convert('RGB')))

opt = torch.optim.AdamW(list(unet.parameters()) + list(text_encoder.parameters()), lr=1e-4)
noise_sched = DDPMScheduler.from_config(pipe.scheduler.config)

for step in range(500):
    img = torch.stack(imgs).to('cuda', dtype=torch.float16)
    noise = torch.randn_like(img)
    t = torch.randint(0, noise_sched.config.num_train_timesteps, (img.shape[0],)).long()
    noisy = noise_sched.add_noise(img, noise, t)
    text_in = pipe.tokenizer(['{instance_prompt}']*img.shape[0], padding='max_length', max_length=77, return_tensors='pt').input_ids.cuda()
    text_emb = text_encoder(text_in)[0]
    pred = unet(noisy, t, text_emb).sample
    loss = torch.nn.functional.mse_loss(pred, noise)
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 50 == 0: print(f'step {{step}} loss {{loss.item():.4f}}')

unet.save_pretrained('{config.output_dir}/unet_lora')
text_encoder.save_pretrained('{config.output_dir}/text_lora')
print('DIFFUSION_FT_COMPLETE')
"""
        Path(config.output_dir, "train.py").write_text(script)
        return VisionResult(output_dir=config.output_dir)
