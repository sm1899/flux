# Complete Flux Training Guide

This guide provides comprehensive instructions for training Flux models from scratch and fine-tuning with LoRA.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Data Preparation](#data-preparation)
4. [Training Scripts](#training-scripts)
5. [Training Examples](#training-examples)
6. [Advanced Training](#advanced-training)
7. [Troubleshooting](#troubleshooting)

## Overview

Flux is a state-of-the-art text-to-image diffusion model that uses a dual-stream transformer architecture. This guide covers:

- **Full Model Training**: Training Flux models from scratch
- **LoRA Fine-tuning**: Efficient fine-tuning using Low-Rank Adaptation
- **Specialized Models**: Training for specific tasks (fill, control, etc.)

## Prerequisites

### System Requirements

- **GPU**: NVIDIA GPU with at least 24GB VRAM (for full training)
- **RAM**: 64GB+ system RAM
- **Storage**: 500GB+ SSD for datasets and checkpoints
- **Python**: 3.8+

### Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install transformers diffusers accelerate
pip install safetensors huggingface_hub
pip install tqdm pillow numpy

# Optional but recommended
pip install wandb tensorboard
pip install xformers  # For memory efficiency
```

### Environment Setup

```bash
# Clone the Flux repository
git clone https://github.com/black-forest-labs/FLUX.1
cd FLUX.1

# Install the package
pip install -e .

# Set up environment variables
export HF_TOKEN="your_huggingface_token"
export CUDA_VISIBLE_DEVICES=0  # Specify GPU
```

## Data Preparation

### Dataset Structure

Your training data should be organized as follows:

```
dataset/
├── metadata.jsonl
├── image1.jpg
├── image2.png
├── image3.webp
└── ...
```

### Metadata Format

The `metadata.jsonl` file should contain one JSON object per line:

```json
{"file_name": "image1.jpg", "text": "a beautiful landscape with mountains"}
{"file_name": "image2.png", "text": "a portrait of a woman in traditional dress"}
{"file_name": "image3.webp", "text": "a futuristic city skyline at sunset"}
```

### Data Preprocessing Script

```python
import json
import os
from PIL import Image

def create_metadata(dataset_path, output_file="metadata.jsonl"):
    """Create metadata.jsonl from a directory of images"""
    metadata = []
    
    for filename in os.listdir(dataset_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            # You can implement custom caption generation here
            # For now, we'll use the filename as caption
            caption = filename.replace('_', ' ').replace('.', ' ')
            
            metadata.append({
                "file_name": filename,
                "text": caption
            })
    
    with open(output_file, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')

# Usage
create_metadata("./your_dataset", "metadata.jsonl")
```

## Training Scripts

### 1. Basic Training Script

The main training script (`flux_training_script.py`) supports:

- **Full model training** from scratch
- **LoRA fine-tuning** for efficient training
- **Multiple model variants** (flux-dev, flux-schnell, etc.)
- **Checkpointing and validation**

### 2. LoRA Training Script

For efficient fine-tuning, use the LoRA approach:

```python
# LoRA configuration
lora_config = {
    "r": 128,           # LoRA rank
    "alpha": 128,       # LoRA alpha
    "dropout": 0.0,     # LoRA dropout
    "target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"]
}
```

### 3. Specialized Training Scripts

#### Fill Model Training

```bash
python flux_training_script.py \
    --model_name flux-dev-fill \
    --train_data_dir ./fill_dataset \
    --output_dir ./fill_model \
    --learning_rate 1e-4 \
    --num_train_epochs 50 \
    --train_batch_size 1
```

#### Control Model Training

```bash
python flux_training_script.py \
    --model_name flux-dev-canny \
    --train_data_dir ./control_dataset \
    --output_dir ./control_model \
    --learning_rate 1e-4 \
    --num_train_epochs 30 \
    --train_batch_size 1
```

## Training Examples

### Example 1: Basic Text-to-Image Training

```bash
# Train a basic Flux model
python flux_training_script.py \
    --model_name flux-dev \
    --train_data_dir ./dataset \
    --output_dir ./trained_model \
    --learning_rate 1e-4 \
    --num_train_epochs 100 \
    --train_batch_size 1 \
    --resolution 1024 \
    --checkpointing_steps 500 \
    --validation_steps 100 \
    --validation_prompt "a beautiful landscape painting"
```

### Example 2: LoRA Fine-tuning

```bash
# Fine-tune with LoRA for efficiency
python flux_training_script.py \
    --model_name flux-dev \
    --train_data_dir ./dataset \
    --output_dir ./lora_model \
    --use_peft_lora \
    --lora_r 128 \
    --lora_alpha 128 \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --train_batch_size 1
```

### Example 3: Kontext Model Training

```bash
# Train for context-aware generation
python flux_training_script.py \
    --model_name flux-dev-kontext \
    --train_data_dir ./kontext_dataset \
    --output_dir ./kontext_model \
    --learning_rate 1e-4 \
    --num_train_epochs 50 \
    --train_batch_size 1 \
    --resolution 1024
```

### Example 4: Fill Model Training

```bash
# Train for inpainting/outpainting
python flux_training_script.py \
    --model_name flux-dev-fill \
    --train_data_dir ./fill_dataset \
    --output_dir ./fill_model \
    --learning_rate 1e-4 \
    --num_train_epochs 30 \
    --train_batch_size 1 \
    --resolution 1024
```

## Advanced Training

### Multi-GPU Training

```bash
# Using torchrun for distributed training
torchrun --nproc_per_node=2 flux_training_script.py \
    --model_name flux-dev \
    --train_data_dir ./dataset \
    --output_dir ./distributed_model \
    --learning_rate 1e-4 \
    --num_train_epochs 100 \
    --train_batch_size 2
```

### Gradient Checkpointing

```bash
# Enable gradient checkpointing for memory efficiency
python flux_training_script.py \
    --model_name flux-dev \
    --train_data_dir ./dataset \
    --output_dir ./checkpointed_model \
    --gradient_checkpointing \
    --learning_rate 1e-4 \
    --num_train_epochs 100
```

### Custom Learning Rate Schedules

```bash
# Use cosine learning rate schedule
python flux_training_script.py \
    --model_name flux-dev \
    --train_data_dir ./dataset \
    --output_dir ./cosine_model \
    --lr_scheduler cosine \
    --lr_warmup_steps 1000 \
    --learning_rate 1e-4 \
    --num_train_epochs 100
```

## Training Configuration

### Model Variants

| Model | Purpose | Input Channels | Guidance |
|-------|---------|----------------|----------|
| `flux-dev` | General text-to-image | 64 | Yes |
| `flux-schnell` | Fast generation | 64 | No |
| `flux-dev-fill` | Inpainting/outpainting | 384 | Yes |
| `flux-dev-kontext` | Context-aware | 64 | Yes |
| `flux-dev-canny` | Edge conditioning | 128 | Yes |
| `flux-dev-depth` | Depth conditioning | 128 | Yes |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-4 | Initial learning rate |
| `train_batch_size` | 1 | Batch size per device |
| `num_train_epochs` | 100 | Number of training epochs |
| `resolution` | 1024 | Input image resolution |
| `checkpointing_steps` | 500 | Save checkpoint every N steps |
| `validation_steps` | 100 | Run validation every N steps |

### LoRA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_r` | 128 | LoRA rank |
| `lora_alpha` | 128 | LoRA alpha |
| `lora_dropout` | 0.0 | LoRA dropout |

## Monitoring Training

### TensorBoard Logging

```bash
# Start TensorBoard
tensorboard --logdir ./trained_model/logs

# View at http://localhost:6006
```

### WandB Integration

```bash
# Login to WandB
wandb login

# Training with WandB logging
python flux_training_script.py \
    --model_name flux-dev \
    --train_data_dir ./dataset \
    --output_dir ./wandb_model \
    --report_to wandb
```

## Model Evaluation

### Validation During Training

The training script automatically runs validation every N steps:

```bash
python flux_training_script.py \
    --model_name flux-dev \
    --train_data_dir ./dataset \
    --output_dir ./model \
    --validation_steps 100 \
    --validation_prompt "a beautiful landscape painting"
```

### Post-Training Evaluation

```python
from src.flux.sampling import denoise, get_noise, prepare
from src.flux.util import load_flow_model, load_ae, load_t5, load_clip

# Load trained model
model = load_flow_model("path/to/trained/model")
ae = load_ae("path/to/trained/model")
t5 = load_t5()
clip = load_clip()

# Generate test images
prompt = "a beautiful landscape painting"
# ... implementation details
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Solution**: Reduce batch size or enable gradient checkpointing

```bash
python flux_training_script.py \
    --train_batch_size 1 \
    --gradient_checkpointing
```

#### 2. Slow Training

**Solution**: Use mixed precision and optimize data loading

```bash
python flux_training_script.py \
    --mixed_precision bf16 \
    --dataloader_num_workers 4
```

#### 3. Poor Convergence

**Solution**: Adjust learning rate and schedule

```bash
python flux_training_script.py \
    --learning_rate 5e-5 \
    --lr_scheduler cosine \
    --lr_warmup_steps 1000
```

### Memory Optimization

```bash
# Enable memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# Use gradient checkpointing
python flux_training_script.py --gradient_checkpointing
```

### Performance Tips

1. **Use SSD storage** for faster data loading
2. **Enable mixed precision** (bf16/fp16) for faster training
3. **Use xformers** for memory-efficient attention
4. **Optimize data loading** with multiple workers
5. **Use gradient accumulation** for larger effective batch sizes

## Advanced Topics

### Custom Loss Functions

You can modify the loss function in the training script:

```python
# Custom loss function
def custom_loss(pred, target, timesteps):
    # Implement your custom loss
    return F.mse_loss(pred, target)
```

### Custom Sampling Schedules

```python
# Custom timestep sampling
def custom_timesteps(batch_size, device):
    # Implement custom timestep sampling
    return torch.linspace(0, 1, 1000, device=device)
```

### Multi-Modal Training

For training with multiple conditioning types:

```python
# Multi-modal training setup
def prepare_multimodal_batch(batch, model_type):
    if model_type == "flux-dev-fill":
        return prepare_fill(...)
    elif model_type == "flux-dev-canny":
        return prepare_control(...)
    else:
        return prepare(...)
```

## Resources

### Official Documentation
- [Flux Model Cards](https://huggingface.co/black-forest-labs)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)

### Community Resources
- [Flux Training Examples](https://github.com/Bilal143260/FLUX.1-Fill-dev-Training)
- [LoRA Training Guide](https://github.com/hvppycoding/hvppyfluxfill)

### Papers and References
- [Flux: A Foundation Model for Text-to-Image Generation](https://arxiv.org/abs/2401.11605)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

## Support

For issues and questions:
1. Check the [troubleshooting section](#troubleshooting)
2. Review the [official documentation](https://huggingface.co/docs/diffusers)
3. Open an issue on the [Flux repository](https://github.com/black-forest-labs/FLUX.1)
4. Join the [Flux community discussions](https://huggingface.co/black-forest-labs/FLUX.1-dev/discussions)