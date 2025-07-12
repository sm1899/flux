# FLUX Training Resources & Guide

This comprehensive guide consolidates all training resources, documentation, and examples for training FLUX models.

## 📚 Table of Contents

1. [Overview](#overview)
2. [Available Resources](#available-resources)
3. [Model Architecture](#model-architecture)
4. [Training Setup](#training-setup)
5. [Data Preparation](#data-preparation)
6. [Training Scripts](#training-scripts)
7. [Training Examples](#training-examples)
8. [Advanced Training](#advanced-training)
9. [Model Variants](#model-variants)
10. [Troubleshooting](#troubleshooting)
11. [External Resources](#external-resources)

## 🎯 Overview

FLUX is a state-of-the-art text-to-image diffusion model using a dual-stream transformer architecture. This guide covers:

- **Full Model Training**: Training FLUX models from scratch
- **LoRA Fine-tuning**: Efficient fine-tuning using Low-Rank Adaptation
- **Specialized Models**: Training for specific tasks (fill, control, kontext, etc.)
- **Multi-GPU Training**: Distributed training across multiple GPUs

## 📖 Available Resources

### Official Documentation

| Resource | Description | Link |
|----------|-------------|------|
| **Main README** | Installation and basic usage | [README.md](README.md) |
| **Training Guide** | Comprehensive training instructions | [FLUX_TRAINING_GUIDE.md](FLUX_TRAINING_GUIDE.md) |
| **Training Script** | Complete training implementation | [flux_training_script.py](flux_training_script.py) |
| **Dataset Preparation** | Data preprocessing utilities | [prepare_dataset.py](prepare_dataset.py) |

### Model Documentation

| Model | Documentation | HuggingFace |
|-------|---------------|-------------|
| FLUX.1-dev | [Model Card](model_cards/FLUX.1-dev.md) | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| FLUX.1-kontext-dev | [Model Card](model_cards/FLUX.1-kontext-dev.md) | [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) |
| FLUX.1-schnell | [Model Card](model_cards/FLUX.1-schnell.md) | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) |

### Usage Documentation

| Feature | Documentation |
|---------|---------------|
| Text-to-Image | [docs/text-to-image.md](docs/text-to-image.md) |
| Fill/Inpainting | [docs/fill.md](docs/fill.md) |
| Structural Conditioning | [docs/structural-conditioning.md](docs/structural-conditioning.md) |
| Image Variation | [docs/image-variation.md](docs/image-variation.md) |
| Image Editing | [docs/image-editing.md](docs/image-editing.md) |

## 🏗️ Model Architecture

### Core Components

FLUX uses a dual-stream transformer architecture with the following key components:

- **Flow Matching**: State-of-the-art training method
- **Dual-Stream Transformer**: Handles both text and image conditioning
- **Autoencoder**: Compresses images to latent space
- **Text Encoder**: T5-based text understanding
- **CLIP**: Image-text alignment

### Model Variants Comparison

| Model | Purpose | Input Channels | Conditioning | Use Case |
|-------|---------|----------------|-------------|----------|
| `flux-dev` | General text-to-image | 64 | Text | Standard generation |
| `flux-schnell` | Fast generation | 64 | Text | Quick inference |
| `flux-dev-fill` | Inpainting/outpainting | 384 | Text + Mask | Image editing |
| `flux-dev-kontext` | Context-aware | 64 | Text + Reference | Contextual generation |
| `flux-dev-canny` | Edge conditioning | 128 | Text + Edges | Structural control |
| `flux-dev-depth` | Depth conditioning | 128 | Text + Depth | 3D-aware generation |

## ⚙️ Training Setup

### System Requirements

- **GPU**: NVIDIA GPU with at least 24GB VRAM (for full training)
- **RAM**: 64GB+ system RAM
- **Storage**: 500GB+ SSD for datasets and checkpoints
- **Python**: 3.10+

### Installation

```bash
# Clone the repository
git clone https://github.com/black-forest-labs/flux
cd flux

# Create virtual environment
python3.10 -m venv .venv
source .venv/bin/activate

# Install with all dependencies
pip install -e ".[all]"

# For TensorRT support
pip install -e ".[tensorrt]" --extra-index-url https://pypi.nvidia.com
```

### Environment Variables

```bash
# Set up environment variables
export HF_TOKEN="your_huggingface_token"
export CUDA_VISIBLE_DEVICES=0  # Specify GPU
export BFL_API_KEY="your_api_key_here"  # For commercial usage tracking
```

## 📊 Data Preparation

### Dataset Structure

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

### Data Preprocessing

Use the provided `prepare_dataset.py` script:

```bash
python prepare_dataset.py \
    --input_dir ./raw_images \
    --output_dir ./processed_dataset \
    --metadata_file metadata.jsonl \
    --resolution 1024 \
    --validation_split 0.1
```

## 🚀 Training Scripts

### Main Training Script

The `flux_training_script.py` supports:

- **Full model training** from scratch
- **LoRA fine-tuning** for efficient training
- **Multiple model variants** (flux-dev, flux-schnell, etc.)
- **Checkpointing and validation**
- **Multi-GPU training**
- **Gradient checkpointing**

### Basic Training Command

```bash
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

### LoRA Fine-tuning

```bash
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

## 📝 Training Examples

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

### Example 2: Kontext Model Training

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

### Example 3: Fill Model Training

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

### Example 4: Control Model Training

```bash
# Train for structural conditioning
python flux_training_script.py \
    --model_name flux-dev-canny \
    --train_data_dir ./control_dataset \
    --output_dir ./control_model \
    --learning_rate 1e-4 \
    --num_train_epochs 30 \
    --train_batch_size 1
```

## 🔧 Advanced Training

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

### Memory Optimization

```bash
# Enable memory optimizations
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# Use gradient checkpointing
python flux_training_script.py --gradient_checkpointing
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

## 📊 Training Configuration

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

## 📈 Monitoring Training

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

## 🔍 Model Evaluation

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

## 🛠️ Troubleshooting

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

### Performance Tips

1. **Use SSD storage** for faster data loading
2. **Enable mixed precision** (bf16/fp16) for faster training
3. **Use xformers** for memory-efficient attention
4. **Optimize data loading** with multiple workers
5. **Use gradient accumulation** for larger effective batch sizes

## 🌐 External Resources

### Official Resources

| Resource | Description | Link |
|----------|-------------|------|
| **API Documentation** | Official BFL API docs | [docs.bfl.ai](https://docs.bfl.ai/) |
| **Commercial Licensing** | Licensing for commercial use | [bfl.ai/pricing/licensing](https://bfl.ai/pricing/licensing) |
| **Helpdesk** | Support and licensing info | [help.bfl.ai](https://help.bfl.ai/collections/6939000511-licensing) |

### Community Resources

| Resource | Description | Link |
|----------|-------------|------|
| **Flux Training Examples** | Community training examples | [github.com/Bilal143260/FLUX.1-Fill-dev-Training](https://github.com/Bilal143260/FLUX.1-Fill-dev-Training) |
| **LoRA Training Guide** | LoRA fine-tuning guide | [github.com/hvppycoding/hvppyfluxfill](https://github.com/hvppycoding/hvppyfluxfill) |

### Papers and References

| Paper | Description | Link |
|-------|-------------|------|
| **FLUX.1 Kontext Paper** | Flow Matching for In-Context Image Generation | [arxiv.org/abs/2506.15742](https://arxiv.org/abs/2506.15742) |
| **LoRA Paper** | Low-Rank Adaptation of Large Language Models | [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685) |

### HuggingFace Models

| Model | Repository | License |
|-------|------------|---------|
| FLUX.1-dev | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | Non-Commercial |
| FLUX.1-schnell | [black-forest-labs/FLUX.1-schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell) | Apache-2.0 |
| FLUX.1-Fill-dev | [black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) | Non-Commercial |
| FLUX.1-Canny-dev | [black-forest-labs/FLUX.1-Canny-dev](https://huggingface.co/black-forest-labs/FLUX.1-Canny-dev) | Non-Commercial |
| FLUX.1-Depth-dev | [black-forest-labs/FLUX.1-Depth-dev](https://huggingface.co/black-forest-labs/FLUX.1-Depth-dev) | Non-Commercial |
| FLUX.1-Kontext-dev | [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) | Non-Commercial |

## 📞 Support

For issues and questions:

1. **Check the troubleshooting section** above
2. **Review the official documentation** at [docs.bfl.ai](https://docs.bfl.ai/)
3. **Open an issue** on the [Flux repository](https://github.com/black-forest-labs/flux)
4. **Join community discussions** on [HuggingFace](https://huggingface.co/black-forest-labs/FLUX.1-dev/discussions)
5. **Contact BFL support** for commercial licensing questions

## 📄 Citation

If you find the provided code or models useful for your research, consider citing them as:

```bibtex
@misc{labs2025flux1kontextflowmatching,
      title={FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space},
      author={Black Forest Labs and Stephen Batifol and Andreas Blattmann and Frederic Boesel and Saksham Consul and Cyril Diagne and Tim Dockhorn and Jack English and Zion English and Patrick Esser and Sumith Kulal and Kyle Lacey and Yam Levi and Cheng Li and Dominik Lorenz and Jonas Müller and Dustin Podell and Robin Rombach and Harry Saini and Axel Sauer and Luke Smith},
      year={2025},
      eprint={2506.15742},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2506.15742},
}

@misc{flux2024,
    author={Black Forest Labs},
    title={FLUX},
    year={2024},
    howpublished={\url{https://github.com/black-forest-labs/flux}},
}
```

---

**Note**: This guide consolidates all available training resources for FLUX models. For the most up-to-date information, always refer to the official documentation and repository.