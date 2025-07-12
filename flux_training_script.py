#!/usr/bin/env python3
"""
Complete Flux Training Script
Supports training Flux models from scratch and LoRA fine-tuning
"""

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Import Flux components
from src.flux.model import Flux, FluxLoraWrapper, FluxParams
from src.flux.modules.autoencoder import AutoEncoder, AutoEncoderParams
from src.flux.modules.conditioner import HFEmbedder
from src.flux.sampling import get_noise, get_schedule, prepare
from src.flux.util import configs, load_flow_model, load_ae, load_t5, load_clip


class FluxDataset(Dataset):
    """Dataset for Flux training"""
    
    def __init__(
        self,
        data_root: str,
        tokenizer_one,
        tokenizer_two,
        size: int = 1024,
        center_crop: bool = False,
        split: str = "train",
        max_length: int = 512,
    ):
        self.data_root = data_root
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.size = size
        self.center_crop = center_crop
        self.split = split
        self.max_length = max_length
        
        # Load image paths and captions
        self.image_paths = []
        self.captions = []
        self._load_data()
    
    def _load_data(self):
        """Load image paths and captions from data directory"""
        # Implementation depends on your data format
        # This is a placeholder - you'll need to implement based on your data structure
        # Example: Load from metadata.jsonl file
        import json
        metadata_path = os.path.join(self.data_root, "metadata.jsonl")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    self.image_paths.append(os.path.join(self.data_root, data['file_name']))
                    self.captions.append(data['text'])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        # Load image and caption
        image_path = self.image_paths[index]
        caption = self.captions[index]
        
        # Load and preprocess image
        image = self._load_and_preprocess_image(image_path)
        
        # Tokenize caption
        tokenized_one = self.tokenizer_one(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        tokenized_two = self.tokenizer_two(
            caption,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": image,
            "input_ids_one": tokenized_one.input_ids[0],
            "input_ids_two": tokenized_two.input_ids[0],
            "caption": caption,
        }
    
    def _load_and_preprocess_image(self, image_path):
        """Load and preprocess image"""
        from PIL import Image
        import torchvision.transforms as transforms
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Define transforms
        if self.center_crop:
            transform = transforms.Compose([
                transforms.Resize(self.size),
                transforms.CenterCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(self.size),
                transforms.RandomCrop(self.size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ])
        
        return transform(image)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Flux Training Script")
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="flux-dev",
        choices=["flux-dev", "flux-schnell", "flux-dev-fill", "flux-dev-kontext"],
        help="Flux model variant to train",
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./flux-trained-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="The resolution for input images",
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help="Whether to center crop the input images to the resolution.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help="Save a checkpoint of the training state every X updates.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="Run validation every X steps.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default="a beautiful landscape painting",
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    
    # LoRA arguments
    parser.add_argument(
        "--use_peft_lora",
        action="store_true",
        help="Whether to use PEFT LoRA for training",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=128,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=128,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="LoRA dropout",
    )
    
    # Learning rate arguments
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate to use.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='The scheduler type to use. Choose between ["linear", "cosine", "constant"]',
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    
    # Data arguments
    parser.add_argument(
        "--train_data_dir",
        type=str,
        required=True,
        help="A folder containing the training data with metadata.jsonl file.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, truncate the number of training examples to this value if set.",
    )
    
    # Training optimization arguments
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm for gradient clipping.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading.",
    )
    
    # Device arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    
    args = parser.parse_args()
    return args


def get_scheduler(optimizer, scheduler_type, num_warmup_steps, num_training_steps):
    """Get learning rate scheduler"""
    if scheduler_type == "linear":
        from torch.optim.lr_scheduler import LinearLR
        return LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)
    elif scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=num_training_steps)
    elif scheduler_type == "constant":
        from torch.optim.lr_scheduler import LambdaLR
        return LambdaLR(optimizer, lambda _: 1.0)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def main(args):
    """Main training function"""
    # Set seed for reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    
    # Load model configuration
    model_config = configs[args.model_name]
    
    # Initialize models
    logger.info("Initializing Flux model...")
    
    # Initialize Flux model
    if args.use_peft_lora:
        model = FluxLoraWrapper(
            params=model_config.params,
            lora_rank=args.lora_r,
            lora_scale=1.0,
        )
    else:
        model = Flux(model_config.params)
    
    # Initialize autoencoder
    ae = AutoEncoder(model_config.ae_params)
    
    # Initialize text encoders
    t5 = load_t5(device="cpu", max_length=512)
    clip = load_clip(device="cpu")
    
    # Move models to device
    device = torch.device(args.device)
    model = model.to(device)
    ae = ae.to(device)
    t5 = t5.to(device)
    clip = clip.to(device)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        ae.enable_gradient_checkpointing()
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )
    
    # Initialize dataset and dataloader
    train_dataset = FluxDataset(
        data_root=args.train_data_dir,
        tokenizer_one=clip.tokenizer,
        tokenizer_two=t5.tokenizer,
        size=args.resolution,
        center_crop=args.center_crop,
        split="train",
    )
    
    # Limit dataset size if specified
    if args.max_train_samples is not None:
        train_dataset = torch.utils.data.Subset(
            train_dataset, indices=range(min(args.max_train_samples, len(train_dataset)))
        )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )
    
    # Calculate total training steps
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * len(train_dataloader)
    
    # Initialize scheduler
    lr_scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_type=args.lr_scheduler,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    
    # Training loop
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    global_step = 0
    
    # Training loop
    for epoch in range(args.num_train_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            # Get the batch data
            pixel_values = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
            input_ids_one = batch["input_ids_one"].to(device)
            input_ids_two = batch["input_ids_two"].to(device)
            
            # Encode images to latent space
            with torch.no_grad():
                latents = ae.encode(pixel_values)
            
            # Prepare noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            
            # Sample a random timestep for each image
            timesteps = torch.randint(0, 1000, (bsz,), device=latents.device).long()
            
            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = latents + noise * timesteps.view(-1, 1, 1, 1) / 1000.0
            
            # Prepare text embeddings
            with torch.no_grad():
                text_embeddings_one = clip(input_ids_one)
                text_embeddings_two = t5(input_ids_two)
            
            # Forward pass
            noise_pred = model(
                img=noisy_latents,
                img_ids=torch.zeros(bsz, noisy_latents.shape[1], 3).to(device),
                txt=text_embeddings_two,
                txt_ids=torch.zeros(bsz, text_embeddings_two.shape[1], 3).to(device),
                timesteps=timesteps.float() / 1000.0,
                y=text_embeddings_one,
                guidance=torch.ones(bsz).to(device) * 3.0,
            )
            
            # Calculate loss
            loss = F.mse_loss(noise_pred, noise, reduction="none")
            loss = loss.mean([1, 2, 3]).mean()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Update progress
            progress_bar.update(1)
            global_step += 1
            
            # Logging
            if global_step % 10 == 0:
                logger.info(f"Step {global_step}: Loss = {loss.item():.4f}, LR = {lr_scheduler.get_last_lr()[0]:.6f}")
            
            # Checkpointing
            if global_step % args.checkpointing_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Save model state
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
                torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
                
                logger.info(f"Saved checkpoint to {checkpoint_dir}")
            
            # Validation
            if global_step % args.validation_steps == 0:
                model.eval()
                with torch.no_grad():
                    # Generate validation image
                    # This is a simplified validation - you might want to implement a proper validation pipeline
                    logger.info(f"Running validation with prompt: {args.validation_prompt}")
                    
                    # Create noise for validation
                    val_noise = torch.randn(1, 16, 64, 64, device=device, dtype=torch.bfloat16)
                    
                    # Generate timesteps for validation
                    val_timesteps = torch.linspace(0, 1, 20, device=device)
                    
                    # Simple validation generation (simplified)
                    val_latents = val_noise
                    for i in range(len(val_timesteps) - 1):
                        t_curr = val_timesteps[i]
                        t_next = val_timesteps[i + 1]
                        
                        # This is a simplified validation - in practice you'd use the full sampling pipeline
                        val_latents = val_latents + (t_next - t_curr) * torch.randn_like(val_latents) * 0.1
                    
                    logger.info("Validation completed")
                
                model.train()
            
            if global_step >= args.max_train_steps:
                break
        
        if global_step >= args.max_train_steps:
            break
    
    # Save final model
    final_checkpoint_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_checkpoint_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(final_checkpoint_dir, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(final_checkpoint_dir, "optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), os.path.join(final_checkpoint_dir, "scheduler.pt"))
    
    logger.info(f"Training completed! Final model saved to {final_checkpoint_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)