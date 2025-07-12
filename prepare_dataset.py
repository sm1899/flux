#!/usr/bin/env python3
"""
Dataset Preparation Script for Flux Training
This script helps prepare datasets for training Flux models.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple

from PIL import Image
import torchvision.transforms as transforms


def create_metadata_from_directory(
    dataset_path: str,
    output_file: str = "metadata.jsonl",
    caption_prefix: str = "",
    caption_suffix: str = "",
    use_filename_as_caption: bool = True,
    custom_captions: Dict[str, str] = None,
) -> None:
    """
    Create metadata.jsonl from a directory of images.
    
    Args:
        dataset_path: Path to directory containing images
        output_file: Output metadata file path
        caption_prefix: Prefix to add to all captions
        caption_suffix: Suffix to add to all captions
        use_filename_as_caption: Whether to use filename as caption
        custom_captions: Dictionary mapping filenames to custom captions
    """
    metadata = []
    dataset_path = Path(dataset_path)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    
    # Get all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(dataset_path.glob(f"*{ext}"))
        image_files.extend(dataset_path.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} image files in {dataset_path}")
    
    for image_file in image_files:
        filename = image_file.name
        
        # Generate caption
        if custom_captions and filename in custom_captions:
            caption = custom_captions[filename]
        elif use_filename_as_caption:
            # Convert filename to caption
            caption = filename.replace('_', ' ').replace('-', ' ')
            caption = caption.replace('.jpg', '').replace('.png', '').replace('.webp', '')
            caption = caption.replace('.jpeg', '').replace('.bmp', '').replace('.tiff', '')
            caption = caption.strip()
        else:
            caption = "an image"  # Default caption
        
        # Add prefix and suffix
        caption = f"{caption_prefix}{caption}{caption_suffix}".strip()
        
        metadata.append({
            "file_name": filename,
            "text": caption
        })
    
    # Write metadata file
    with open(output_file, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')
    
    print(f"Created metadata file: {output_file}")
    print(f"Total entries: {len(metadata)}")


def validate_dataset(dataset_path: str, metadata_file: str = "metadata.jsonl") -> Tuple[bool, List[str]]:
    """
    Validate that all images referenced in metadata exist and are valid.
    
    Args:
        dataset_path: Path to dataset directory
        metadata_file: Path to metadata file
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    dataset_path = Path(dataset_path)
    metadata_path = Path(metadata_file)
    
    if not metadata_path.exists():
        return False, [f"Metadata file not found: {metadata_file}"]
    
    if not dataset_path.exists():
        return False, [f"Dataset directory not found: {dataset_path}"]
    
    # Read metadata
    with open(metadata_file, 'r') as f:
        metadata = [json.loads(line) for line in f]
    
    # Check each entry
    for i, entry in enumerate(metadata):
        if 'file_name' not in entry:
            errors.append(f"Entry {i}: Missing 'file_name' field")
            continue
        
        if 'text' not in entry:
            errors.append(f"Entry {i}: Missing 'text' field")
            continue
        
        image_path = dataset_path / entry['file_name']
        if not image_path.exists():
            errors.append(f"Entry {i}: Image file not found: {entry['file_name']}")
            continue
        
        # Try to open image to validate it
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception as e:
            errors.append(f"Entry {i}: Invalid image file {entry['file_name']}: {str(e)}")
    
    return len(errors) == 0, errors


def split_dataset(
    dataset_path: str,
    metadata_file: str = "metadata.jsonl",
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    output_dir: str = "split_dataset"
) -> None:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        dataset_path: Path to dataset directory
        metadata_file: Path to metadata file
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        output_dir: Output directory for split datasets
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Read metadata
    with open(metadata_file, 'r') as f:
        metadata = [json.loads(line) for line in f]
    
    # Shuffle metadata
    random.shuffle(metadata)
    
    # Calculate split indices
    total_samples = len(metadata)
    train_end = int(total_samples * train_ratio)
    val_end = train_end + int(total_samples * val_ratio)
    
    # Split metadata
    train_metadata = metadata[:train_end]
    val_metadata = metadata[train_end:val_end]
    test_metadata = metadata[val_end:]
    
    # Create output directories
    output_path = Path(output_dir)
    train_path = output_path / "train"
    val_path = output_path / "val"
    test_path = output_path / "test"
    
    for path in [train_path, val_path, test_path]:
        path.mkdir(parents=True, exist_ok=True)
    
    # Copy files and create metadata
    dataset_path = Path(dataset_path)
    
    def copy_split(split_metadata, split_path, split_name):
        for entry in split_metadata:
            src_file = dataset_path / entry['file_name']
            dst_file = split_path / entry['file_name']
            
            if src_file.exists():
                # Copy image file
                import shutil
                shutil.copy2(src_file, dst_file)
        
        # Write metadata
        metadata_file = split_path / "metadata.jsonl"
        with open(metadata_file, 'w') as f:
            for entry in split_metadata:
                f.write(json.dumps(entry) + '\n')
        
        print(f"{split_name}: {len(split_metadata)} samples")
    
    copy_split(train_metadata, train_path, "Train")
    copy_split(val_metadata, val_path, "Validation")
    copy_split(test_metadata, test_path, "Test")
    
    print(f"Dataset split completed. Output directory: {output_dir}")


def create_sample_dataset(output_dir: str = "sample_dataset", num_samples: int = 10) -> None:
    """
    Create a sample dataset for testing.
    
    Args:
        output_dir: Output directory for sample dataset
        num_samples: Number of sample images to create
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample captions
    captions = [
        "a beautiful landscape with mountains",
        "a portrait of a woman in traditional dress",
        "a futuristic city skyline at sunset",
        "a cute cat sitting on a windowsill",
        "a vintage car parked on a street",
        "a colorful abstract painting",
        "a serene lake with trees",
        "a modern office building",
        "a flower garden in spring",
        "a cozy coffee shop interior"
    ]
    
    # Create sample images (simple colored rectangles)
    for i in range(min(num_samples, len(captions))):
        # Create a simple colored image
        img = Image.new('RGB', (512, 512), color=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ))
        
        filename = f"sample_{i:03d}.jpg"
        img.save(output_path / filename)
        
        # Create metadata entry
        metadata_entry = {
            "file_name": filename,
            "text": captions[i]
        }
        
        # Append to metadata file
        metadata_file = output_path / "metadata.jsonl"
        with open(metadata_file, 'a') as f:
            f.write(json.dumps(metadata_entry) + '\n')
    
    print(f"Created sample dataset with {num_samples} images in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Dataset Preparation for Flux Training")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Create metadata command
    create_parser = subparsers.add_parser('create', help='Create metadata from directory')
    create_parser.add_argument('dataset_path', help='Path to dataset directory')
    create_parser.add_argument('--output', default='metadata.jsonl', help='Output metadata file')
    create_parser.add_argument('--prefix', default='', help='Caption prefix')
    create_parser.add_argument('--suffix', default='', help='Caption suffix')
    create_parser.add_argument('--no-filename-caption', action='store_true', help='Don\'t use filename as caption')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate dataset')
    validate_parser.add_argument('dataset_path', help='Path to dataset directory')
    validate_parser.add_argument('--metadata', default='metadata.jsonl', help='Metadata file path')
    
    # Split command
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val/test')
    split_parser.add_argument('dataset_path', help='Path to dataset directory')
    split_parser.add_argument('--metadata', default='metadata.jsonl', help='Metadata file path')
    split_parser.add_argument('--output', default='split_dataset', help='Output directory')
    split_parser.add_argument('--train-ratio', type=float, default=0.8, help='Training set ratio')
    split_parser.add_argument('--val-ratio', type=float, default=0.1, help='Validation set ratio')
    split_parser.add_argument('--test-ratio', type=float, default=0.1, help='Test set ratio')
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Create sample dataset')
    sample_parser.add_argument('--output', default='sample_dataset', help='Output directory')
    sample_parser.add_argument('--num-samples', type=int, default=10, help='Number of sample images')
    
    args = parser.parse_args()
    
    if args.command == 'create':
        create_metadata_from_directory(
            dataset_path=args.dataset_path,
            output_file=args.output,
            caption_prefix=args.prefix,
            caption_suffix=args.suffix,
            use_filename_as_caption=not args.no_filename_caption
        )
    
    elif args.command == 'validate':
        is_valid, errors = validate_dataset(args.dataset_path, args.metadata)
        if is_valid:
            print("✅ Dataset is valid!")
        else:
            print("❌ Dataset validation failed:")
            for error in errors:
                print(f"  - {error}")
    
    elif args.command == 'split':
        split_dataset(
            dataset_path=args.dataset_path,
            metadata_file=args.metadata,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            output_dir=args.output
        )
    
    elif args.command == 'sample':
        create_sample_dataset(args.output, args.num_samples)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()