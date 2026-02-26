#!/usr/bin/env python3
"""
VGGT Inference Script

Simple script to run inference on example images and visualize results.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map


def setup_device():
    """Setup torch device and dtype for inference."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Use bfloat16 on Ampere GPUs (RTX 30XX, A100, etc.)
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    
    print(f"Using dtype: {dtype}")
    return device, dtype


def load_model(device):
    """Load VGGT pretrained model."""
    print("\nLoading VGGT model...")
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    print("Model loaded successfully!")
    return model


def load_images(image_paths):
    """Load and preprocess images."""
    print(f"\nLoading {len(image_paths)} images...")
    for path in image_paths:
        print(f"  - {path}")
    
    images = load_and_preprocess_images(image_paths)
    print(f"Images loaded: {images.shape}")
    return images


def run_inference(model, images, device, dtype):
    """Run inference on images."""
    print("\nRunning inference...")
    images = images.to(device)
    
    with torch.no_grad():
        if dtype != torch.float32:
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(images)
        else:
            predictions = model(images)
    
    return predictions


def print_results(predictions):
    """Print prediction statistics."""
    print("\n" + "="*80)
    print("INFERENCE RESULTS")
    print("="*80)
    
    if "pose_enc" in predictions:
        pose_enc = predictions["pose_enc"]
        print(f"\n📷 Camera Pose:")
        print(f"   Shape: {pose_enc.shape}")
        print(f"   Range: [{pose_enc.min():.4f}, {pose_enc.max():.4f}]")
        
        # Extract camera parameters
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc)
        print(f"   Extrinsic matrices: {extrinsic.shape}")
        print(f"   Intrinsic matrices: {intrinsic.shape}")
    
    if "depth" in predictions:
        depth = predictions["depth"]
        depth_conf = predictions.get("depth_conf")
        print(f"\n📊 Depth Maps:")
        print(f"   Shape: {depth.shape}")
        print(f"   Range: [{depth[depth > 0].min():.4f}, {depth.max():.4f}]")
        if depth_conf is not None:
            print(f"   Confidence range: [{depth_conf.min():.4f}, {depth_conf.max():.4f}]")
    
    if "world_points" in predictions:
        world_points = predictions["world_points"]
        world_conf = predictions.get("world_points_conf")
        print(f"\n🌍 3D World Points:")
        print(f"   Shape: {world_points.shape}")
        print(f"   X range: [{world_points[..., 0].min():.4f}, {world_points[..., 0].max():.4f}]")
        print(f"   Y range: [{world_points[..., 1].min():.4f}, {world_points[..., 1].max():.4f}]")
        print(f"   Z range: [{world_points[..., 2].min():.4f}, {world_points[..., 2].max():.4f}]")
        if world_conf is not None:
            print(f"   Confidence range: [{world_conf.min():.4f}, {world_conf.max():.4f}]")
    
    print("\n" + "="*80 + "\n")


def save_predictions(predictions, output_dir):
    """Save predictions to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving predictions to {output_dir}...")
    
    for key, value in predictions.items():
        if key == "images":
            continue  # Skip images
        
        if isinstance(value, torch.Tensor):
            # Save as numpy
            filename = output_dir / f"{key}.npy"
            np.save(filename, value.cpu().numpy())
            print(f"  ✓ Saved {key} to {filename}")
        elif isinstance(value, list):
            # Save list of tensors
            filename = output_dir / f"{key}_list.npz"
            list_np = [v.cpu().numpy() for v in value]
            np.savez(filename, *list_np)
            print(f"  ✓ Saved {key} list to {filename}")


def main():
    parser = argparse.ArgumentParser(description="VGGT Inference Script")
    parser.add_argument(
        "--images",
        type=str,
        default="examples/kitchen",
        help="Path to directory containing images or image files (space-separated)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="inference_results",
        help="Output directory for predictions"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save predictions to disk"
    )
    
    args = parser.parse_args()
    
    # Setup
    device, dtype = setup_device()
    model = load_model(device)
    
    # Load images
    image_dir = Path(args.images)
    if image_dir.is_dir():
        image_paths = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
        if not image_paths:
            print(f"❌ No images found in {image_dir}")
            return
    else:
        image_paths = [Path(p) for p in args.images.split()]
    
    # Convert to strings for load_and_preprocess_images
    image_paths = [str(p) for p in image_paths]
    
    images = load_images(image_paths)
    images = images.to(device)
    
    # Run inference
    predictions = run_inference(model, images, device, dtype)
    
    # Print results
    print_results(predictions)
    
    # Save predictions if requested
    if args.save:
        save_predictions(predictions, args.output)


if __name__ == "__main__":
    main()
