#!/usr/bin/env python3
"""
VGGT Inference Script

Simple script to run inference on example images and visualize results.
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path
import gradio as gr
from PIL import Image
import matplotlib.cm as cm

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


def _compute_patch_grid(H, W, patch_size):
    grid_h = H // patch_size
    grid_w = W // patch_size
    return grid_h, grid_w


def _xy_to_patch_index(x, y, patch_size, grid_w):
    col = int(x) // patch_size
    row = int(y) // patch_size
    return row * grid_w + col


def _extract_query_attention(attn, q_index, head_idx=None):
    # attn shape is either [B, heads, Nq, Nk] or [B, heads, N, N]
    if attn is None:
        return None

    if attn.dim() != 4:
        raise ValueError(f"Unexpected attention tensor shape: {attn.shape}")

    # squeeze batch
    attn = attn[0]

    if head_idx is None:
        # average heads
        # shape: [Nq, Nk]
        return attn.mean(dim=0)[q_index]

    return attn[head_idx, q_index]


def _compute_and_capture_attention(model, images_b, device, dtype, q_patch):
    B, S, _, H, W = images_b.shape
    patch_size = model.aggregator.patch_size
    grid_h, grid_w = _compute_patch_grid(H, W, patch_size)
    P = grid_h * grid_w
    patch_start_idx = model.aggregator.patch_start_idx

    P_total = patch_start_idx + P
    q_global = 0 * P_total + (patch_start_idx + q_patch)

    model.aggregator.enable_global_attention_capture(block_idx=None, query_indices=[q_global])

    with torch.no_grad():
        if dtype != torch.float32:
            with torch.cuda.amp.autocast(dtype=dtype):
                _ = model(images_b.to(device))
        else:
            _ = model(images_b.to(device))

    attn = model.aggregator.get_captured_global_attention()
    model.aggregator.disable_global_attention_capture()

    if attn is None:
        raise RuntimeError("Attention capture returned None.")

    return attn, grid_h, grid_w, patch_start_idx, P


def _attention_heatmaps_from_capture(attn, S, grid_h, grid_w, patch_start_idx, P, head_selection):
    # attn: [B, heads, 1, Nk]
    attn = attn[0]
    if head_selection == "avg":
        vec = attn.mean(dim=0)[0]
    else:
        vec = attn[int(head_selection), 0]

    P_total = patch_start_idx + P
    expected = S * P_total
    if vec.numel() != expected:
        raise ValueError(f"Expected attention length {expected}, got {vec.numel()}")

    vec = vec.detach().cpu().float()
    heatmaps = []
    for t in range(S):
        start = t * P_total + patch_start_idx
        end = start + P
        a = vec[start:end].view(grid_h, grid_w).numpy()
        # normalize for display
        a = a - a.min()
        if a.max() > 0:
            a = a / a.max()
        heatmaps.append(a)
    return heatmaps


def _heatmap_to_rgb_uint8(hm):
    hm = np.asarray(hm)
    hm = np.clip(hm, 0.0, 1.0)
    rgba = cm.get_cmap("magma")(hm)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb


def _upsample_heatmap_to_image(hm, patch_size):
    hm = np.asarray(hm)
    hm_up = np.repeat(np.repeat(hm, patch_size, axis=0), patch_size, axis=1)
    return hm_up


def _overlay_heatmap_on_image(image_rgb_uint8, heat_rgb_uint8, alpha=0.45):
    img = image_rgb_uint8.astype(np.float32)
    heat = heat_rgb_uint8.astype(np.float32)
    out = img * (1.0 - alpha) + heat * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _gradio_attention_ui(model, images, device, dtype, output_dir, validate=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(images.shape) == 4:
        images_b = images.unsqueeze(0)
    else:
        images_b = images

    B, S, _, H, W = images_b.shape
    frame0 = images_b[0, 0].detach().cpu().permute(1, 2, 0).numpy()
    num_heads = model.aggregator.global_blocks[-1].attn.num_heads

    if validate:
        _validate_fused_vs_unfused(model, images, device, dtype)

    head_choices = ["avg"] + [str(i) for i in range(num_heads)]

    def _render_gallery(attn, grid_h, grid_w, patch_start_idx, P, head_sel, q_patch):
        heatmaps = _attention_heatmaps_from_capture(attn, S, grid_h, grid_w, patch_start_idx, P, head_sel)
        imgs = []
        for t, hm in enumerate(heatmaps):
            hm_up = _upsample_heatmap_to_image(hm, model.aggregator.patch_size)
            hm_rgb = _heatmap_to_rgb_uint8(hm_up)

            frame_t = images_b[0, t].detach().cpu().permute(1, 2, 0).numpy()
            frame_t_uint8 = (np.clip(frame_t, 0.0, 1.0) * 255.0).astype(np.uint8)
            overlay = _overlay_heatmap_on_image(frame_t_uint8, hm_rgb, alpha=0.45)

            out = output_dir / f"attn_overlay_f0_p{q_patch}_to_f{t}_head{head_sel}.png"
            Image.fromarray(overlay).save(out)
            imgs.append((overlay, f"frame {t} overlay"))
        return imgs, f"query_patch={q_patch}, head={head_sel}"

    def on_select(evt: gr.SelectData, head_sel, cache):
        x, y = evt.index
        patch_size = model.aggregator.patch_size
        grid_h, grid_w = _compute_patch_grid(H, W, patch_size)
        q_patch = _xy_to_patch_index(x, y, patch_size, grid_w)

        if q_patch < 0 or q_patch >= grid_h * grid_w:
            return gr.update(), "click inside image", cache

        # compute once per patch
        attn, gh, gw, psi, P = _compute_and_capture_attention(model, images_b, device, dtype, q_patch)
        cache = {
            "q_patch": q_patch,
            "attn": attn.cpu(),
            "grid_h": gh,
            "grid_w": gw,
            "patch_start_idx": psi,
            "P": P,
        }
        gallery, status = _render_gallery(attn.cpu(), gh, gw, psi, P, head_sel, q_patch)
        return gallery, status, cache

    def on_head_change(head_sel, cache):
        if cache is None or "attn" not in cache:
            return gr.update(), "click a patch first"
        attn = cache["attn"]
        gallery, status = _render_gallery(
            attn,
            cache["grid_h"],
            cache["grid_w"],
            cache["patch_start_idx"],
            cache["P"],
            head_sel,
            cache["q_patch"],
        )
        return gallery, status

    with gr.Blocks() as demo:
        gr.Markdown("## VGGT Global Attention (last global block)\nClick frame 0 to choose a query patch. Change head to update without recomputing forward.")
        cache = gr.State(None)

        with gr.Row():
            img = gr.Image(value=frame0, label="Frame 0", interactive=True)
            with gr.Column():
                head_sel = gr.Dropdown(choices=head_choices, value="avg", label="Head")
                status = gr.Textbox(value="click a patch", label="Status")

        gallery = gr.Gallery(label="Attention heatmaps (per frame)", columns=4, rows=2, height=420)

        img.select(on_select, inputs=[head_sel, cache], outputs=[gallery, status, cache])
        head_sel.change(on_head_change, inputs=[head_sel, cache], outputs=[gallery, status])

    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=server_name, server_port=server_port)


def _validate_fused_vs_unfused(model, images, device, dtype):
    if len(images.shape) == 4:
        images_b = images.unsqueeze(0)
    else:
        images_b = images

    blk = model.aggregator.global_blocks[-1]
    attn_mod = blk.attn
    original_fused = getattr(attn_mod, "fused_attn", True)

    with torch.no_grad():
        attn_mod.fused_attn = True
        if dtype != torch.float32:
            with torch.cuda.amp.autocast(dtype=dtype):
                pred_fused = model(images_b.to(device))
        else:
            pred_fused = model(images_b.to(device))

        attn_mod.fused_attn = False
        if dtype != torch.float32:
            with torch.cuda.amp.autocast(dtype=dtype):
                pred_unfused = model(images_b.to(device))
        else:
            pred_unfused = model(images_b.to(device))

    # restore
    attn_mod.fused_attn = original_fused

    # Compare a couple of outputs that should be stable
    diffs = {}
    for k in ("pose_enc", "depth", "world_points"):
        if k in pred_fused and k in pred_unfused and isinstance(pred_fused[k], torch.Tensor):
            a = pred_fused[k].detach().float().cpu()
            b = pred_unfused[k].detach().float().cpu()
            diffs[k] = {
                "max_abs": (a - b).abs().max().item(),
                "mean_abs": (a - b).abs().mean().item(),
            }

    print("\nFUSED vs UNFUSED validation (last global block):")
    for k, d in diffs.items():
        print(f"  {k}: max_abs={d['max_abs']:.6e}, mean_abs={d['mean_abs']:.6e}")


def visualize_global_attention(model, images, device, dtype, output_dir, validate=False):
    print("\nStarting global attention UI (Gradio)...")
    _gradio_attention_ui(model, images, device, dtype, output_dir, validate=validate)


def print_results(predictions, image_size_hw=None):
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
        if image_size_hw is not None:
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, image_size_hw=image_size_hw)
            print(f"   Extrinsic matrices: {extrinsic.shape}")
            print(f"   Intrinsic matrices: {intrinsic.shape}")
        else:
            print("   (Skipping extrinsic/intrinsic extraction without image dimensions)")
    
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
        default="examples/kitchen/images",
        help="Path to directory containing images (e.g. examples/room/images or examples/kitchen/images)"
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

    parser.add_argument(
        "--attn-ui",
        action="store_true",
        help="Open an interactive UI (Gradio) to visualize global attention from the last global block"
    )

    parser.add_argument(
        "--attn-gradio",
        action="store_true",
        help="Alias for --attn-ui"
    )

    parser.add_argument(
        "--attn-validate",
        action="store_true",
        default=False,
        help="Validate fused vs unfused attention equivalence before opening the UI"
    )

    parser.add_argument(
        "--attn-output",
        type=str,
        default="attention_viz",
        help="Output directory for attention visualizations"
    )
    
    args = parser.parse_args()
    
    # Setup
    device, dtype = setup_device()
    model = load_model(device)
    
    # Load images
    image_dir = Path(args.images)
    if image_dir.is_dir():
        image_paths = sorted(
            list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")) +
            list(image_dir.glob("images/*.png")) + list(image_dir.glob("images/*.jpg"))
        )
        if not image_paths:
            print(f"No images found in {image_dir}")
            return
        print(f"Found {len(image_paths)} images in {image_dir}")
    else:
        image_paths = [Path(p) for p in args.images.split()]
    
    # Convert to strings for load_and_preprocess_images
    image_paths = [str(p) for p in image_paths]
    
    images = load_images(image_paths)
    images = images.to(device)
    
    # Extract image dimensions (H, W) from the loaded images
    # images shape: (N, 3, H, W)
    image_size_hw = (images.shape[2], images.shape[3])
    print(f"Image size (H, W): {image_size_hw}")
    
    # # Run inference
    # predictions = run_inference(model, images, device, dtype)
    
    # # Print results
    # print_results(predictions, image_size_hw=image_size_hw)
    
    # # Save predictions if requested
    # if args.save:
    #     save_predictions(predictions, args.output)

    if args.attn_ui or args.attn_gradio:
        visualize_global_attention(model, images, device, dtype, args.attn_output, validate=args.attn_validate)


if __name__ == "__main__":
    main()
