# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

# Set headless mode environment variables
os.environ['DISPLAY'] = ''
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

import torch
import numpy as np
import gradio as gr
import shutil
from datetime import datetime
import glob
import gc
import time
from pathlib import Path

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

print("\n" + "="*60)
print("VGGT Gradio App - Initialization Starting")
print("="*60)
print(f"Device: {device}")
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n[1/4] Initializing VGGT model architecture...")
model = VGGT()
print("      ✓ Model architecture initialized")

print("[2/4] Loading model weights...")
# Load from cache to avoid hanging on HuggingFace download
cached_model_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/model.pt")
if os.path.exists(cached_model_path):
    print(f"      Loading cached model from {cached_model_path}")
    model_state = torch.load(cached_model_path, map_location="cpu")
else:
    print("      Downloading model from HuggingFace (first time)...")
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model_state = torch.hub.load_state_dict_from_url(_URL)

print("      ✓ Weights loaded, setting model state...")
model.load_state_dict(model_state)
print("      ✓ Model state set")

print("      ✓ Setting model to eval mode and moving to device...")
model.eval()
model = model.to(device)
print(f"      ✓ Model on device: {device}")

# Global variable to store matching GT data
print("[3/4] Loading sparse attention correspondence masks...")
MATCHING_GT = None
try:
    # Try multiple possible locations for matching_gt
    possible_paths = [
        "notebooks/matching_data/matching_gt.npy",
        "/scratch/cvlab/home/pilligua/mVGGT/notebooks/matching_data/matching_gt.npy",
        "/scratch/cvlab/home/pilligua/mVGGT/notebooks/matching_data/matching_gt.npy",
    ]
    
    for matching_gt_path in possible_paths:
        if os.path.exists(matching_gt_path):
            MATCHING_GT = torch.from_numpy(np.load(matching_gt_path)).bool().to(device)
            print(f"      ✓ Loaded matching GT from {matching_gt_path}")
            print(f"        Shape: {MATCHING_GT.shape}")
            break
    
    if MATCHING_GT is None:
        print(f"      ⚠ Matching GT not found - sparse attention comparison disabled")
except Exception as e:
    print(f"      ⚠ Could not load matching GT: {e}")


# -------------------------------------------------------------------------
# 1) Core model inference
# -------------------------------------------------------------------------
def run_model(target_dir, model) -> dict:
    """
    Run the VGGT model on images in the 'target_dir/images' folder and return predictions.
    """
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    # Run inference
    print("Running inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # Convert pose encoding to extrinsic and intrinsic matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension
    predictions['pose_enc_list'] = None # remove pose_enc_list

    # Generate world points from depth map
    print("Computing world points from depth map...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # Clean up
    torch.cuda.empty_cache()
    return predictions


# -------------------------------------------------------------------------
# 1b) Sparse vs Full Attention Comparison
# -------------------------------------------------------------------------
def run_model_comparison(target_dir, model, compare_sparse=True) -> tuple:
    """
    Run VGGT inference with both full and sparse attention, measuring timing and memory.
    Returns (full_attn_predictions, sparse_attn_predictions, timing_info).
    """
    print(f"\n{'='*80}")
    print("STARTING FULL VS SPARSE ATTENTION COMPARISON")
    print(f"{'='*80}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")
    
    images = load_and_preprocess_images(image_names).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    timing_info = {}

    # ---- FULL ATTENTION ----
    print("[1/2] Running FULL ATTENTION inference...")
    print("-" * 80)
    
    model.disable_sparse_attention()
    torch.cuda.empty_cache() if device == "cuda" else None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    with torch.no_grad():
        start_time = time.time()
        with torch.cuda.amp.autocast(dtype=dtype):
            full_predictions = model(images)
        full_time = time.time() - start_time
    
    full_memory = 0
    if device == "cuda":
        full_memory = torch.cuda.max_memory_allocated() / 1e9
    
    # Post-process full attention results
    extrinsic_full, intrinsic_full = pose_encoding_to_extri_intri(full_predictions["pose_enc"], images.shape[-2:])
    full_predictions["extrinsic"] = extrinsic_full
    full_predictions["intrinsic"] = intrinsic_full
    
    for key in full_predictions.keys():
        if isinstance(full_predictions[key], torch.Tensor):
            full_predictions[key] = full_predictions[key].cpu().numpy().squeeze(0)
    full_predictions['pose_enc_list'] = None
    
    depth_map_full = full_predictions["depth"]
    world_points_full = unproject_depth_map_to_point_map(depth_map_full, full_predictions["extrinsic"], full_predictions["intrinsic"])
    full_predictions["world_points_from_depth"] = world_points_full
    
    print(f"✓ Full Attention - Time: {full_time:.4f}s, Memory: {full_memory:.2f}GB")
    timing_info['full_time'] = full_time
    timing_info['full_memory'] = full_memory

    # ---- SPARSE ATTENTION ----
    sparse_predictions = None
    sparse_time = 0
    sparse_memory = 0
    
    if compare_sparse and MATCHING_GT is not None:
        print("\n[2/2] Running SPARSE ATTENTION inference...")
        print("-" * 80)
        
        model.enable_sparse_attention(matching_gt=MATCHING_GT)
        torch.cuda.empty_cache() if device == "cuda" else None
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            start_time = time.time()
            with torch.cuda.amp.autocast(dtype=dtype):
                sparse_predictions = model(images)
            sparse_time = time.time() - start_time
        
        if device == "cuda":
            sparse_memory = torch.cuda.max_memory_allocated() / 1e9
        
        # Post-process sparse attention results
        extrinsic_sparse, intrinsic_sparse = pose_encoding_to_extri_intri(sparse_predictions["pose_enc"], images.shape[-2:])
        sparse_predictions["extrinsic"] = extrinsic_sparse
        sparse_predictions["intrinsic"] = intrinsic_sparse
        
        for key in sparse_predictions.keys():
            if isinstance(sparse_predictions[key], torch.Tensor):
                sparse_predictions[key] = sparse_predictions[key].cpu().numpy().squeeze(0)
        sparse_predictions['pose_enc_list'] = None
        
        depth_map_sparse = sparse_predictions["depth"]
        world_points_sparse = unproject_depth_map_to_point_map(depth_map_sparse, sparse_predictions["extrinsic"], sparse_predictions["intrinsic"])
        sparse_predictions["world_points_from_depth"] = world_points_sparse
        
        print(f"✓ Sparse Attention - Time: {sparse_time:.4f}s, Memory: {sparse_memory:.2f}GB")
        timing_info['sparse_time'] = sparse_time
        timing_info['sparse_memory'] = sparse_memory
        
        # Compute speedup
        speedup = full_time / sparse_time if sparse_time > 0 else 0
        memory_reduction = full_memory / sparse_memory if sparse_memory > 0 else 0
        
        timing_info['speedup'] = speedup
        timing_info['memory_reduction'] = memory_reduction
        
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"  {'Attribute':<25} {'Full Attention':<20} {'Sparse Attention':<20}")
        print(f"  {'-'*65}")
        print(f"  {'Inference Time (s)':<25} {full_time:<20.4f} {sparse_time:<20.4f}")
        print(f"  {'Memory (GB)':<25} {full_memory:<20.2f} {sparse_memory:<20.2f}")
        print(f"  {'Speedup':<25} {'1.0x':<20} {f'{speedup:.2f}x':<20}")
        print(f"  {'Memory Reduction':<25} {'1.0x':<20} {f'{memory_reduction:.2f}x':<20}")
        
    torch.cuda.empty_cache()
    
    return full_predictions, sparse_predictions, timing_info


def compare_attention_methods(target_dir, model) -> str:
    """
    Generate a detailed comparison report between full and sparse attention.
    """
    if not os.path.isdir(target_dir):
        return "❌ No valid target directory found."
    
    try:
        full_preds, sparse_preds, timing = run_model_comparison(target_dir, model, compare_sparse=True)
        
        report = "## Attention Comparison Results\n\n"
        
        if timing:
            report += "### Performance Metrics\n\n"
            report += "| Metric | Full Attention | Sparse Attention |\n"
            report += "|--------|----------------|------------------|\n"
            report += f"| Inference Time (s) | {timing.get('full_time', 0):.4f} | {timing.get('sparse_time', 0):.4f} |\n"
            report += f"| Memory (GB) | {timing.get('full_memory', 0):.2f} | {timing.get('sparse_memory', 0):.2f} |\n"
            
            if 'speedup' in timing:
                report += f"| **Speedup** | - | **{timing['speedup']:.2f}x** |\n"
            if 'memory_reduction' in timing:
                report += f"| **Memory Reduction** | - | **{timing['memory_reduction']:.2f}x** |\n"
        
        report += "\n✅ Comparison complete! Both reconstructions are now available for visualization."
        return report
        
    except Exception as e:
        return f"❌ Error during comparison: {str(e)}"


# -------------------------------------------------------------------------
# 2) Handle uploaded images --> produce target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_images):
    """
    Create a new 'target_dir' + 'images' subfolder, and place user-uploaded
    images into it. Return (target_dir, image_paths).
    Note: Video support requires OpenGL libraries not available in headless containers.
    Please pre-extract video frames and upload as images instead.
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Create a unique folder name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    # Clean up if somehow that folder already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- Handle images ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # Sort final images for gallery
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"Files copied to {target_dir_images}; took {end_time - start_time:.3f} seconds")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) Update gallery on upload
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_images):
    """
    Whenever user uploads or changes files, immediately handle them
    and show in the gallery. Return (target_dir, image_paths).
    If nothing is uploaded, returns "None" and empty list.
    """
    if not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_images)
    return None, target_dir, image_paths, "Upload complete. Click 'Reconstruct' to begin 3D processing."


# -------------------------------------------------------------------------
# 4) Reconstruction: uses the target_dir plus any viz parameters
# -------------------------------------------------------------------------
def gradio_demo(
    target_dir,
    conf_thres=3.0,
    frame_filter="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    prediction_mode="Pointmap Regression",
):
    """
    Perform reconstruction using the already-created target_dir/images.
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory found. Please upload first.", None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # Prepare frame_filter dropdown
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["All"] + all_files

    print("Running run_model...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    # Save predictions
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # Handle None frame_filter
    if frame_filter is None:
        frame_filter = "All"

    # Build a GLB file name
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    # Convert predictions to GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds (including IO)")
    log_msg = f"Reconstruction Success ({len(all_files)} frames). Waiting for visualization."

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)


# -------------------------------------------------------------------------
# 5) Helper functions for UI resets + re-visualization
# -------------------------------------------------------------------------
def clear_fields():
    """
    Clears the 3D viewer, the stored target_dir, and empties the gallery.
    """
    return None


def update_log():
    """
    Display a quick log message while waiting.
    """
    return "Loading and Reconstructing..."


def update_visualization(
    target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode, is_example
):
    """
    Reload saved predictions from npz, create (or reuse) the GLB for new parameters,
    and return it for the 3D viewer. If is_example == "True", skip.
    """

    # If it's an example click, skip as requested
    if is_example == "True":
        return None, "No reconstruction available. Please click the Reconstruct button first."

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "No reconstruction available. Please click the Reconstruct button first."

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"No reconstruction available at {predictions_path}. Please run 'Reconstruct' first."

    key_list = [
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "images",
        "extrinsic",
        "intrinsic",
        "world_points_from_depth",
    ]

    loaded = np.load(predictions_path)
    predictions = {key: np.array(loaded[key]) for key in key_list}

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "Updating Visualization"


# -------------------------------------------------------------------------
# Pre-extracted scene discovery and loading
# -------------------------------------------------------------------------
SCENE_DATA_BASE = "/scratch/cvlab/home/pilligua/mVGGT/notebooks"

def discover_scenes():
    """Find all matching_data* directories (recursively) with images.npy and matching_gt.npy."""
    scenes = {}
    base = Path(SCENE_DATA_BASE)
    if not base.exists():
        print(f"      ⚠ Scene data base not found: {SCENE_DATA_BASE}")
        return scenes
    for d in sorted(base.rglob("matching_data*")):
        if not d.is_dir():
            continue
        if (d / "images.npy").exists() and (d / "matching_gt.npy").exists():
            name = d.name
            if name.startswith("matching_data_"):
                name = name[len("matching_data_") :]
            if name == "":
                name = "original"
            scenes[name] = str(d)
    return scenes

AVAILABLE_SCENES = discover_scenes()
print(f"      ✓ Found {len(AVAILABLE_SCENES)} pre-extracted scenes: {list(AVAILABLE_SCENES.keys())}")


def load_scene_from_npy(scene_name):
    """
    Load a pre-extracted scene's images from .npy, save them as PNGs
    in a new target_dir/images folder, and return (target_dir, image_paths, status).
    """
    if scene_name is None or scene_name not in AVAILABLE_SCENES:
        return None, None, "Please select a scene."

    scene_dir = AVAILABLE_SCENES[scene_name]
    images_npy = np.load(os.path.join(scene_dir, "images.npy"))  # [1, V, 3, H, W]
    images = images_npy[0]  # [V, 3, H, W]

    # Create target dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")
    os.makedirs(target_dir_images, exist_ok=True)

    image_paths = []
    for i in range(images.shape[0]):
        img = images[i].transpose(1, 2, 0)  # [H, W, 3]
        img = (img * 255).clip(0, 255).astype(np.uint8)
        from PIL import Image as PILImage
        path = os.path.join(target_dir_images, f"view_{i:02d}.png")
        PILImage.fromarray(img).save(path)
        image_paths.append(path)

    print(f"Loaded scene '{scene_name}': {len(image_paths)} views from {scene_dir}")
    return target_dir, image_paths, f"Loaded scene **{scene_name}** ({len(image_paths)} views). Click 'Reconstruct' to begin."


# Note: Video examples removed - provide pre-extracted image frames instead


# -------------------------------------------------------------------------
# 6) Build Gradio UI
# -------------------------------------------------------------------------
print("[4/4] Building Gradio web interface...")
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

custom_css = """
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }
    
    .example-log * {
        font-style: italic;
        font-size: 16px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }
    
    #my_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
    }

    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }
    """

with gr.Blocks() as demo:
    # Instead of gr.State, we use a hidden Textbox:
    is_example = gr.Textbox(label="is_example", visible=False, value="None")
    num_images = gr.Textbox(label="num_images", visible=False, value="None")

    gr.HTML(
        """
    <h1>🏛️ VGGT: Visual Geometry Grounded Transformer</h1>
    <p>
    <a href="https://github.com/facebookresearch/vggt">🐙 GitHub Repository</a> |
    <a href="#">Project Page</a>
    </p>

    <div style="font-size: 16px; line-height: 1.5;">
    <p>Upload a set of images to create a 3D reconstruction of a scene or object. VGGT takes these images and generates a 3D point cloud, along with estimated camera poses.</p>

    <h3>Getting Started:</h3>
    <ol>
        <li><strong>Upload Your Images:</strong> Use the "Upload Images" button on the left to provide your input. For best results, provide multiple views of the same scene.</li>
        <li><strong>Preview:</strong> Your uploaded images will appear in the gallery on the left.</li>
        <li><strong>Reconstruct:</strong> Click the "Reconstruct" button to start the 3D reconstruction process.</li>
        <li><strong>Visualize:</strong> The 3D reconstruction will appear in the viewer on the right. You can rotate, pan, and zoom to explore the model, and download the GLB file. Note the visualization of 3D points may be slow for a large number of input images.</li>
        <li>
        <strong>Adjust Visualization (Optional):</strong>
        After reconstruction, you can fine-tune the visualization using the options below
        <details style="display:inline;">
            <summary style="display:inline;">(<strong>click to expand</strong>):</summary>
            <ul>
            <li><em>Confidence Threshold:</em> Adjust the filtering of points based on confidence.</li>
            <li><em>Show Points from Frame:</em> Select specific frames to display in the point cloud.</li>
            <li><em>Show Camera:</em> Toggle the display of estimated camera positions.</li>
            <li><em>Filter Sky / Filter Black Background:</em> Remove sky or black-background points.</li>
            <li><em>Select a Prediction Mode:</em> Choose between "Depthmap and Camera Branch" or "Pointmap Branch."</li>
            </ul>
        </details>
        </li>
    </ol>
    <p><strong style="color: #0ea5e9;">Please note:</strong> <span style="color: #0ea5e9; font-weight: bold;">VGGT typically reconstructs a scene in less than 1 second. However, visualizing 3D points may take tens of seconds due to third-party rendering, which are independent of VGGT's processing time. </span></p>
    </div>
    """
    )

    target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Load a Pre-extracted Scene")
            scene_choices = list(AVAILABLE_SCENES.keys()) if AVAILABLE_SCENES else []
            scene_selector = gr.Dropdown(
                choices=scene_choices,
                value=scene_choices[0] if scene_choices else None,
                label="Select Scene",
                info=f"{len(scene_choices)} scenes available (ScanNet indoor + VKITTI driving)",
                interactive=True,
            )
            load_scene_btn = gr.Button("Load Scene", variant="secondary")

            gr.Markdown("---\n*Or upload your own images:*")
            input_images = gr.File(file_count="multiple", label="Upload Images", interactive=True)

            image_gallery = gr.Gallery(
                label="Preview",
                columns=4,
                height="300px",
                object_fit="contain",
            )

        with gr.Column(scale=4):
            with gr.Column():
                gr.Markdown("**3D Reconstruction (Point Cloud and Camera Poses)**")
                log_output = gr.Markdown(
                    "Please upload a video or images, then click Reconstruct.", elem_classes=["custom-log"]
                )
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

            with gr.Row():
                submit_btn = gr.Button("Reconstruct", scale=1, variant="primary")
                clear_btn = gr.ClearButton(
                    [input_images, reconstruction_output, log_output, target_dir_output, image_gallery],
                    scale=1,
                )

            with gr.Row():
                prediction_mode = gr.Radio(
                    ["Depthmap and Camera Branch", "Pointmap Branch"],
                    label="Select a Prediction Mode",
                    value="Depthmap and Camera Branch",
                    scale=1,
                    elem_id="my_radio",
                )

            with gr.Row():
                conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
                frame_filter = gr.Dropdown(choices=["All"], value="All", label="Show Points from Frame")
                with gr.Column():
                    show_cam = gr.Checkbox(label="Show Camera", value=True)
                    mask_sky = gr.Checkbox(label="Filter Sky", value=False)
                    mask_black_bg = gr.Checkbox(label="Filter Black Background", value=False)
                    mask_white_bg = gr.Checkbox(label="Filter White Background", value=False)

    # ---------------------- Sparse Attention Comparison Section ----------------------
    gr.Markdown(
        """
        ---
        ## 🚀 Sparse vs Full Attention Comparison
        Compare 3D reconstructions using full multi-view attention vs optimized sparse attention.
        This demonstrates the computational efficiency gains from our sparse attention mechanism.
        """
    )
    
    if MATCHING_GT is not None:
        with gr.Row():
            with gr.Column(scale=2):
                comparison_log = gr.Markdown(
                    "📝 Click 'Compare Attention Methods' to run both inference modes and compare performance.",
                    elem_classes=["example-log"]
                )
            
            with gr.Column(scale=1):
                compare_btn = gr.Button("Compare Attention Methods", variant="secondary", scale=1)
        
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("**Full Attention (Baseline)**")
                full_attn_output = gr.Model3D(height=450, zoom_speed=0.5, pan_speed=0.5)
            
            with gr.Column(scale=2):
                gr.Markdown("**Sparse Attention (Optimized)**")
                sparse_attn_output = gr.Model3D(height=450, zoom_speed=0.5, pan_speed=0.5)
        
        with gr.Row():
            comparison_report = gr.Markdown(
                "Comparison details will appear here after running the comparison.",
                label="Comparison Report"
            )
    else:
        gr.Markdown(
            """
            ⚠️ **Sparse Attention Comparison Unavailable**
            
            The sparse attention comparison feature requires matching ground truth data.
            To enable this feature, ensure `notebooks/matching_data/matching_gt.npy` is available.
            
            See `/mnt/cvlab/scratch/cvlab/home/pilligua/claude.md` for more details about the sparse attention implementation.
            """
        )

    # ---------------------- Examples section ----------------------
    examples = [
        ["22", None, 20.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        ["30", None, 35.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        ["1", None, 15.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        ["1", None, 20.0, False, False, True, True, "Depthmap and Camera Branch", "True"],
        ["8", None, 5.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        ["25", None, 50.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
        ["20", None, 45.0, False, False, True, False, "Depthmap and Camera Branch", "True"],
    ]

    def example_pipeline(
        num_images_str,
        input_images,
        conf_thres,
        mask_black_bg,
        mask_white_bg,
        show_cam,
        mask_sky,
        prediction_mode,
        is_example_str,
    ):
        """
        1) Copy example images to new target_dir
        2) Reconstruct
        3) Return model3D + logs + new_dir + updated dropdown + gallery
        We do NOT return is_example. It's just an input.
        """
        target_dir, image_paths = handle_uploads(input_images)
        # Always use "All" for frame_filter in examples
        frame_filter = "All"
        glbfile, log_msg, dropdown = gradio_demo(
            target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode
        )
        return glbfile, log_msg, target_dir, dropdown, image_paths

    gr.Markdown("Click any row to load an example.", elem_classes=["example-log"])

    gr.Examples(
        examples=examples,
        inputs=[
            num_images,
            input_images,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        outputs=[reconstruction_output, log_output, target_dir_output, frame_filter, image_gallery],
        fn=example_pipeline,
        cache_examples=False,
        examples_per_page=50,
    )

    # -------------------------------------------------------------------------
    # "Reconstruct" button logic:
    #  - Clear fields
    #  - Update log
    #  - gradio_demo(...) with the existing target_dir
    #  - Then set is_example = "False"
    # -------------------------------------------------------------------------
    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo,
        inputs=[
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        outputs=[reconstruction_output, log_output, frame_filter],
    ).then(
        fn=lambda: "False", inputs=[], outputs=[is_example]  # set is_example to "False"
    )

    # -------------------------------------------------------------------------
    # Real-time Visualization Updates
    # -------------------------------------------------------------------------
    conf_thres.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    frame_filter.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    mask_black_bg.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    mask_white_bg.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    show_cam.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    mask_sky.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    prediction_mode.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )

    # -------------------------------------------------------------------------
    # Sparse Attention Comparison Button Logic
    # -------------------------------------------------------------------------
    if MATCHING_GT is not None:
        def run_comparison_and_generate_models(target_dir):
            """
            Run full vs sparse attention comparison and generate GLB files for both.
            """
            if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
                return None, None, "❌ No valid target directory. Please reconstruct first."
            
            try:
                print("\n" + "="*80)
                print("GENERATING FULL VS SPARSE ATTENTION COMPARISON")
                print("="*80)
                
                # Run comparison
                full_preds, sparse_preds, timing = run_model_comparison(target_dir, model, compare_sparse=True)
                
                # Generate GLB file for full attention
                print("\n[1/2] Generating GLB for full attention...")
                glbfile_full = os.path.join(target_dir, "glbscene_full_attention.glb")
                glbscene_full = predictions_to_glb(
                    full_preds,
                    conf_thres=50.0,
                    filter_by_frames="All",
                    mask_black_bg=False,
                    mask_white_bg=False,
                    show_cam=True,
                    mask_sky=False,
                    target_dir=target_dir,
                    prediction_mode="Depthmap and Camera Branch",
                )
                glbscene_full.export(file_obj=glbfile_full)
                print(f"✓ Full attention GLB saved: {glbfile_full}")
                
                # Generate GLB file for sparse attention
                glbfile_sparse = None
                if sparse_preds is not None:
                    print("\n[2/2] Generating GLB for sparse attention...")
                    glbfile_sparse = os.path.join(target_dir, "glbscene_sparse_attention.glb")
                    glbscene_sparse = predictions_to_glb(
                        sparse_preds,
                        conf_thres=50.0,
                        filter_by_frames="All",
                        mask_black_bg=False,
                        mask_white_bg=False,
                        show_cam=True,
                        mask_sky=False,
                        target_dir=target_dir,
                        prediction_mode="Depthmap and Camera Branch",
                    )
                    glbscene_sparse.export(file_obj=glbfile_sparse)
                    print(f"✓ Sparse attention GLB saved: {glbfile_sparse}")
                
                # Generate report
                report = "## ⚡ Sparse vs Full Attention Comparison\n\n"
                
                if timing:
                    report += "### Performance Metrics\n\n"
                    report += "| Metric | Full Attention | Sparse Attention |\n"
                    report += "|--------|----------------|------------------|\n"
                    report += f"| Inference Time (s) | {timing.get('full_time', 0):.4f} | {timing.get('sparse_time', 0):.4f} |\n"
                    report += f"| Memory (GB) | {timing.get('full_memory', 0):.2f} | {timing.get('sparse_memory', 0):.2f} |\n"
                    
                    if 'speedup' in timing:
                        report += f"| **Speedup** | - | **{timing['speedup']:.2f}x** |\n"
                    if 'memory_reduction' in timing:
                        report += f"| **Memory Reduction** | - | **{timing['memory_reduction']:.2f}x** |\n"
                
                report += "\n✅ Both reconstructions generated successfully!\n"
                report += "Compare the 3D models on the left (Full Attention) and right (Sparse Attention)."
                
                gc.collect()
                torch.cuda.empty_cache()
                
                return glbfile_full, glbfile_sparse, report
                
            except Exception as e:
                import traceback
                error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
                return None, None, error_msg
        
        # Only add the button callback if MATCHING_GT is available
        try:
            compare_btn.click(
                fn=run_comparison_and_generate_models,
                inputs=[target_dir_output],
                outputs=[full_attn_output, sparse_attn_output, comparison_report],
            )
        except NameError:
            print("⚠ Sparse attention comparison UI not available")

    # -------------------------------------------------------------------------
    # Load pre-extracted scene from dropdown
    # -------------------------------------------------------------------------
    def on_load_scene(scene_name):
        """Load a pre-extracted scene and populate gallery + target_dir."""
        if not scene_name:
            return None, None, None, "Please select a scene from the dropdown."
        target_dir, image_paths, status = load_scene_from_npy(scene_name)
        if target_dir is None:
            return None, None, None, status
        return None, target_dir, image_paths, status

    load_scene_btn.click(
        fn=on_load_scene,
        inputs=[scene_selector],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )

    # -------------------------------------------------------------------------
    # Auto-update gallery whenever user uploads or changes their files
    # -------------------------------------------------------------------------
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )

    # Launch with share=True to get a public URL
    print("      ✓ UI built successfully")
    print("\n" + "="*60)
    print("Starting Gradio server...")
    print("="*60)
    demo.queue(max_size=20).launch(
        theme=theme,
        css=custom_css,
        show_error=True, 
        share=True
    )
