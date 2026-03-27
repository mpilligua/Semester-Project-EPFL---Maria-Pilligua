# Token Correspondence Visualization for Sparse Multi-View Attention
#
# Visualize which tokens correspond across different views using matching_gt data.
# Select a scene, click on any token in a reference view, and see the corresponding
# tokens highlighted in all other views.
#
# Usage:
#   runai exec pilligua-inter-sparse-attention -- python3 \
#     /scratch/cvlab/home/pilligua/Semester-Project-EPFL---Maria-Pilligua/vggt/demo_gradio_only_comparison.py

import os
import sys

os.environ['DISPLAY'] = ''
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

import torch
import numpy as np
import gradio as gr
import shutil
import glob
import gc
import time
from datetime import datetime
from pathlib import Path
from PIL import Image as PILImage, ImageDraw, ImageFont
import matplotlib
import matplotlib.cm as cm
import io
import base64
import json
import torch.nn.functional as F

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

# ─── Constants ────────────────────────────────────────────────────────────────
SCENE_DATA_BASE = "/scratch/cvlab/home/pilligua/mVGGT/notebooks"
TOKEN_PATCH = 14
NUM_VIEWS = 8
DEMO_CACHE_DIR = "/scratch/cvlab/home/pilligua/vggt_demo_cache"

# Colors (RGBA for overlays)
COLOR_SOURCE = (255, 50, 50, 180)        # Red — the clicked token
COLOR_CORRESPOND = (50, 220, 50, 150)    # Green — corresponding tokens
COLOR_GRID = (200, 200, 200, 40)         # Light gray grid lines


# ─── Scene Discovery ─────────────────────────────────────────────────────────

def _is_scene_dir(d: Path) -> bool:
    return d.is_dir() and (d / "images.npy").exists() and (d / "matching_gt.npy").exists()


def discover_scenes():
    """
    Find all extracted scenes.

    Searches in two places under SCENE_DATA_BASE:
      1. dataset_examples/<DatasetName>/<SceneName>/   (new organised layout)
         → displayed as "DatasetName/SceneName"
      2. matching_data_*/ at any depth                 (legacy flat layout)
         → displayed as "legacy/<stripped-name>"
    """
    scenes = {}
    base = Path(SCENE_DATA_BASE)
    if not base.exists():
        print(f"  Warning: Scene data base not found: {SCENE_DATA_BASE}")
        return scenes

    # 1. New structured layout: dataset_examples/DatasetName/SceneName/
    examples_base = base / "dataset_examples"
    if examples_base.exists():
        for dataset_dir in sorted(examples_base.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for scene_dir in sorted(dataset_dir.iterdir()):
                if _is_scene_dir(scene_dir):
                    name = f"{dataset_dir.name}/{scene_dir.name}"
                    scenes[name] = str(scene_dir)

    # 2. Legacy matching_data_* directories anywhere under base
    for d in sorted(base.rglob("matching_data*")):
        if not _is_scene_dir(d):
            continue
        name = d.name
        if name.startswith("matching_data_"):
            name = name[len("matching_data_"):]
        if name == "":
            name = "original"
        # prefix with "legacy/" to distinguish from organised layout
        legacy_key = f"legacy/{name}"
        if legacy_key not in scenes:
            scenes[legacy_key] = str(d)

    return scenes


AVAILABLE_SCENES = discover_scenes()
print(f"Found {len(AVAILABLE_SCENES)} scenes: {list(AVAILABLE_SCENES.keys())}")


# ─── Model Loading ────────────────────────────────────────────────────────────

print("Loading VGGT model...")
model = VGGT()
cached_model_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/model.pt")
if os.path.exists(cached_model_path):
    print(f"  Loading cached model from {cached_model_path}")
    model_state = torch.load(cached_model_path, map_location="cpu")
else:
    print("  Downloading model from HuggingFace (first time)...")
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model_state = torch.hub.load_state_dict_from_url(_URL)
model.load_state_dict(model_state)
model.eval()
model = model.to(device)
print(f"  Model on device: {device}")

# Load default matching GT for sparse attention
MATCHING_GT = None
try:
    for p in [
        "notebooks/matching_data/matching_gt.npy",
        "/scratch/cvlab/home/pilligua/mVGGT/notebooks/matching_data/matching_gt.npy",
    ]:
        if os.path.exists(p):
            MATCHING_GT = torch.from_numpy(np.load(p)).bool().to(device)
            print(f"  Default matching GT loaded from {p}: {MATCHING_GT.shape}")
            break
    if MATCHING_GT is None:
        print("  Warning: No default matching GT found")
except Exception as e:
    print(f"  Warning: Could not load matching GT: {e}")


# ─── Rendering Helpers ────────────────────────────────────────────────────────

def np_to_pil(img_chw):
    """Convert [3, H, W] float32 (0-1) numpy array to PIL Image."""
    img = (img_chw.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    return PILImage.fromarray(img)


def draw_token_grid(pil_img, token_h, token_w):
    """Draw a light token grid overlay on a PIL image (in-place via RGBA composite)."""
    overlay = PILImage.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = pil_img.size

    for r in range(1, token_h):
        y = r * TOKEN_PATCH
        draw.line([(0, y), (w, y)], fill=COLOR_GRID, width=1)
    for c in range(1, token_w):
        x = c * TOKEN_PATCH
        draw.line([(x, 0), (x, h)], fill=COLOR_GRID, width=1)

    base = pil_img.convert("RGBA")
    return PILImage.alpha_composite(base, overlay).convert("RGB")


def highlight_token(pil_img, th, tw, color, border_width=2):
    """Highlight a single token cell on a PIL image."""
    overlay = PILImage.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    x0 = tw * TOKEN_PATCH
    y0 = th * TOKEN_PATCH
    x1 = x0 + TOKEN_PATCH - 1
    y1 = y0 + TOKEN_PATCH - 1

    # Fill
    draw.rectangle([x0, y0, x1, y1], fill=color)
    # Border (fully opaque version of the color)
    border_color = (color[0], color[1], color[2], 255)
    for i in range(border_width):
        draw.rectangle([x0 + i, y0 + i, x1 - i, y1 - i], outline=border_color)

    base = pil_img.convert("RGBA")
    return PILImage.alpha_composite(base, overlay).convert("RGB")


def add_view_label(pil_img, label, is_source=False):
    """Add a small label in the top-left corner."""
    draw = ImageDraw.Draw(pil_img)
    bg_color = (200, 50, 50) if is_source else (50, 50, 50)
    text_color = (255, 255, 255)

    # Draw background rectangle
    bbox = draw.textbbox((0, 0), label)
    tw_text = bbox[2] - bbox[0] + 8
    th_text = bbox[3] - bbox[1] + 6
    draw.rectangle([0, 0, tw_text, th_text], fill=bg_color)
    draw.text((4, 2), label, fill=text_color)
    return pil_img


def render_view_plain(img_chw, view_idx):
    """Render a view with token grid and label, no highlights."""
    pil = np_to_pil(img_chw)
    h, w = img_chw.shape[1], img_chw.shape[2]
    token_h = h // TOKEN_PATCH
    token_w = w // TOKEN_PATCH
    pil = draw_token_grid(pil, token_h, token_w)
    pil = add_view_label(pil, f"View {view_idx}")
    return pil


def render_correspondence_view(img_chw, view_idx, correspond_tokens, is_source=False, src_th=None, src_tw=None):
    """
    Render a view with correspondences highlighted.

    Args:
        img_chw: [3, H, W] float32
        view_idx: int
        correspond_tokens: list of (th, tw) tuples to highlight in green
        is_source: if True, this is the source view
        src_th, src_tw: source token position (highlighted in red if is_source)
    """
    pil = np_to_pil(img_chw)
    h, w = img_chw.shape[1], img_chw.shape[2]
    token_h = h // TOKEN_PATCH
    token_w = w // TOKEN_PATCH
    pil = draw_token_grid(pil, token_h, token_w)

    # Highlight corresponding tokens in green
    for (th, tw) in correspond_tokens:
        pil = highlight_token(pil, th, tw, COLOR_CORRESPOND)

    # Highlight source token in red
    if is_source and src_th is not None and src_tw is not None:
        pil = highlight_token(pil, src_th, src_tw, COLOR_SOURCE, border_width=3)

    n_matches = len(correspond_tokens)
    label = f"View {view_idx} (SOURCE)" if is_source else f"View {view_idx} ({n_matches} matches)"
    pil = add_view_label(pil, label, is_source=is_source)
    return pil


# ─── Core Logic ───────────────────────────────────────────────────────────────

def load_scene(scene_name):
    """Load scene data: images [V,3,H,W] and matching_gt [V,Th,Tw,V,Th,Tw]."""
    if scene_name is None or scene_name not in AVAILABLE_SCENES:
        return None, None, "Please select a scene."

    scene_dir = AVAILABLE_SCENES[scene_name]
    images = np.load(os.path.join(scene_dir, "images.npy"))[0]       # [V, 3, H, W]
    matching_gt = np.load(os.path.join(scene_dir, "matching_gt.npy"))[0]  # [V, Th, Tw, V, Th, Tw]

    print(f"Loaded scene '{scene_name}': images {images.shape}, matching_gt {matching_gt.shape}")
    # --- FIX: Resize images to model input resolution (518x518) ---
    # VGGT patch grid assumes a 14px patch size, so 518/14 = 37.
    img_tensor = torch.from_numpy(images).float()
    img_resized = F.interpolate(
        img_tensor, size=(518, 518), mode="bilinear", align_corners=False
    )
    images = img_resized.cpu().numpy()
    # ---------------------------------------------------------------

    return images, matching_gt, (
        f"Loaded **{scene_name}** — Resized to 518x518; "
        f"token grid {matching_gt.shape[1]}x{matching_gt.shape[2]}"
    )


def get_plain_views(images):
    """Render all views with token grid, no highlights."""
    if images is None:
        return [None] * NUM_VIEWS
    return [render_view_plain(images[i], i) for i in range(min(NUM_VIEWS, images.shape[0]))]


def compute_correspondences(scene_data, src_view, pixel_x, pixel_y):
    """
    Given a click position, compute and render all views with correspondences.

    Args:
        scene_data: (images, matching_gt) tuple from gr.State
        src_view: int, source view index
        pixel_x, pixel_y: click coordinates in pixels

    Returns:
        list of 8 PIL images + status string
    """
    if scene_data is None:
        return [None] * NUM_VIEWS, "Load a scene first."

    images, matching_gt = scene_data
    V = images.shape[0]
    H, W = images.shape[2], images.shape[3]
    token_h = H // TOKEN_PATCH
    token_w = W // TOKEN_PATCH

    # Pixel → token
    src_th = min(int(pixel_y) // TOKEN_PATCH, token_h - 1)
    src_tw = min(int(pixel_x) // TOKEN_PATCH, token_w - 1)
    # After resizing to 518x518, token indices should be in [0, 36].
    src_th = max(0, min(int(src_th), 36))
    src_tw = max(0, min(int(src_tw), 36))
    src_th = max(0, src_th)
    src_tw = max(0, src_tw)

    result_images = []
    total_matches = 0

    for tgt_v in range(V):
        # Get correspondence mask for this target view
        correspond_mask = matching_gt[src_view, src_th, src_tw, tgt_v]  # [Th, Tw]
        correspond_tokens = list(zip(*np.where(correspond_mask)))  # list of (th, tw)

        is_source = (tgt_v == src_view)
        pil = render_correspondence_view(
            images[tgt_v], tgt_v,
            correspond_tokens,
            is_source=is_source,
            src_th=src_th if is_source else None,
            src_tw=src_tw if is_source else None,
        )
        result_images.append(pil)

        if not is_source:
            total_matches += len(correspond_tokens)

    status = (
        f"Source: View {src_view}, Token ({src_th}, {src_tw}) — "
        f"pixel ({int(pixel_x)}, {int(pixel_y)}) — "
        f"**{total_matches} correspondences** across {V - 1} other views"
    )
    return result_images, status


# ─── Model Comparison ─────────────────────────────────────────────────────────

def save_scene_images_to_dir(images_np):
    """
    Save scene images (from .npy) as PNGs in a target_dir/images folder
    so the model can load them via load_and_preprocess_images.
    Returns target_dir path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")
    os.makedirs(target_dir_images, exist_ok=True)

    for i in range(images_np.shape[0]):
        img = images_np[i].transpose(1, 2, 0)  # [H, W, 3]
        img = (img * 255).clip(0, 255).astype(np.uint8)
        PILImage.fromarray(img).save(os.path.join(target_dir_images, f"view_{i:02d}.png"))

    return target_dir


def run_model_comparison(target_dir, matching_gt_tensor) -> tuple:
    """
    Run VGGT with both full and sparse attention, capturing Q/K from all global blocks.
    Returns (full_preds, sparse_preds, timing, full_qk, sparse_qk, sparse_mask, model_info).
    model_info: dict with frames_uint8, grid_h, grid_w, patch_start_idx, P, S from preprocessed images.
    """
    image_names = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
    if len(image_names) == 0:
        raise ValueError("No images found in target_dir.")

    images = load_and_preprocess_images(image_names).to(device)
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    timing = {}

    def postprocess(preds):
        ext, intr = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
        preds["extrinsic"] = ext
        preds["intrinsic"] = intr
        for k in preds.keys():
            if isinstance(preds[k], torch.Tensor):
                preds[k] = preds[k].cpu().numpy().squeeze(0)
        preds['pose_enc_list'] = None
        preds["world_points_from_depth"] = unproject_depth_map_to_point_map(
            preds["depth"], preds["extrinsic"], preds["intrinsic"]
        )
        return preds

    # ── Full Attention (with QK capture) ──
    print("[1/2] Running FULL ATTENTION (+ QK capture)...")
    model.disable_sparse_attention()
    model.aggregator.enable_global_qk_capture()
    torch.cuda.empty_cache()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        t0 = time.time()
        with torch.cuda.amp.autocast(dtype=dtype):
            full_preds = model(images)
        timing['full_time'] = time.time() - t0
    timing['full_memory'] = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
    full_qk = model.aggregator.get_captured_qk_all()
    # Move captured Q/K to CPU to avoid CUDA OOM during later attention rendering.
    full_qk = [
        (q.detach().cpu().half() if q is not None else None,
         k.detach().cpu().half() if k is not None else None)
        for (q, k) in full_qk
    ]
    model.aggregator.disable_global_qk_capture()
    full_preds = postprocess(full_preds)
    qk_mem = sum((q.nelement() * 2 + k.nelement() * 2) for q, k in full_qk if q is not None) / 1e6
    print(f"  Full: {timing['full_time']:.3f}s, {timing['full_memory']:.2f}GB, QK stored: {qk_mem:.0f}MB")

    # ── Sparse Attention (with QK capture) ──
    sparse_preds = None
    sparse_qk = None
    sparse_mask = None
    if matching_gt_tensor is not None:
        print("[2/2] Running SPARSE ATTENTION (+ QK capture)...")
        model.enable_sparse_attention(matching_gt=matching_gt_tensor)
        model.aggregator.enable_global_qk_capture()
        torch.cuda.empty_cache()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            t0 = time.time()
            with torch.cuda.amp.autocast(dtype=dtype):
                sparse_preds = model(images)
            timing['sparse_time'] = time.time() - t0
        timing['sparse_memory'] = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
        sparse_qk = model.aggregator.get_captured_qk_all()
        # Move captured Q/K to CPU to avoid CUDA OOM during attention rendering.
        sparse_qk = [
            (q.detach().cpu().half() if q is not None else None,
             k.detach().cpu().half() if k is not None else None)
            for (q, k) in sparse_qk
        ]
        model.aggregator.disable_global_qk_capture()

        # Capture the sparse attention mask that was used (reconstruct it)
        # The mask is stored during forward; we rebuild it here for visualization
        B_img = images.shape[0] if images.dim() == 5 else 1
        S_img = images.shape[1] if images.dim() == 5 else images.shape[0]
        H_img, W_img = images.shape[-2], images.shape[-1]
        patch_size = model.aggregator.patch_size
        patch_h, patch_w = H_img // patch_size, W_img // patch_size
        P_patches = patch_h * patch_w
        psi = model.aggregator.patch_start_idx
        P_total = psi + P_patches

        # Rebuild sparse mask from matching_gt
        from vggt.utils.multi_view_matcher import MultiViewMatcher
        matcher = model.aggregator.multi_view_matcher
        matcher.num_patches_h = patch_h
        matcher.num_patches_w = patch_w
        matcher.num_tokens_per_view = P_patches
        gt_subset = matching_gt_tensor[:, :S_img, :, :, :S_img, :, :]
        raw_mask, _ = matcher(matching_gt=gt_subset)
        raw_mask = raw_mask.to(dtype=torch.float32)

        # Pad to include special tokens
        padded = torch.zeros(B_img, S_img * P_total, S_img * P_total)
        for s in range(S_img):
            s0 = s * P_total
            se = s0 + psi
            sp = se
            send = s0 + P_total
            for s2 in range(S_img):
                s20 = s2 * P_total
                s2e = s20 + psi
                s2p = s2e
                s2end = s20 + P_total
                if s == s2:
                    padded[:, s0:send, s20:s2end] = 1.0
                else:
                    padded[:, s0:se, s20:s2e] = 1.0
                    padded[:, sp:send, s2p:s2end] = raw_mask[:,
                        s * P_patches:(s+1) * P_patches,
                        s2 * P_patches:(s2+1) * P_patches]
        sparse_mask = padded.cpu().half()

        model.disable_sparse_attention()
        sparse_preds = postprocess(sparse_preds)

        timing['speedup'] = timing['full_time'] / timing['sparse_time'] if timing['sparse_time'] > 0 else 0
        timing['memory_reduction'] = timing['full_memory'] / timing['sparse_memory'] if timing['sparse_memory'] > 0 else 0
        print(f"  Sparse: {timing['sparse_time']:.3f}s, {timing['sparse_memory']:.2f}GB")
        print(f"  Speedup: {timing['speedup']:.2f}x, Memory reduction: {timing['memory_reduction']:.2f}x")
    else:
        print("  Skipping sparse attention (no matching GT for this scene)")

    # Compute model_info from the actual preprocessed image dimensions
    if images.dim() == 4:
        images_5d = images.unsqueeze(0)
    else:
        images_5d = images
    S_actual = images_5d.shape[1]
    H_actual, W_actual = images_5d.shape[-2], images_5d.shape[-1]
    patch_size = model.aggregator.patch_size
    grid_h_actual = H_actual // patch_size
    grid_w_actual = W_actual // patch_size
    P_actual = grid_h_actual * grid_w_actual
    psi_actual = model.aggregator.patch_start_idx
    frames_uint8 = (images_5d[0].detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    model_info = {
        "frames_uint8": frames_uint8,
        "grid_h": grid_h_actual, "grid_w": grid_w_actual,
        "patch_start_idx": psi_actual, "P": P_actual, "S": S_actual,
    }
    print(f"  Model processed images: {H_actual}x{W_actual}, grid: {grid_h_actual}x{grid_w_actual}, P={P_actual}")

    torch.cuda.empty_cache()
    return full_preds, sparse_preds, timing, full_qk, sparse_qk, sparse_mask, model_info


# ─── Disk Cache Helpers ────────────────────────────────────────────────────────

def _cache_scene_dir(scene_name: str) -> str:
    safe = scene_name.replace("/", "__").replace(" ", "_")
    return os.path.join(DEMO_CACHE_DIR, safe)


def _save_reconstruction_cache(scene_name, timing, full_qk, sparse_qk, sparse_mask,
                                model_info, glb_full_path, glb_sparse_path):
    """Persist all comparison outputs to DEMO_CACHE_DIR/<scene>/ ."""
    cache_dir = _cache_scene_dir(scene_name)
    os.makedirs(cache_dir, exist_ok=True)

    # GLB files — copy into the cache so they survive temp-dir cleanup
    shutil.copy2(glb_full_path, os.path.join(cache_dir, "full_attention.glb"))
    if glb_sparse_path and os.path.exists(glb_sparse_path):
        shutil.copy2(glb_sparse_path, os.path.join(cache_dir, "sparse_attention.glb"))

    # QK tensors (fp16, CPU) — can be large but are needed for attention viz
    torch.save(full_qk, os.path.join(cache_dir, "full_qk.pt"))
    if sparse_qk is not None:
        torch.save(sparse_qk, os.path.join(cache_dir, "sparse_qk.pt"))
    if sparse_mask is not None:
        torch.save(sparse_mask, os.path.join(cache_dir, "sparse_mask.pt"))

    # model_info: scalar fields go to JSON, frames_uint8 goes to .npy
    np.save(os.path.join(cache_dir, "frames_uint8.npy"), model_info["frames_uint8"])
    meta = {k: int(v) for k, v in model_info.items() if k != "frames_uint8"}
    with open(os.path.join(cache_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f)

    # Timing
    with open(os.path.join(cache_dir, "timing.json"), "w") as f:
        json.dump(timing, f)

    print(f"[Cache] Saved reconstruction for '{scene_name}' → {cache_dir}")


def _load_reconstruction_cache(scene_name):
    """
    Load cached comparison outputs.
    Returns (glb_full, glb_sparse, timing, full_qk, sparse_qk, sparse_mask, model_info)
    or None if no valid cache exists.
    """
    cache_dir = _cache_scene_dir(scene_name)
    glb_full = os.path.join(cache_dir, "full_attention.glb")

    if not os.path.exists(glb_full):
        return None

    try:
        glb_sparse = os.path.join(cache_dir, "sparse_attention.glb")
        if not os.path.exists(glb_sparse):
            glb_sparse = None

        full_qk = torch.load(os.path.join(cache_dir, "full_qk.pt"), map_location="cpu")

        sparse_qk_path = os.path.join(cache_dir, "sparse_qk.pt")
        sparse_qk = torch.load(sparse_qk_path, map_location="cpu") if os.path.exists(sparse_qk_path) else None

        sparse_mask_path = os.path.join(cache_dir, "sparse_mask.pt")
        sparse_mask = torch.load(sparse_mask_path, map_location="cpu") if os.path.exists(sparse_mask_path) else None

        frames_uint8 = np.load(os.path.join(cache_dir, "frames_uint8.npy"))
        with open(os.path.join(cache_dir, "model_meta.json")) as f:
            meta = json.load(f)
        model_info = {
            "frames_uint8": frames_uint8,
            "grid_h": int(meta["grid_h"]),
            "grid_w": int(meta["grid_w"]),
            "patch_start_idx": int(meta["patch_start_idx"]),
            "P": int(meta["P"]),
            "S": int(meta["S"]),
        }

        with open(os.path.join(cache_dir, "timing.json")) as f:
            timing = json.load(f)

        print(f"[Cache] Loaded reconstruction for '{scene_name}' from {cache_dir}")
        return glb_full, glb_sparse, timing, full_qk, sparse_qk, sparse_mask, model_info

    except Exception as e:
        print(f"[Cache] Failed to load cache for '{scene_name}': {e}")
        return None


# ─── Attention Visualization Helpers ──────────────────────────────────────────

def _compute_patch_grid(H, W, patch_size):
    return H // patch_size, W // patch_size


def _xy_to_patch_index(x, y, patch_size, grid_w):
    col = int(x) // patch_size
    row = int(y) // patch_size
    return row * grid_w + col


def _heatmap_to_rgb_uint8(hm):
    hm = np.clip(np.asarray(hm), 0.0, 1.0)
    rgba = matplotlib.colormaps["magma"](hm)
    return (rgba[..., :3] * 255.0).astype(np.uint8)


def _overlay_heatmap_on_image(image_rgb_uint8, heat_rgb_uint8, alpha=0.45):
    img = image_rgb_uint8.astype(np.float32)
    heat = heat_rgb_uint8.astype(np.float32)
    return np.clip(img * (1.0 - alpha) + heat * alpha, 0, 255).astype(np.uint8)


def _compute_attention_from_qk(qk_list, q_global_idx, scale, mask=None):
    """
    Compute softmax attention for a single query token from stored Q/K matrices.
    Args:
        qk_list: list of (Q, K) tuples, each [B, H, N, D] in fp16 on CPU
        q_global_idx: global token index of the query
        scale: attention scale factor (head_dim ** -0.5)
        mask: optional [B, N, N] binary mask (fp16 on CPU) for sparse attention
    Returns:
        list of attention vectors [H, N] per layer (float32 numpy)
    """
    results = []
    qi = int(q_global_idx)
    for q_mat, k_mat in qk_list:
        if q_mat is None or k_mat is None:
            results.append(None)
            continue
        # q_mat: [B, H, N, D] -> select query row -> [B, H, 1, D]
        q_sel = q_mat[:, :, qi:qi+1, :].float()
        k_f = k_mat.float()
        attn = (q_sel * scale) @ k_f.transpose(-2, -1)  # [B, H, 1, N]
        if mask is not None:
            mask_sel = mask[:, qi:qi+1, :].unsqueeze(1).float()  # [B, 1, 1, N]
            attn = attn.masked_fill(mask_sel == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        results.append(attn[0, :, 0, :].numpy())  # [H, N]
    return results


def _compute_attention_from_qk_layer(qk_list, layer_idx, q_global_idx, scale, mask=None):
    """
    Compute softmax attention for a single query token, for exactly one layer.
    Returns attention vectors [H, N] (float32 numpy) or None.
    """
    if qk_list is None:
        return None
    li = int(layer_idx)
    if li < 0 or li >= len(qk_list):
        return None
    q_mat, k_mat = qk_list[li]
    if q_mat is None or k_mat is None:
        return None

    qi = int(q_global_idx)
    # Ensure computations happen on CPU to avoid CUDA allocations.
    if q_mat.is_cuda:
        q_mat = q_mat.cpu()
    if k_mat.is_cuda:
        k_mat = k_mat.cpu()

    q_sel = q_mat[:, :, qi:qi + 1, :].float()  # [B, H, 1, D]
    k_f = k_mat.float()  # [B, H, N, D]
    attn = (q_sel * scale) @ k_f.transpose(-2, -1)  # [B, H, 1, N]

    if mask is not None:
        if mask.is_cuda:
            mask = mask.cpu()
        mask_sel = mask[:, qi:qi + 1, :].unsqueeze(1).float()  # [B, 1, 1, N]
        attn = attn.masked_fill(mask_sel == 0, float('-inf'))

    attn = attn.softmax(dim=-1)
    attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
    return attn[0, :, 0, :].numpy()  # [H, N]


def _attn_heatmap_overlay(attn_vec_HN, head_idx, frame_idx, frame_rgb_uint8,
                           grid_h, grid_w, patch_start_idx, P,
                           out_size=64, alpha=0.45):
    """Render a single attention heatmap thumbnail overlaid on the frame.
    attn_vec_HN: [H, N] numpy array of attention weights for one layer.
    """
    vec = attn_vec_HN[int(head_idx)]  # [N]
    P_total = patch_start_idx + P
    start = int(frame_idx) * P_total + patch_start_idx
    end = start + P
    a = vec[start:end].reshape(grid_h, grid_w)
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max()
    heat_rgb = _heatmap_to_rgb_uint8(a)
    # Resize heatmap to match the original frame resolution (preserves aspect ratio).
    H_img, W_img = frame_rgb_uint8.shape[0], frame_rgb_uint8.shape[1]
    heat_im = PILImage.fromarray(heat_rgb).resize(
        (W_img, H_img), resample=PILImage.BILINEAR
    )
    return _overlay_heatmap_on_image(frame_rgb_uint8, np.array(heat_im), alpha=alpha)


def _attn_heatmap_overlay_with_no_attention(attn_vec_HN, head_idx, frame_idx, frame_rgb_uint8,
                                            grid_h, grid_w, patch_start_idx, P,
                                            out_size=64, alpha=0.45, text="no attention"):
    """
    Same as `_attn_heatmap_overlay`, but if the attention for this frame is all zeros
    (e.g. fully masked in sparse attention), render a label on the frame.
    """
    vec = attn_vec_HN[int(head_idx)]  # [N]
    P_total = patch_start_idx + P
    start = int(frame_idx) * P_total + patch_start_idx
    end = start + P
    a_raw = vec[start:end].reshape(grid_h, grid_w)

    # Detect "no attention": all attention weights for this frame segment are 0.
    if float(np.max(a_raw)) <= 0.0:
        frame_im = PILImage.fromarray(frame_rgb_uint8).convert("RGB")
        draw = ImageDraw.Draw(frame_im)
        bbox = draw.textbbox((0, 0), text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        pad = 4
        draw.rectangle([0, 0, tw + pad * 2, th + pad * 2], fill=(0, 0, 0))
        draw.text((pad, pad), text, fill=(255, 255, 255))
        return np.array(frame_im)

    # Otherwise: normal heatmap overlay.
    a = a_raw
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max()
    heat_rgb = _heatmap_to_rgb_uint8(a)
    H_img, W_img = frame_rgb_uint8.shape[0], frame_rgb_uint8.shape[1]
    heat_im = PILImage.fromarray(heat_rgb).resize(
        (W_img, H_img), resample=PILImage.BILINEAR
    )
    return _overlay_heatmap_on_image(frame_rgb_uint8, np.array(heat_im), alpha=alpha)


def _render_attention_grid_image(attn_all_HN, frames_uint8, S, mode, sel, frame_idx,
                                  grid_h, grid_w, patch_start_idx, P,
                                  num_heads, num_layers, cell=80, gap=2, label_prefix=""):
    """Render attention grid as a PIL Image.
    attn_all_HN: list of [H, N] numpy arrays, one per layer (from _compute_attention_from_qk).
    """
    cell = int(cell)
    gap = int(gap)
    header_h = 22
    left_w = 52

    def _blank(w, h, color=(255, 255, 255)):
        return PILImage.new("RGB", (int(w), int(h)), color)

    def _paste_rgb(img_pil, rgb_uint8, x, y):
        im = PILImage.fromarray(rgb_uint8)
        if im.size != (cell, cell):
            im = im.resize((cell, cell), PILImage.Resampling.BILINEAR)
        img_pil.paste(im, (int(x), int(y)))

    def _draw_centered(draw, box, text, fill=(55, 65, 81)):
        x0, y0, x1, y1 = box
        try:
            tw, th = draw.textbbox((0, 0), text)[2:]
        except Exception:
            tw, th = draw.textsize(text)
        tx = int(x0 + (max(1, x1 - x0) - tw) / 2)
        ty = int(y0 + (max(1, y1 - y0) - th) / 2)
        draw.text((tx, ty), text, fill=fill)

    if mode == "default":
        num_heads_show = min(int(num_heads), 8)
        ncols = int(num_layers)
        nrows = int(num_heads_show)
        W = left_w + gap + ncols * cell + max(0, ncols - 1) * gap
        H = header_h + gap + nrows * cell + max(0, nrows - 1) * gap + 18
        out = _blank(W, H)
        draw = ImageDraw.Draw(out)
        if label_prefix:
            _draw_centered(draw, (0, 0, W, 16), label_prefix, fill=(30, 100, 200))
        for l in range(num_layers):
            x = left_w + gap + l * (cell + gap)
            _draw_centered(draw, (x, 16, x + cell, 16 + header_h), f"L{l}")
        frame_rgb = frames_uint8[int(frame_idx)]
        for h in range(num_heads_show):
            y = 16 + header_h + gap + h * (cell + gap)
            _draw_centered(draw, (0, y, left_w, y + cell), f"H{h}")
            for l in range(num_layers):
                x = left_w + gap + l * (cell + gap)
                if attn_all_HN[l] is not None:
                    rgb = _attn_heatmap_overlay(
                        attn_all_HN[l], h, frame_idx, frame_rgb,
                        grid_h, grid_w, patch_start_idx, P, out_size=cell
                    )
                    _paste_rgb(out, rgb, x, y)
        return out

    if mode == "head":
        head_idx = int(sel)
        ncols = int(num_layers)
        nrows = int(S)
        W = left_w + gap + ncols * cell + max(0, ncols - 1) * gap
        H = header_h + gap + nrows * cell + max(0, nrows - 1) * gap + 18
        out = _blank(W, H)
        draw = ImageDraw.Draw(out)
        if label_prefix:
            _draw_centered(draw, (0, 0, W, 16), label_prefix, fill=(30, 100, 200))
        for l in range(num_layers):
            x = left_w + gap + l * (cell + gap)
            _draw_centered(draw, (x, 16, x + cell, 16 + header_h), f"L{l}")
        for f in range(S):
            y = 16 + header_h + gap + f * (cell + gap)
            _draw_centered(draw, (0, y, left_w, y + cell), f"F{f}")
            frame_rgb = frames_uint8[int(f)]
            for l in range(num_layers):
                x = left_w + gap + l * (cell + gap)
                if attn_all_HN[l] is not None:
                    rgb = _attn_heatmap_overlay(
                        attn_all_HN[l], head_idx, f, frame_rgb,
                        grid_h, grid_w, patch_start_idx, P, out_size=cell
                    )
                    _paste_rgb(out, rgb, x, y)
        return out

    if mode == "layer":
        layer_idx = int(sel)
        ncols = int(S)
        nrows = int(num_heads)
        W = left_w + gap + ncols * cell + max(0, ncols - 1) * gap
        H = header_h + gap + nrows * cell + max(0, nrows - 1) * gap + 18
        out = _blank(W, H)
        draw = ImageDraw.Draw(out)
        if label_prefix:
            _draw_centered(draw, (0, 0, W, 16), label_prefix, fill=(30, 100, 200))
        for f in range(S):
            x = left_w + gap + f * (cell + gap)
            _draw_centered(draw, (x, 16, x + cell, 16 + header_h), f"F{f}")
        if attn_all_HN[layer_idx] is not None:
            for h in range(num_heads):
                y = 16 + header_h + gap + h * (cell + gap)
                _draw_centered(draw, (0, y, left_w, y + cell), f"H{h}")
                for f in range(S):
                    x = left_w + gap + f * (cell + gap)
                    frame_rgb = frames_uint8[int(f)]
                    rgb = _attn_heatmap_overlay(
                        attn_all_HN[layer_idx], h, f, frame_rgb,
                        grid_h, grid_w, patch_start_idx, P, out_size=cell
                    )
                    _paste_rgb(out, rgb, x, y)
        return out

    return _blank(1, 1)


def _render_patch_grid_preview(frame_rgb_uint8, patch_size, q_patch, grid_w, grid_h):
    """Render frame with patch grid and optional query highlight."""
    im = PILImage.fromarray(frame_rgb_uint8.copy()).convert("RGBA")
    draw = ImageDraw.Draw(im, "RGBA")
    H, W = frame_rgb_uint8.shape[0], frame_rgb_uint8.shape[1]

    line = (255, 255, 255, 70)
    for x in range(0, W + 1, patch_size):
        draw.line([(x, 0), (x, H)], fill=line, width=1)
    for y in range(0, H + 1, patch_size):
        draw.line([(0, y), (W, y)], fill=line, width=1)

    if q_patch is not None:
        row = int(q_patch) // int(grid_w)
        col = int(q_patch) % int(grid_w)
        x0 = col * patch_size
        y0 = row * patch_size
        x1 = min(W - 1, x0 + patch_size)
        y1 = min(H - 1, y0 + patch_size)
        draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 255), width=3)

    return np.array(im.convert("RGB"))


# ─── Gradio UI ────────────────────────────────────────────────────────────────

print("Building Gradio interface...")

custom_css = """
    .view-image { min-height: 160px; cursor: crosshair; }
    .status-text * {
        font-size: 16px !important;
        font-weight: 500 !important;
    }
"""

with gr.Blocks(
    title="Token Correspondence + Attention Viewer",
    theme=gr.themes.Soft(),
    css=custom_css,
) as demo:

    # Hidden state for scene data
    scene_state = gr.State(value=None)  # will hold (images, matching_gt)

    gr.HTML("""
    <h1>🔍 Token Correspondence + Attention Viewer</h1>
    <p style="font-size:16px;">
        Visualize which tokens correspond across views in multi-view scenes.
        <strong>Select a scene → click on any view</strong>
        to see corresponding tokens (green) in all other views.
        🔴 Red = selected token &nbsp; 🟢 Green = corresponding tokens
    </p>
    """)

    # ── Scene loading ──
    with gr.Row():
        with gr.Column(scale=2):
            scene_choices = list(AVAILABLE_SCENES.keys())
            scene_selector = gr.Dropdown(
                choices=scene_choices,
                value=scene_choices[0] if scene_choices else None,
                label="Scene",
                info=f"{len(scene_choices)} scenes available",
            )
        with gr.Column(scale=1):
            load_btn = gr.Button("Load Scene", variant="primary")
        with gr.Column(scale=3):
            status_output = gr.Markdown("Select a scene and click **Load Scene**.", elem_classes=["status-text"])

    # ── Sparse vs Full Attention 3D Reconstruction Comparison ──
    gr.Markdown("---")
    gr.Markdown("""
    ## 3D Reconstruction: Full vs Sparse Attention
    Compare 3D reconstructions from VGGT using **full attention** (baseline) vs 
    **sparse attention** (using the token correspondences shown above).
    """)

    with gr.Row():
        compare_btn = gr.Button("Run Comparison", variant="primary", scale=1)
        force_refresh_cb = gr.Checkbox(label="Force refresh (ignore cache)", value=False, scale=1)
        comparison_status = gr.Markdown("Click **Run Comparison** after loading a scene.", elem_classes=["status-text"])

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**Full Attention (Baseline)**")
            full_attn_output = gr.Model3D(height=450, zoom_speed=0.5, pan_speed=0.5)

        with gr.Column(scale=1):
            gr.Markdown("**Sparse Attention (Optimized)**")
            sparse_attn_output = gr.Model3D(height=450, zoom_speed=0.5, pan_speed=0.5)

    comparison_report = gr.Markdown("", label="Performance Report")

    # ── Attention Visualization ──
    gr.Markdown("---")
    gr.Markdown("""
    ## Attention Visualization: Full vs Sparse
    Click on a token in the **Token Correspondence** row to select the query token.
    After running the comparison, the **Sparse** and **Full** attention rows will update instantly
    for the selected layer (fixed head).
    """)

    # Backend-only attention state (stored QK from comparison runs)
    num_heads = model.aggregator.global_blocks[-1].attn.num_heads
    head_dim = model.aggregator.global_blocks[-1].attn.head_dim
    attn_scale = head_dim ** -0.5
    num_layers = len(model.aggregator.global_blocks)
    attn_holder = {
        "full_qk": None,       # list of (Q,K) tuples per layer
        "sparse_qk": None,     # list of (Q,K) tuples per layer
        "sparse_mask": None,   # [B, N, N] padded mask
        "frames_uint8": None,  # [S, H, W, 3] uint8
        "grid_h": None, "grid_w": None,
        "patch_start_idx": None, "P": None, "S": None,
        # Cache computed attention vectors for the currently selected query token
        # (q_frame, q_patch). This enables fast re-rendering when only layer/size
        # sliders change.
        "cached_q_global": None,
        "cached_full_attn_HN": None,    # list of [H, N] numpy arrays, one per layer
        "cached_sparse_attn_HN": None,  # list of [H, N] numpy arrays, one per layer
        # Internal flag to print alignment diagnostics once per comparison run.
        "printed_alignment_debug": False,
    }

    # ---- Controls ----
    with gr.Row():
        with gr.Column(scale=1):
            fixed_head_idx = 1 if int(num_heads) > 1 else 0
            attn_layer_idx = gr.Slider(
                0,
                max(int(num_layers) - 1, 0),
                value=min(1, int(num_layers) - 1) if int(num_layers) > 0 else 0,
                step=1,
                label=f"Layer index (Head fixed to {fixed_head_idx})",
            )
            # Larger than the old grid thumbnails for readability.
            attn_thumb_size = gr.Slider(96, 320, value=200, step=8, label="Heatmap Size")
        with gr.Column(scale=2):
            attn_status = gr.Markdown(
                "Click a frame in the first row to select a query token. "
                "Run **Comparison** to enable sparse/full attention heatmaps."
            )

    # ---- 1) Token correspondences (clickable) ----
    gr.Markdown("### Token Correspondence (click to select query token)")
    view_outputs = []
    with gr.Row():
        for i in range(4):
            with gr.Column(scale=1, min_width=200):
                img = gr.Image(
                    label=f"View {i}",
                    type="pil",
                    interactive=False,
                    elem_classes=["view-image"],
                    height=280,
                )
                view_outputs.append(img)
    with gr.Row():
        for i in range(4, 8):
            with gr.Column(scale=1, min_width=200):
                img = gr.Image(
                    label=f"View {i}",
                    type="pil",
                    interactive=False,
                    elem_classes=["view-image"],
                    height=280,
                )
                view_outputs.append(img)

    # ---- 2) Sparse attention ----
    gr.Markdown("### Sparse Attention (Head fixed)")
    attn_sparse_frame_imgs = []
    for r in range(2):
        with gr.Row():
            for c in range(4):
                f_idx = r * 4 + c
                img = gr.Image(
                    label=f"Frame {f_idx}",
                    type="numpy",
                    interactive=False,
                            height=280,
                )
                attn_sparse_frame_imgs.append(img)

    # ---- 3) Full attention ----
    gr.Markdown("### Full Attention (Head fixed)")
    attn_full_frame_imgs = []
    for r in range(2):
        with gr.Row():
            for c in range(4):
                f_idx = r * 4 + c
                img = gr.Image(
                    label=f"Frame {f_idx}",
                    type="numpy",
                    interactive=False,
                            height=280,
                )
                attn_full_frame_imgs.append(img)

    # Hidden states for attention query (set by clicking the correspondence row)
    attn_q_patch_state = gr.State(None)
    attn_q_frame_state = gr.State(0)

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def on_load_scene(scene_name):
        """Load scene, show plain views, return state."""
        images, matching_gt, status = load_scene(scene_name)
        if images is None:
            return (None,) + (None,) * NUM_VIEWS + (status,)

        scene_data = (images, matching_gt)
        plain_views = get_plain_views(images)
        return (scene_data, *plain_views, status)

    load_btn.click(
        fn=on_load_scene,
        inputs=[scene_selector],
        outputs=[scene_state] + view_outputs + [status_output],
    )

    # ── Click handler factory: one per view ──
    def make_click_handler(src_view_idx):
        def on_view_click(scene_data, layer_idx, thumb_size, evt: gr.SelectData):
            if scene_data is None:
                none8 = [None] * NUM_VIEWS
                return (
                    *none8,
                    "Load a scene first.",
                    None,
                    src_view_idx,
                    *([None] * 8),
                    *([None] * 8),
                    "Run **Comparison** first."
                )

            images, matching_gt = scene_data
            pixel_x, pixel_y = evt.index[0], evt.index[1]
            result_images, corr_status = compute_correspondences(
                scene_data, src_view_idx, pixel_x, pixel_y
            )

            # Map click -> query token indices using the same token grid as matching_gt.
            # This avoids mismatches when the displayed image resolution differs from
            # the attention frame resolution used by the model.
            token_h = int(matching_gt.shape[1])
            token_w = int(matching_gt.shape[2])
            src_th = min(int(pixel_y) // TOKEN_PATCH, token_h - 1)
            src_tw = min(int(pixel_x) // TOKEN_PATCH, token_w - 1)
            src_th = max(0, src_th)
            src_tw = max(0, src_tw)

            # Attention uses its own patch grid (grid_h/grid_w). Convert (th,tw) to its patch index.
            attn_grid_w = int(attn_holder.get("grid_w") or token_w)
            q_patch = src_th * attn_grid_w + src_tw
            attn_grid_h = int(attn_holder.get("grid_h") or token_h)
            q_patch = max(0, min(int(q_patch), attn_grid_h * attn_grid_w - 1))
            q_frame = int(src_view_idx)

            full_imgs, sparse_imgs, attn_status_str = _render_attention_frames_head_layer(
                q_patch=q_patch,
                q_frame=q_frame,
                layer_idx=layer_idx,
                head_idx=fixed_head_idx,
                thumb_size=thumb_size,
            )

            return (
                *result_images,
                corr_status,
                q_patch,
                q_frame,
                *full_imgs,
                *sparse_imgs,
                attn_status_str,
            )

        return on_view_click

    for i in range(NUM_VIEWS):
        view_outputs[i].select(
            fn=make_click_handler(i),
            inputs=[scene_state, attn_layer_idx, attn_thumb_size],
            outputs=view_outputs
            + [status_output, attn_q_patch_state, attn_q_frame_state]
            + attn_full_frame_imgs
            + attn_sparse_frame_imgs
            + [attn_status],
        )

    # ── Comparison button callback ──
    def on_run_comparison(scene_data, scene_name, q_patch, q_frame, layer_idx, thumb_size, force_refresh):
        """Run full vs sparse attention, generate 3D reconstructions, and store QK for attention viz.
        Results are cached to disk so subsequent runs for the same scene are instant."""
        if scene_data is None:
            none8 = [None] * 8
            return None, None, "", "Load a scene first.", *none8, *none8, "Load a scene first."

        images_np, matching_gt_np = scene_data

        try:
            # ── Try disk cache first ──────────────────────────────────────────
            cached = None if force_refresh else _load_reconstruction_cache(scene_name)
            if cached is not None:
                glb_full, glb_sparse, timing, full_qk, sparse_qk, sparse_mask, model_info = cached
                from_cache = True
            else:
                # ── Cache miss (or forced refresh): run the models ────────────
                from_cache = False
                target_dir = save_scene_images_to_dir(images_np)

                scene_dir = AVAILABLE_SCENES.get(scene_name)
                matching_gt_tensor = None
                if scene_dir:
                    gt_path = os.path.join(scene_dir, "matching_gt.npy")
                    if os.path.exists(gt_path):
                        matching_gt_tensor = torch.from_numpy(np.load(gt_path)).bool().to(device)
                if matching_gt_tensor is None:
                    matching_gt_tensor = MATCHING_GT

                full_preds, sparse_preds, timing, full_qk, sparse_qk, sparse_mask, model_info = \
                    run_model_comparison(target_dir, matching_gt_tensor)

                # Generate GLB for full attention
                glb_full = os.path.join(target_dir, "full_attention.glb")
                scene_full = predictions_to_glb(
                    full_preds, conf_thres=50.0, filter_by_frames="All",
                    mask_black_bg=False, mask_white_bg=False, show_cam=True,
                    mask_sky=False, target_dir=target_dir,
                    prediction_mode="Depthmap and Camera Branch",
                )
                scene_full.export(file_obj=glb_full)

                glb_sparse = None
                if sparse_preds is not None:
                    glb_sparse = os.path.join(target_dir, "sparse_attention.glb")
                    scene_sparse = predictions_to_glb(
                        sparse_preds, conf_thres=50.0, filter_by_frames="All",
                        mask_black_bg=False, mask_white_bg=False, show_cam=True,
                        mask_sky=False, target_dir=target_dir,
                        prediction_mode="Depthmap and Camera Branch",
                    )
                    scene_sparse.export(file_obj=glb_sparse)

                # Persist to disk so the next click is instant
                _save_reconstruction_cache(
                    scene_name, timing, full_qk, sparse_qk, sparse_mask,
                    model_info, glb_full, glb_sparse,
                )

                gc.collect()
                torch.cuda.empty_cache()

            # ── Populate in-memory attention holder ───────────────────────────
            attn_holder["full_qk"] = full_qk
            attn_holder["sparse_qk"] = sparse_qk
            attn_holder["sparse_mask"] = sparse_mask
            attn_holder["frames_uint8"] = model_info["frames_uint8"]
            attn_holder["grid_h"] = model_info["grid_h"]
            attn_holder["grid_w"] = model_info["grid_w"]
            attn_holder["patch_start_idx"] = model_info["patch_start_idx"]
            attn_holder["P"] = model_info["P"]
            attn_holder["S"] = model_info["S"]
            attn_holder["cached_q_global"] = None
            attn_holder["cached_full_attn_HN"] = None
            attn_holder["cached_sparse_attn_HN"] = None
            attn_holder["printed_alignment_debug"] = False

            qk_total_mb = sum((q.nelement()*2 + k.nelement()*2) for q,k in full_qk if q is not None) / 1e6
            if sparse_qk:
                qk_total_mb += sum((q.nelement()*2 + k.nelement()*2) for q,k in sparse_qk if q is not None) / 1e6
            print(f"  QK stored for attention viz: {qk_total_mb:.0f}MB total")

            cache_note = " *(loaded from cache)*" if from_cache else " *(saved to cache)*"
            speedup_str = ""
            if "speedup" in timing:
                speedup_str = f" | {timing['speedup']:.2f}× speedup"
            comparison_status = (
                f"Comparison complete{cache_note}. QK stored ({qk_total_mb:.0f}MB){speedup_str} — "
                "attention updated for the selected token."
            )

            full_imgs, sparse_imgs, attn_status_str = _render_attention_frames_head_layer(
                q_patch=q_patch,
                q_frame=q_frame,
                layer_idx=layer_idx,
                head_idx=fixed_head_idx,
                thumb_size=thumb_size,
            )

            return (
                glb_full,
                glb_sparse,
                "",
                comparison_status,
                *full_imgs,
                *sparse_imgs,
                attn_status_str,
            )

        except Exception as e:
            import traceback
            return None, None, f"Error: {e}\n\n{traceback.format_exc()}", f"Error: {e}", *([None] * 8), *([None] * 8), f"Error: {e}"

    compare_btn.click(
        fn=on_run_comparison,
        inputs=[
            scene_state,
            scene_selector,
            attn_q_patch_state,
            attn_q_frame_state,
            attn_layer_idx,
            attn_thumb_size,
            force_refresh_cb,
        ],
        outputs=(
            [full_attn_output, sparse_attn_output, comparison_report, comparison_status]
            + attn_full_frame_imgs
            + attn_sparse_frame_imgs
            + [attn_status]
        ),
    )

    # ── Attention visualization callbacks ──

    def _render_attention_frames_head_layer(q_patch, q_frame, layer_idx, head_idx, thumb_size):
        """
        Render 8 per-target-frame heatmaps for a single (head, layer) pair.
        For each of frames 0..7: overlay attention for the selected query token.
        Returns (full_images_8, sparse_images_8, status).
        """
        full_qk = attn_holder.get("full_qk")
        if full_qk is None:
            none8 = [None] * 8
            return none8, none8, "Run **Comparison** first to capture attention data."
        if q_patch is None:
            none8 = [None] * 8
            return none8, none8, "Click a token in the correspondence row to select a query."

        frames_uint8 = attn_holder["frames_uint8"]
        S = int(attn_holder["S"])
        grid_h = int(attn_holder["grid_h"])
        grid_w = int(attn_holder["grid_w"])
        psi = int(attn_holder["patch_start_idx"])
        P = int(attn_holder["P"])
        P_total = psi + P
        q_global = int(q_frame) * P_total + (psi + int(q_patch))

        # Cache computed attention for (q_global, layer_idx) only.
        # We avoid computing attention for ALL layers at once to reduce memory spikes.
        if attn_holder.get("cached_q_global") != q_global:
            sparse_qk = attn_holder.get("sparse_qk")
            attn_holder["cached_q_global"] = q_global
            attn_holder["cached_full_attn_HN"] = {}
            attn_holder["cached_sparse_attn_HN"] = {} if sparse_qk is not None else None

        cached_full_attn_HN = attn_holder.get("cached_full_attn_HN")  # dict: layer_idx -> [H, N]
        cached_sparse_attn_HN = attn_holder.get("cached_sparse_attn_HN")  # dict: layer_idx -> [H, N] or None

        frame_count = min(S, 8)
        out_size = int(thumb_size)
        none8 = [None] * 8

        def render_from_attn(attn_dict, *, sparse=False):
            if attn_dict is None:
                return none8
            if layer_idx is None or int(layer_idx) < 0:
                return none8
            li = int(layer_idx)
            if li not in attn_dict:
                if sparse:
                    sparse_qk = attn_holder.get("sparse_qk")
                    sparse_mask = attn_holder.get("sparse_mask")
                    attn_dict[li] = _compute_attention_from_qk_layer(
                        sparse_qk, li, q_global, attn_scale, mask=sparse_mask
                    )
                else:
                    attn_dict[li] = _compute_attention_from_qk_layer(
                        full_qk, li, q_global, attn_scale, mask=None
                    )
            attn_vec_HN = attn_dict.get(li)
            if attn_vec_HN is None:
                # e.g. this layer had no captured Q/K
                return none8
            imgs = []
            for f in range(frame_count):
                frame_rgb = frames_uint8[int(f)]
                if sparse:
                    imgs.append(
                        _attn_heatmap_overlay_with_no_attention(
                            attn_vec_HN, int(head_idx), f, frame_rgb,
                            grid_h, grid_w, psi, P,
                            out_size=out_size,
                        )
                    )
                else:
                    imgs.append(
                        _attn_heatmap_overlay(
                            attn_vec_HN, int(head_idx), f, frame_rgb,
                            grid_h, grid_w, psi, P,
                            out_size=out_size,
                        )
                    )
            imgs += [None] * (8 - frame_count)
            return imgs

        full_imgs = render_from_attn(cached_full_attn_HN, sparse=False)
        sparse_imgs = render_from_attn(cached_sparse_attn_HN, sparse=True)

        row = int(q_patch) // grid_w
        col = int(q_patch) % grid_w
        status = (
            f"Query: frame={int(q_frame)}, patch={int(q_patch)} (row={row}, col={col}) "
            f"— layer={int(layer_idx)}, head={int(head_idx)}"
        )
        return full_imgs, sparse_imgs, status

    def on_attn_settings_change(q_patch, q_frame, layer_idx, thumb_size):
        """Re-render heatmaps when layer/size sliders change."""
        fixed_head_idx = 1 if int(num_heads) > 1 else 0
        full_imgs, sparse_imgs, status = _render_attention_frames_head_layer(
            q_patch=q_patch,
            q_frame=q_frame,
            layer_idx=layer_idx,
            head_idx=fixed_head_idx,
            thumb_size=thumb_size,
        )
        return (*full_imgs, *sparse_imgs, status)

    for ctrl in [attn_layer_idx, attn_thumb_size]:
        ctrl.change(
            on_attn_settings_change,
            inputs=[attn_q_patch_state, attn_q_frame_state, attn_layer_idx, attn_thumb_size],
            outputs=attn_full_frame_imgs + attn_sparse_frame_imgs + [attn_status],
        )

# ─── Launch ───────────────────────────────────────────────────────────────────
print("Starting server...")
demo.queue(max_size=10).launch(
    show_error=True,
    share=True,
    allowed_paths=[DEMO_CACHE_DIR],
)
