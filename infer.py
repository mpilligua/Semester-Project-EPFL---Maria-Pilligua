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
import base64
import io
import time
from PIL import Image, ImageDraw
import matplotlib.cm as cm
import trimesh

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


def _predictions_to_colored_pointcloud_glb(predictions, images_b, output_path, conf_thres=0.0, max_points=250000):
    # predictions expected tensors on device
    if "world_points" in predictions:
        wp = predictions["world_points"]  # [B,S,H,W,3]
        wp_conf = predictions.get("world_points_conf")
    elif "depth" in predictions and "pose_enc" in predictions:
        extr, intr = pose_encoding_to_extri_intri(predictions["pose_enc"], images_b.shape[-2:])
        wp = unproject_depth_map_to_point_map(predictions["depth"], extr, intr)
        wp_conf = predictions.get("depth_conf")
    else:
        raise ValueError("Predictions must contain world_points or (depth + pose_enc).")

    # move to cpu numpy
    if isinstance(wp, torch.Tensor):
        wp = wp.detach().cpu().numpy()
    if isinstance(wp_conf, torch.Tensor):
        wp_conf = wp_conf.detach().cpu().numpy()

    if isinstance(images_b, torch.Tensor):
        imgs = images_b.detach().cpu().numpy()
    else:
        imgs = images_b

    # squeeze batch
    wp = wp[0]
    imgs = imgs[0]  # [S,3,H,W]

    S, _, H, W = imgs.shape
    imgs_hw3 = np.transpose(imgs, (0, 2, 3, 1))
    imgs_hw3 = np.clip(imgs_hw3, 0.0, 1.0)

    # build masks
    valid = np.isfinite(wp).all(axis=-1)
    if wp_conf is not None:
        # wp_conf can be [B,S,H,W] or [B,S,H,W,1]
        wp_conf = wp_conf[0]
        if wp_conf.ndim == 4:
            wp_conf = wp_conf[..., 0]
        valid = valid & (wp_conf >= conf_thres)

    # flatten
    pts = wp.reshape(-1, 3)
    cols = imgs_hw3.reshape(-1, 3)
    valid_f = valid.reshape(-1)

    pts = pts[valid_f]
    cols = cols[valid_f]

    # Convert to a more standard 3D viewer convention (Y-up).
    # VGGT world coordinates are often in an image-like convention where Y increases down.
    # Flipping Y makes the reconstruction appear upright in glTF viewers.
    pts[:, 1] *= -1.0

    # subsample
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx]

    cols_u8 = (cols * 255.0).astype(np.uint8)

    pc = trimesh.points.PointCloud(vertices=pts.astype(np.float32), colors=cols_u8)
    scene = trimesh.Scene(pc)
    scene.export(output_path)
    return output_path


def _predictions_to_lightweight_cpu(predictions):
    out = {}
    for k in ("world_points", "world_points_conf", "depth", "depth_conf", "pose_enc"):
        if k in predictions:
            v = predictions[k]
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().cpu()
            else:
                out[k] = v
    return out


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
                preds = model(images_b.to(device))
        else:
            preds = model(images_b.to(device))

    attn = model.aggregator.get_captured_global_attention()
    model.aggregator.disable_global_attention_capture()

    if attn is None:
        raise RuntimeError("Attention capture returned None.")

    return attn, grid_h, grid_w, patch_start_idx, P


def _compute_and_capture_attention_all_global_blocks(model, images_b, device, dtype, q_patch, q_frame=0):
    B, S, _, H, W = images_b.shape
    patch_size = model.aggregator.patch_size
    grid_h, grid_w = _compute_patch_grid(H, W, patch_size)
    P = grid_h * grid_w
    patch_start_idx = model.aggregator.patch_start_idx

    P_total = patch_start_idx + P
    q_global = int(q_frame) * P_total + (patch_start_idx + q_patch)

    model.aggregator.enable_global_attention_capture_all(query_indices=[q_global])

    with torch.no_grad():
        if dtype != torch.float32:
            with torch.cuda.amp.autocast(dtype=dtype):
                preds = model(images_b.to(device))
        else:
            preds = model(images_b.to(device))

    attn_all = model.aggregator.get_captured_global_attention_all()
    model.aggregator.disable_global_attention_capture()

    if attn_all is None or any(a is None for a in attn_all):
        raise RuntimeError("Attention capture returned None for one or more blocks.")

    return preds, attn_all, grid_h, grid_w, patch_start_idx, P


def _compute_and_capture_attention_for_block(model, images_b, device, dtype, q_patch, block_idx: int):
    B, S, _, H, W = images_b.shape
    patch_size = model.aggregator.patch_size
    grid_h, grid_w = _compute_patch_grid(H, W, patch_size)
    P = grid_h * grid_w
    patch_start_idx = model.aggregator.patch_start_idx

    P_total = patch_start_idx + P
    q_global = 0 * P_total + (patch_start_idx + q_patch)

    model.aggregator.enable_global_attention_capture(block_idx=int(block_idx), query_indices=[q_global])

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


def _rgb_uint8_to_data_uri(rgb_uint8):
    im = Image.fromarray(rgb_uint8)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _render_patch_grid_preview(frame0_rgb_uint8, patch_size, q_patch, grid_w, grid_h):
    im = Image.fromarray(frame0_rgb_uint8.copy()).convert("RGBA")
    draw = ImageDraw.Draw(im, "RGBA")
    H, W = frame0_rgb_uint8.shape[0], frame0_rgb_uint8.shape[1]

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


def _attention_thumbnail_rgb(attn_block, S, frame_idx, grid_h, grid_w, patch_start_idx, P, head_idx, out_size=64):
    attn_block = attn_block[0]
    vec = attn_block[int(head_idx), 0]
    P_total = patch_start_idx + P
    start = int(frame_idx) * P_total + patch_start_idx
    end = start + P
    a = vec[start:end].detach().cpu().float().view(grid_h, grid_w).numpy()
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max()
    rgb = _heatmap_to_rgb_uint8(a)
    im = Image.fromarray(rgb)
    im = im.resize((out_size, out_size), resample=Image.NEAREST)
    return np.array(im)


def _attention_thumbnail_overlay_rgb(attn_block, S, frame_idx, frame_rgb_uint8, grid_h, grid_w, patch_start_idx, P, head_idx, out_size=64, alpha=0.45):
    heat_rgb = _attention_thumbnail_rgb(attn_block, S, frame_idx, grid_h, grid_w, patch_start_idx, P, head_idx, out_size=out_size)
    frame_im = Image.fromarray(frame_rgb_uint8)
    frame_im = frame_im.resize((out_size, out_size), resample=Image.BILINEAR)
    frame_small = np.array(frame_im)
    return _overlay_heatmap_on_image(frame_small, heat_rgb, alpha=alpha)


def _render_attention_grid_html(attn_all, frames_uint8, S, mode, sel, frame_idx, grid_h, grid_w, patch_start_idx, P, num_heads, num_layers, cell=96, gap=2):
    cell = int(cell)
    gap = int(gap)
    header_h = 22
    left_w = 46

    style = f"""
    <style>
      .attn-wrap {{ overflow: auto; width: 100%; height: 100%; border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px; }}
      .attn-grid {{ display: grid; gap: {gap}px; align-items: center; }}
      .attn-top {{ position: sticky; top: 0; background: white; z-index: 3; height: {header_h}px; display:flex; align-items:center; justify-content:center; font-size: 11px; color: #374151; }}
      .attn-left {{ position: sticky; left: 0; background: white; z-index: 2; display:flex; align-items:center; justify-content:center; font-size: 11px; color: #374151; }}
      .attn-corner {{ position: sticky; top: 0; left: 0; z-index: 4; background: white; height: {header_h}px; }}
      .attn-img {{ width: {cell}px; height: {cell}px; border-radius: 6px; border: 1px solid #f3f4f6; cursor: zoom-in; }}
      .attn-cell {{ display:flex; align-items:center; justify-content:center; }}
    </style>
    """

    if mode == "default":
        # Rendering all heads × layers as base64 images can be extremely heavy in the browser.
        # Cap heads in the default overview; use "head" / "layer" modes for full detail.
        num_heads_show = min(int(num_heads), 8)
        ncols = num_layers
        parts = [style, '<div class="attn-wrap">', f'<div class="attn-grid" style="grid-template-columns: {left_w}px repeat({ncols}, {cell}px);">']
        parts.append('<div class="attn-corner"></div>')
        for l in range(num_layers):
            parts.append(f'<div class="attn-top">L{l}</div>')

        frame_rgb_uint8 = frames_uint8[int(frame_idx)]
        for h in range(num_heads_show):
            parts.append(f'<div class="attn-left">H{h}</div>')
            for l in range(num_layers):
                rgb = _attention_thumbnail_overlay_rgb(
                    attn_all[l], S, frame_idx, frame_rgb_uint8, grid_h, grid_w, patch_start_idx, P, h, out_size=cell
                )
                uri = _rgb_uint8_to_data_uri(rgb)
                parts.append(f'<div class="attn-cell"><a href="{uri}" target="_self"><img class="attn-img" src="{uri}" loading="lazy" decoding="async"/></a></div>')
        parts.append('</div></div>')
        return "".join(parts)

    if mode == "head":
        head_idx = int(sel)
        ncols = num_layers
        parts = [style, '<div class="attn-wrap">', f'<div class="attn-grid" style="grid-template-columns: {left_w}px repeat({ncols}, {cell}px);">']
        parts.append('<div class="attn-corner"></div>')
        for l in range(num_layers):
            parts.append(f'<div class="attn-top">L{l}</div>')
        for f in range(S):
            parts.append(f'<div class="attn-left">F{f}</div>')
            frame_rgb_uint8 = frames_uint8[int(f)]
            for l in range(num_layers):
                rgb = _attention_thumbnail_overlay_rgb(
                    attn_all[l], S, f, frame_rgb_uint8, grid_h, grid_w, patch_start_idx, P, head_idx, out_size=cell
                )
                uri = _rgb_uint8_to_data_uri(rgb)
                parts.append(f'<div class="attn-cell"><a href="{uri}" target="_self"><img class="attn-img" src="{uri}" loading="lazy" decoding="async"/></a></div>')
        parts.append('</div></div>')
        return "".join(parts)

    if mode == "layer":
        layer_idx = int(sel)
        ncols = S
        parts = [style, '<div class="attn-wrap">', f'<div class="attn-grid" style="grid-template-columns: {left_w}px repeat({ncols}, {cell}px);">']
        parts.append('<div class="attn-corner"></div>')
        for f in range(S):
            parts.append(f'<div class="attn-top">F{f}</div>')
        for h in range(num_heads):
            parts.append(f'<div class="attn-left">H{h}</div>')
            for f in range(S):
                frame_rgb_uint8 = frames_uint8[int(f)]
                rgb = _attention_thumbnail_overlay_rgb(
                    attn_all[layer_idx], S, f, frame_rgb_uint8, grid_h, grid_w, patch_start_idx, P, h, out_size=cell
                )
                uri = _rgb_uint8_to_data_uri(rgb)
                parts.append(f'<div class="attn-cell"><a href="{uri}" target="_self"><img class="attn-img" src="{uri}" loading="lazy" decoding="async"/></a></div>')
        parts.append('</div></div>')
        return "".join(parts)

    return style


def _render_attention_grid_image(attn_all, frames_uint8, S, mode, sel, frame_idx, grid_h, grid_w, patch_start_idx, P, num_heads, num_layers, cell=96, gap=2):
    cell = int(cell)
    gap = int(gap)
    header_h = 22
    left_w = 46

    def _blank(w, h, color=(255, 255, 255)):
        return Image.new("RGB", (int(w), int(h)), color)

    def _paste_rgb(img_pil, rgb_uint8, x, y):
        im = Image.fromarray(rgb_uint8)
        if im.size != (cell, cell):
            im = im.resize((cell, cell), Image.Resampling.BILINEAR)
        img_pil.paste(im, (int(x), int(y)))

    def _draw_centered(draw, box, text, fill=(55, 65, 81)):
        x0, y0, x1, y1 = box
        w = max(1, int(x1 - x0))
        h = max(1, int(y1 - y0))
        try:
            tw, th = draw.textbbox((0, 0), text)[2:]
        except Exception:
            tw, th = draw.textsize(text)
        tx = int(x0 + (w - tw) / 2)
        ty = int(y0 + (h - th) / 2)
        draw.text((tx, ty), text, fill=fill)

    if mode == "default":
        num_heads_show = min(int(num_heads), 8)
        ncols = int(num_layers)
        nrows = int(num_heads_show)
        W = left_w + gap + ncols * cell + max(0, ncols - 1) * gap
        H = header_h + gap + nrows * cell + max(0, nrows - 1) * gap
        out = _blank(W, H)
        draw = ImageDraw.Draw(out)
        for l in range(num_layers):
            x = left_w + gap + l * (cell + gap)
            _draw_centered(draw, (x, 0, x + cell, header_h), f"L{l}")
        frame_rgb_uint8 = frames_uint8[int(frame_idx)]
        for h in range(num_heads_show):
            y = header_h + gap + h * (cell + gap)
            _draw_centered(draw, (0, y, left_w, y + cell), f"H{h}")
            for l in range(num_layers):
                x = left_w + gap + l * (cell + gap)
                rgb = _attention_thumbnail_overlay_rgb(
                    attn_all[l], S, frame_idx, frame_rgb_uint8, grid_h, grid_w, patch_start_idx, P, h, out_size=cell
                )
                _paste_rgb(out, rgb, x, y)
        return out

    if mode == "head":
        head_idx = int(sel)
        ncols = int(num_layers)
        nrows = int(S)
        W = left_w + gap + ncols * cell + max(0, ncols - 1) * gap
        H = header_h + gap + nrows * cell + max(0, nrows - 1) * gap
        out = _blank(W, H)
        draw = ImageDraw.Draw(out)
        for l in range(num_layers):
            x = left_w + gap + l * (cell + gap)
            _draw_centered(draw, (x, 0, x + cell, header_h), f"L{l}")
        for f in range(S):
            y = header_h + gap + f * (cell + gap)
            _draw_centered(draw, (0, y, left_w, y + cell), f"F{f}")
            frame_rgb_uint8 = frames_uint8[int(f)]
            for l in range(num_layers):
                x = left_w + gap + l * (cell + gap)
                rgb = _attention_thumbnail_overlay_rgb(
                    attn_all[l], S, f, frame_rgb_uint8, grid_h, grid_w, patch_start_idx, P, head_idx, out_size=cell
                )
                _paste_rgb(out, rgb, x, y)
        return out

    if mode == "layer":
        layer_idx = int(sel)
        ncols = int(S)
        nrows = int(num_heads)
        W = left_w + gap + ncols * cell + max(0, ncols - 1) * gap
        H = header_h + gap + nrows * cell + max(0, nrows - 1) * gap
        out = _blank(W, H)
        draw = ImageDraw.Draw(out)
        for f in range(S):
            x = left_w + gap + f * (cell + gap)
            _draw_centered(draw, (x, 0, x + cell, header_h), f"F{f}")
        for h in range(num_heads):
            y = header_h + gap + h * (cell + gap)
            _draw_centered(draw, (0, y, left_w, y + cell), f"H{h}")
            for f in range(S):
                x = left_w + gap + f * (cell + gap)
                frame_rgb_uint8 = frames_uint8[int(f)]
                rgb = _attention_thumbnail_overlay_rgb(
                    attn_all[layer_idx], S, f, frame_rgb_uint8, grid_h, grid_w, patch_start_idx, P, h, out_size=cell
                )
                _paste_rgb(out, rgb, x, y)
        return out

    return _blank(1, 1)


def _gradio_attention_ui(model, images, device, dtype, output_dir, validate=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _as_batched(images_any):
        if len(images_any.shape) == 4:
            return images_any.unsqueeze(0)
        return images_any

    images_b0 = _as_batched(images)
    B, S0, _, H0, W0 = images_b0.shape
    frames0_uint8 = (images_b0[0].detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)

    # Backend-only holder (do NOT put tensors into gr.State / component values).
    scene_holder = {
        "images_b": images_b0,
        "frames_uint8": frames0_uint8,
        "S": int(S0),
        "H": int(H0),
        "W": int(W0),
    }

    attn_holder = {
        "attn_all": None,
        "preds": None,
        "grid_h": None,
        "grid_w": None,
        "patch_start_idx": None,
        "P": None,
        "S": int(S0),
        "html_cache": {},
    }
    num_heads = model.aggregator.global_blocks[-1].attn.num_heads
    num_layers = len(model.aggregator.global_blocks)

    if validate:
        _validate_fused_vs_unfused(model, images, device, dtype)

    def _render_label(mode, sel, frame_idx):
        if mode == "head":
            return f"<div class='frame-label'>Head {int(sel)}</div>"
        if mode == "layer":
            return f"<div class='frame-label'>Layer {int(sel)}</div>"
        return f"<div class='frame-label'>frame {int(frame_idx)}</div>"

    def on_select(evt: gr.SelectData, frame_idx, query_frame, view_mode, view_sel, thumb_size, cache):
        x, y = evt.index
        patch_size = model.aggregator.patch_size
        images_b = scene_holder.get("images_b")
        if images_b is None:
            return gr.update(), gr.update(), gr.update(), cache
        B, S, _, H, W = images_b.shape
        grid_h, grid_w = _compute_patch_grid(H, W, patch_size)
        q_patch = _xy_to_patch_index(x, y, patch_size, grid_w)

        if q_patch < 0 or q_patch >= grid_h * grid_w:
            label = _render_label(str(view_mode), int(view_sel), int(frame_idx))
            return gr.update(), label, gr.update(), cache

        preds, attn_all, gh, gw, psi, P = _compute_and_capture_attention_all_global_blocks(
            model, images_b, device, dtype, q_patch, q_frame=int(query_frame)
        )
        # Store heavy outputs in backend-only holder to avoid Gradio serialization hangs.
        attn_holder["attn_all"] = [a.detach().cpu() for a in attn_all]
        attn_holder["preds"] = _predictions_to_lightweight_cpu(preds)
        attn_holder["grid_h"] = int(gh)
        attn_holder["grid_w"] = int(gw)
        attn_holder["patch_start_idx"] = int(psi)
        attn_holder["P"] = int(P)
        attn_holder["S"] = int(S)
        attn_holder["html_cache"] = {}

        cache = {
            "q_patch": int(q_patch),
            "q_frame": int(query_frame),
            "thumb_size": int(thumb_size),
        }

        t0 = time.time()
        key = (str(view_mode), int(view_sel), int(frame_idx), int(cache.get("thumb_size", 96)))
        html = _render_attention_grid_html(
            attn_holder["attn_all"],
            scene_holder["frames_uint8"],
            attn_holder["S"],
            str(view_mode),
            int(view_sel),
            int(frame_idx),
            attn_holder["grid_h"],
            attn_holder["grid_w"],
            attn_holder["patch_start_idx"],
            attn_holder["P"],
            num_heads,
            num_layers,
            cell=int(cache.get("thumb_size", 96)),
            gap=2,
        )
        attn_holder["html_cache"][key] = html
        print(f"[attn-ui] render grid html: {time.time() - t0:.3f}s")

        frame_q_uint8 = scene_holder["frames_uint8"][int(query_frame)]
        patch_preview = _render_patch_grid_preview(frame_q_uint8, patch_size, q_patch, grid_w, grid_h)
        label = _render_label(str(view_mode), int(view_sel), int(frame_idx))
        return patch_preview, label, html, cache

    def on_view_change(view_mode, view_sel, frame_idx, cache):
        if cache is None or attn_holder.get("attn_all") is None:
            return _render_label(str(view_mode), int(view_sel), int(frame_idx)), gr.update()
        thumb_size = int(cache.get("thumb_size", 96)) if isinstance(cache, dict) else 96
        key = (str(view_mode), int(view_sel), int(frame_idx), thumb_size)
        html_cache = attn_holder.get("html_cache")
        if isinstance(html_cache, dict) and key in html_cache:
            return _render_label(str(view_mode), int(view_sel), int(frame_idx)), html_cache[key]
        t0 = time.time()
        html = _render_attention_grid_html(
            attn_holder["attn_all"],
            scene_holder["frames_uint8"],
            attn_holder["S"],
            str(view_mode),
            int(view_sel),
            int(frame_idx),
            attn_holder["grid_h"],
            attn_holder["grid_w"],
            attn_holder["patch_start_idx"],
            attn_holder["P"],
            num_heads,
            num_layers,
            cell=thumb_size,
            gap=2,
        )
        if isinstance(attn_holder.get("html_cache"), dict):
            attn_holder["html_cache"][key] = html
        print(f"[attn-ui] render grid html (view change): {time.time() - t0:.3f}s")
        return _render_label(str(view_mode), int(view_sel), int(frame_idx)), html

    def on_prev(view_mode, view_sel, frame_idx, cache):
        view_mode = str(view_mode)
        if view_mode == "head":
            new_sel = max(0, int(view_sel) - 1)
            label, html = on_view_change(view_mode, new_sel, int(frame_idx), cache)
            return view_mode, new_sel, int(frame_idx), label, html
        if view_mode == "layer":
            new_sel = max(0, int(view_sel) - 1)
            label, html = on_view_change(view_mode, new_sel, int(frame_idx), cache)
            return view_mode, new_sel, int(frame_idx), label, html
        new_idx = max(0, int(frame_idx) - 1)
        label, html = on_view_change(view_mode, int(view_sel), new_idx, cache)
        return view_mode, int(view_sel), new_idx, label, html

    def on_next(view_mode, view_sel, frame_idx, cache):
        view_mode = str(view_mode)
        if view_mode == "head":
            new_sel = min(num_heads - 1, int(view_sel) + 1)
            label, html = on_view_change(view_mode, new_sel, int(frame_idx), cache)
            return view_mode, new_sel, int(frame_idx), label, html
        if view_mode == "layer":
            new_sel = min(num_layers - 1, int(view_sel) + 1)
            label, html = on_view_change(view_mode, new_sel, int(frame_idx), cache)
            return view_mode, new_sel, int(frame_idx), label, html
        S_cur = int(attn_holder.get("S", 1))
        new_idx = min(max(0, S_cur - 1), int(frame_idx) + 1)
        label, html = on_view_change(view_mode, int(view_sel), new_idx, cache)
        return view_mode, int(view_sel), new_idx, label, html

    def _run_reconstruction(conf_thres, max_points, cache):
        out_dir = output_dir / "reconstruction"
        out_dir.mkdir(parents=True, exist_ok=True)
        glb_path = out_dir / "reconstruction.glb"

        if attn_holder.get("preds") is not None and scene_holder.get("frames_uint8") is not None:
            preds = attn_holder["preds"]
            # Rebuild images_b in expected layout for coloring (B,S,3,H,W) in [0,1]
            frames = scene_holder["frames_uint8"].astype(np.float32) / 255.0
            images_for_model = np.transpose(frames, (0, 3, 1, 2))[None, ...]
        else:
            images_b = scene_holder.get("images_b")
            if images_b is None:
                return None, "No images loaded"
            images_for_model = images_b.to(device)

            with torch.no_grad():
                if dtype != torch.float32:
                    with torch.cuda.amp.autocast(dtype=dtype):
                        preds = model(images_for_model)
                else:
                    preds = model(images_for_model)

        glb = _predictions_to_colored_pointcloud_glb(
            preds,
            images_for_model,
            str(glb_path),
            conf_thres=float(conf_thres),
            max_points=int(max_points),
        )
        return glb, "Reconstruction exported"

    with gr.Blocks(
        css="""
        /* Keep layout stable: let the page size naturally and only scroll inside the grid area. */
        #attn-grid { max-height: 72vh; overflow: auto; }
        #frame_nav { align-items: center; }
        #frame_nav .centered { display:flex; justify-content:center; align-items:center; width:100%; }
        .frame-label { width:100%; text-align:center; font-size: 18px; font-weight: 600; }
        """,
    ) as demo:
        cache = gr.State(None)
        frame_state = gr.State(0)
        query_frame_state = gr.State(0)
        view_mode = gr.State("default")
        view_sel = gr.State(0)
        with gr.Row(elem_id="attn-dashboard"):
            with gr.Column(scale=1):
                patch_preview = gr.Image(value=_render_patch_grid_preview(frames0_uint8[0], model.aggregator.patch_size, None, _compute_patch_grid(H0, W0, model.aggregator.patch_size)[1], _compute_patch_grid(H0, W0, model.aggregator.patch_size)[0]), label="query frame (patches)", interactive=True)
                with gr.Row():
                    scene_dir_ui = gr.Textbox(value=str(Path("examples")), label="scene (dir)")
                    load_scene_btn = gr.Button("Load", min_width=60)
                scene_pick_ui = gr.File(file_count="directory", label="or pick folder")
                query_frame_ui = gr.Slider(minimum=0, maximum=max(0, int(S0) - 1), value=0, step=1, label="query frame")
                view_mode_ui = gr.Dropdown(choices=["default", "head", "layer"], value="default", label="view")
                thumb_size_ui = gr.Slider(minimum=48, maximum=220, value=96, step=4, label="thumb size")
                with gr.Row():
                    download_grid_btn = gr.Button("Download grid", min_width=160)
                    download_grid_file = gr.File(label="grid.png")
                gr.Markdown("### reconstruct")
                recon_view = gr.Model3D(height=420, zoom_speed=0.5, pan_speed=0.5)
                with gr.Row():
                    conf_thres = gr.Slider(minimum=0.0, maximum=10.0, value=3.0, step=0.1, label="conf")
                    max_points = gr.Slider(minimum=50000, maximum=300000, value=150000, step=10000, label="max pts")
                with gr.Row():
                    recon_btn = gr.Button("Reconstruct", variant="primary")
                recon_status = gr.Textbox(value="", label="status")

            with gr.Column(scale=3, elem_id="attn-right"):
                with gr.Row(elem_id="frame_nav"):
                    prev_btn = gr.Button("<", min_width=40)
                    frame_label = gr.HTML("<div class='frame-label'>frame 0</div>", elem_classes=["centered"])
                    next_btn = gr.Button(">", min_width=40)
                grid_html = gr.HTML(value="", elem_id="attn-grid")

        def _load_scene_dir(scene_dir: str, view_mode: str):
            p = Path(scene_dir)
            if not p.is_dir():
                return gr.update(), gr.update(), gr.update(), gr.update(), None, "", _render_label(str(view_mode), 0, 0), 0
            image_paths = sorted([x for x in p.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
            if len(image_paths) == 0:
                return gr.update(), gr.update(), gr.update(), gr.update(), None, "", _render_label(str(view_mode), 0, 0), 0
            return _load_scene_paths([str(x) for x in image_paths], view_mode)

        def _extract_file_paths(fobj):
            paths = []
            if fobj is None:
                return paths
            if isinstance(fobj, list):
                items = fobj
            else:
                items = [fobj]
            for it in items:
                if isinstance(it, dict):
                    p = it.get("path") or it.get("name") or it.get("orig_name")
                else:
                    p = getattr(it, "path", None) or getattr(it, "name", None)
                if p is None:
                    p = str(it)
                p = str(p)
                if p and not p.startswith("gr.update") and not p.startswith("update("):
                    paths.append(p)
            return paths

        def _load_scene_paths(image_paths: list, view_mode: str):
            print(f"[attn-ui] loading scene with {len(image_paths)} image(s)")
            imgs = load_images([str(x) for x in image_paths])
            imgs_b = _as_batched(imgs)
            _, S, _, H, W = imgs_b.shape
            print(f"[attn-ui] scene loaded: S={int(S)} frames")
            frames_uint8 = (imgs_b[0].detach().cpu().permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)

            # update backend-only holder
            scene_holder["images_b"] = imgs_b
            scene_holder["frames_uint8"] = frames_uint8
            scene_holder["S"] = int(S)
            scene_holder["H"] = int(H)
            scene_holder["W"] = int(W)

            # reset attention holder
            attn_holder["attn_all"] = None
            attn_holder["preds"] = None
            attn_holder["grid_h"] = None
            attn_holder["grid_w"] = None
            attn_holder["patch_start_idx"] = None
            attn_holder["P"] = None
            attn_holder["S"] = int(S)
            attn_holder["html_cache"] = {}

            patch_size = model.aggregator.patch_size
            gh, gw = _compute_patch_grid(H, W, patch_size)
            qf = int(np.random.randint(0, max(1, int(S))))
            preview = _render_patch_grid_preview(frames_uint8[qf], patch_size, None, gw, gh)

            # outputs: frame_state, query_frame_state, patch_preview, query_frame_ui, cache, grid_html, frame_label, view_sel
            return (
                0,
                qf,
                gr.update(value=preview),
                gr.update(maximum=max(0, int(S) - 1), value=qf),
                None,
                "",
                _render_label(str(view_mode), 0, 0),
                0,
            )

        def _on_scene_pick_and_load(f, view_mode):
            # If gradio provides a directory upload, it may actually provide a list of uploaded files.
            file_paths = _extract_file_paths(f)
            if len(file_paths) > 0:
                # Always prefer loading exactly the uploaded file list; it is the most reliable
                # across Gradio versions (directory uploads often get materialized as multiple files).
                file_paths = sorted(file_paths)
                try:
                    common = os.path.commonpath(file_paths)
                except Exception:
                    common = str(Path(file_paths[0]).parent)
                common_p = Path(common)
                if common_p.suffix:
                    common_p = common_p.parent
                common_dir = str(common_p)

                # If only a single file path is provided, try to expand to its parent dir.
                if len(file_paths) == 1:
                    try_dir = Path(file_paths[0]).parent
                    if try_dir.is_dir():
                        img_exts = {".jpg", ".jpeg", ".png", ".webp"}
                        all_imgs = sorted([str(x) for x in try_dir.iterdir() if x.suffix.lower() in img_exts])
                        if len(all_imgs) > 1:
                            file_paths = all_imgs
                    print(f"[attn-ui] directory upload yielded 1 file; using {len(file_paths)} file(s) from parent dir")

                out = _load_scene_paths(file_paths, str(view_mode))
                return (common_dir,) + out

            path = _on_pick_folder(f)
            if path is None or (not isinstance(path, str)) or len(path) == 0:
                return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

            out = _load_scene_dir(path, str(view_mode))
            # prepend the textbox update
            return (path,) + out

        def _on_pick_folder(f):
            if f is None:
                return None
            # gr.File(directory) can return dict/list or a small object depending on gradio version
            if isinstance(f, list) and len(f) > 0:
                f = f[0]
            if isinstance(f, dict):
                path = f.get("path") or f.get("name") or f.get("orig_name")
            else:
                path = getattr(f, "path", None) or getattr(f, "name", None)
            if path is None:
                path = str(f)
            # avoid huge repr strings from objects
            path = str(path)
            if path.startswith("gr.update") or path.startswith("update("):
                return None
            p = Path(path)
            # Gradio's "directory" upload sometimes yields a representative file path; use its parent.
            if p.exists() and p.is_file():
                p = p.parent
            # If the path doesn't exist locally (e.g. gradio temp handling), still prefer the parent dir.
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                p = p.parent
            return str(p)

        def _on_query_frame_change(qf):
            frames_uint8 = scene_holder.get("frames_uint8")
            if frames_uint8 is None:
                return gr.update(), int(qf)
            qf = int(qf)
            S = int(scene_holder.get("S", len(frames_uint8)))
            qf = max(0, min(max(0, S - 1), qf))
            patch_size = model.aggregator.patch_size
            gh, gw = _compute_patch_grid(int(scene_holder.get("H")), int(scene_holder.get("W")), patch_size)
            preview = _render_patch_grid_preview(frames_uint8[qf], patch_size, None, gw, gh)
            return gr.update(value=preview), qf

        def _on_thumb_size_change(sz, cache, view_mode, view_sel, frame_idx):
            if cache is None or not isinstance(cache, dict):
                return cache, gr.update()
            cache = dict(cache)
            cache["thumb_size"] = int(sz)
            if isinstance(attn_holder.get("html_cache"), dict):
                attn_holder["html_cache"] = {}
            label, html = on_view_change(str(view_mode), int(view_sel), int(frame_idx), cache)
            return cache, html

        def _export_grid_png(view_mode, view_sel, frame_idx, cache):
            attn_all = attn_holder.get("attn_all")
            frames_uint8 = scene_holder.get("frames_uint8")
            if attn_all is None or frames_uint8 is None:
                return None
            grid_h = attn_holder.get("grid_h")
            grid_w = attn_holder.get("grid_w")
            patch_start_idx = attn_holder.get("patch_start_idx")
            P = attn_holder.get("P")
            S = int(scene_holder.get("S", len(frames_uint8)))
            if grid_h is None or grid_w is None or patch_start_idx is None or P is None:
                return None
            thumb_size = 96
            if isinstance(cache, dict) and cache.get("thumb_size") is not None:
                thumb_size = int(cache["thumb_size"])
            img = _render_attention_grid_image(
                attn_all,
                frames_uint8,
                S,
                str(view_mode),
                int(view_sel),
                int(frame_idx),
                int(grid_h),
                int(grid_w),
                int(patch_start_idx),
                int(P),
                int(num_heads),
                int(num_layers),
                cell=int(thumb_size),
                gap=2,
            )
            out_path = output_dir / f"attn_grid_{int(time.time()*1000)}.png"
            img.save(str(out_path))
            return str(out_path)

        def _on_view_mode_change_ui(m):
            return str(m), 0

        view_mode_ui.change(_on_view_mode_change_ui, inputs=[view_mode_ui], outputs=[view_mode, view_sel])
        scene_pick_ui.change(
            _on_scene_pick_and_load,
            inputs=[scene_pick_ui, view_mode],
            outputs=[scene_dir_ui, frame_state, query_frame_state, patch_preview, query_frame_ui, cache, grid_html, frame_label, view_sel],
        )

        load_scene_btn.click(
            _load_scene_dir,
            inputs=[scene_dir_ui, view_mode],
            outputs=[frame_state, query_frame_state, patch_preview, query_frame_ui, cache, grid_html, frame_label, view_sel],
        )
        query_frame_ui.change(_on_query_frame_change, inputs=[query_frame_ui], outputs=[patch_preview, query_frame_state])

        view_mode.change(on_view_change, inputs=[view_mode, view_sel, frame_state, cache], outputs=[frame_label, grid_html])
        view_sel.change(on_view_change, inputs=[view_mode, view_sel, frame_state, cache], outputs=[frame_label, grid_html])

        patch_preview.select(on_select, inputs=[frame_state, query_frame_state, view_mode, view_sel, thumb_size_ui, cache], outputs=[patch_preview, frame_label, grid_html, cache])

        thumb_size_ui.change(_on_thumb_size_change, inputs=[thumb_size_ui, cache, view_mode, view_sel, frame_state], outputs=[cache, grid_html])

        download_grid_btn.click(
            _export_grid_png,
            inputs=[view_mode, view_sel, frame_state, cache],
            outputs=[download_grid_file],
        )

        prev_btn.click(on_prev, inputs=[view_mode, view_sel, frame_state, cache], outputs=[view_mode, view_sel, frame_state, frame_label, grid_html])
        next_btn.click(on_next, inputs=[view_mode, view_sel, frame_state, cache], outputs=[view_mode, view_sel, frame_state, frame_label, grid_html])
        recon_btn.click(_run_reconstruction, inputs=[conf_thres, max_points, cache], outputs=[recon_view, recon_status])

    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.launch(server_name=server_name, server_port=server_port, share=True)


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
