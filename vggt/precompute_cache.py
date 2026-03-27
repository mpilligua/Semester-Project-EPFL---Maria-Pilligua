"""
Precompute 3D reconstructions for all available scenes and persist them to
DEMO_CACHE_DIR so that demo_gradio_only_comparison.py loads them instantly.

Usage:
  python3 precompute_cache.py [--force]

  --force   Recompute and overwrite even if a scene is already cached.
"""

import os, sys, glob, gc, time, json, shutil, argparse
from datetime import datetime
from pathlib import Path

os.environ['DISPLAY'] = ''
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image as PILImage

_PROJECT_ROOT = "/scratch/cvlab/home/pilligua/Semester-Project-EPFL---Maria-Pilligua"
sys.path.insert(0, _PROJECT_ROOT)                        # exposes 'vggt' as a namespace package
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "vggt"))  # exposes visual_util, etc.

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# ── Config ────────────────────────────────────────────────────────────────────
SCENE_DATA_BASE = "/scratch/cvlab/home/pilligua/mVGGT/notebooks"
DEMO_CACHE_DIR  = "/scratch/cvlab/home/pilligua/vggt_demo_cache"
TOKEN_PATCH     = 14
device          = "cuda" if torch.cuda.is_available() else "cpu"

# ── Scene discovery (same logic as the demo) ─────────────────────────────────

def _is_scene_dir(d: Path) -> bool:
    return d.is_dir() and (d / "images.npy").exists() and (d / "matching_gt.npy").exists()


def discover_scenes():
    scenes = {}
    base = Path(SCENE_DATA_BASE)
    examples_base = base / "dataset_examples"
    if examples_base.exists():
        for dataset_dir in sorted(examples_base.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for scene_dir in sorted(dataset_dir.iterdir()):
                if _is_scene_dir(scene_dir):
                    name = f"{dataset_dir.name}/{scene_dir.name}"
                    scenes[name] = str(scene_dir)
    for d in sorted(base.rglob("matching_data*")):
        if not _is_scene_dir(d):
            continue
        name = d.name
        if name.startswith("matching_data_"):
            name = name[len("matching_data_"):]
        if name == "":
            name = "original"
        legacy_key = f"legacy/{name}"
        if legacy_key not in scenes:
            scenes[legacy_key] = str(d)
    return scenes

# ── Disk-cache helpers (must match demo exactly) ──────────────────────────────

def _cache_scene_dir(scene_name: str) -> str:
    safe = scene_name.replace("/", "__").replace(" ", "_")
    return os.path.join(DEMO_CACHE_DIR, safe)


def _is_cached(scene_name: str) -> bool:
    return os.path.exists(os.path.join(_cache_scene_dir(scene_name), "full_attention.glb"))


def _save_reconstruction_cache(scene_name, timing, full_qk, sparse_qk, sparse_mask,
                                model_info, glb_full_path, glb_sparse_path):
    cache_dir = _cache_scene_dir(scene_name)
    os.makedirs(cache_dir, exist_ok=True)

    shutil.copy2(glb_full_path, os.path.join(cache_dir, "full_attention.glb"))
    if glb_sparse_path and os.path.exists(glb_sparse_path):
        shutil.copy2(glb_sparse_path, os.path.join(cache_dir, "sparse_attention.glb"))

    torch.save(full_qk, os.path.join(cache_dir, "full_qk.pt"))
    if sparse_qk is not None:
        torch.save(sparse_qk, os.path.join(cache_dir, "sparse_qk.pt"))
    if sparse_mask is not None:
        torch.save(sparse_mask, os.path.join(cache_dir, "sparse_mask.pt"))

    np.save(os.path.join(cache_dir, "frames_uint8.npy"), model_info["frames_uint8"])
    meta = {k: int(v) for k, v in model_info.items() if k != "frames_uint8"}
    with open(os.path.join(cache_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(cache_dir, "timing.json"), "w") as f:
        json.dump(timing, f)

    print(f"  [Cache] Saved → {cache_dir}")

# ── Model helpers ─────────────────────────────────────────────────────────────

def save_scene_images_to_dir(images_np):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"/tmp/precompute_input_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")
    os.makedirs(target_dir_images, exist_ok=True)
    for i in range(images_np.shape[0]):
        img = images_np[i].transpose(1, 2, 0)
        img = (img * 255).clip(0, 255).astype(np.uint8)
        PILImage.fromarray(img).save(os.path.join(target_dir_images, f"view_{i:02d}.png"))
    return target_dir


def run_model_comparison(model, matching_gt_tensor, target_dir):
    image_names = sorted(glob.glob(os.path.join(target_dir, "images", "*")))
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

    # Full attention
    print("  [1/2] Full attention ...")
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
    full_qk = [
        (q.detach().cpu().half() if q is not None else None,
         k.detach().cpu().half() if k is not None else None)
        for (q, k) in full_qk
    ]
    model.aggregator.disable_global_qk_capture()
    full_preds = postprocess(full_preds)
    print(f"     {timing['full_time']:.1f}s, {timing['full_memory']:.2f}GB")

    # Sparse attention
    sparse_preds = sparse_qk = sparse_mask = None
    if matching_gt_tensor is not None:
        print("  [2/2] Sparse attention ...")
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
        sparse_qk = [
            (q.detach().cpu().half() if q is not None else None,
             k.detach().cpu().half() if k is not None else None)
            for (q, k) in sparse_qk
        ]
        model.aggregator.disable_global_qk_capture()

        # Rebuild sparse mask
        B_img = images.shape[0] if images.dim() == 5 else 1
        S_img = images.shape[1] if images.dim() == 5 else images.shape[0]
        H_img, W_img = images.shape[-2], images.shape[-1]
        patch_size = model.aggregator.patch_size
        patch_h, patch_w = H_img // patch_size, W_img // patch_size
        P_patches = patch_h * patch_w
        psi = model.aggregator.patch_start_idx
        P_total = psi + P_patches

        from vggt.utils.multi_view_matcher import MultiViewMatcher
        matcher = model.aggregator.multi_view_matcher
        matcher.num_patches_h = patch_h
        matcher.num_patches_w = patch_w
        matcher.num_tokens_per_view = P_patches
        gt_subset = matching_gt_tensor[:, :S_img, :, :, :S_img, :, :]
        raw_mask, _ = matcher(matching_gt=gt_subset)
        raw_mask = raw_mask.to(dtype=torch.float32)

        padded = torch.zeros(B_img, S_img * P_total, S_img * P_total)
        for s in range(S_img):
            s0 = s * P_total; se = s0 + psi; sp = se; send = s0 + P_total
            for s2 in range(S_img):
                s20 = s2 * P_total; s2e = s20 + psi; s2p = s2e; s2end = s20 + P_total
                if s == s2:
                    padded[:, s0:send, s20:s2end] = 1.0
                else:
                    padded[:, s0:se, s20:s2e] = 1.0
                    padded[:, sp:send, s2p:s2end] = raw_mask[
                        :, s * P_patches:(s+1) * P_patches, s2 * P_patches:(s2+1) * P_patches
                    ]
        sparse_mask = padded.cpu().half()

        model.disable_sparse_attention()
        sparse_preds = postprocess(sparse_preds)
        timing['speedup'] = timing['full_time'] / timing['sparse_time'] if timing['sparse_time'] > 0 else 0
        timing['memory_reduction'] = timing['full_memory'] / timing['sparse_memory'] if timing['sparse_memory'] > 0 else 0
        print(f"     {timing['sparse_time']:.1f}s, {timing['sparse_memory']:.2f}GB | "
              f"speedup {timing['speedup']:.2f}x")
    else:
        print("  [2/2] Skipping sparse attention (no matching GT)")

    # model_info
    images_5d = images.unsqueeze(0) if images.dim() == 4 else images
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

    torch.cuda.empty_cache()
    return full_preds, sparse_preds, timing, full_qk, sparse_qk, sparse_mask, model_info


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Recompute even if already cached")
    args = parser.parse_args()

    scenes = discover_scenes()
    print(f"\nFound {len(scenes)} scenes.")

    to_process = []
    for name in sorted(scenes):
        if not args.force and _is_cached(name):
            print(f"  [SKIP]  {name}  (already cached)")
        else:
            to_process.append(name)
            print(f"  [TODO]  {name}")

    if not to_process:
        print("\nAll scenes already cached. Use --force to recompute.")
        return

    print(f"\nLoading VGGT model ...")
    model = VGGT()
    cached_model_path = os.path.expanduser("~/.cache/torch/hub/checkpoints/model.pt")
    if os.path.exists(cached_model_path):
        model_state = torch.load(cached_model_path, map_location="cpu")
    else:
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model_state = torch.hub.load_state_dict_from_url(_URL)
    model.load_state_dict(model_state)
    model.eval()
    model = model.to(device)
    print(f"  Model on {device}\n")

    total = len(to_process)
    t_start_all = time.time()

    for idx, scene_name in enumerate(to_process, 1):
        scene_dir = scenes[scene_name]
        print(f"\n[{idx}/{total}] {scene_name}")
        t0 = time.time()

        try:
            # Load + resize images
            images = np.load(os.path.join(scene_dir, "images.npy"))[0]      # [V,3,H,W]
            img_tensor = torch.from_numpy(images).float()
            img_resized = F.interpolate(img_tensor, size=(518, 518), mode="bilinear", align_corners=False)
            images = img_resized.cpu().numpy()

            # Load matching GT
            gt_path = os.path.join(scene_dir, "matching_gt.npy")
            matching_gt_tensor = torch.from_numpy(np.load(gt_path)).bool().to(device)

            # Save images to temp dir for model loader
            target_dir = save_scene_images_to_dir(images)

            # Run both model passes
            full_preds, sparse_preds, timing, full_qk, sparse_qk, sparse_mask, model_info = \
                run_model_comparison(model, matching_gt_tensor, target_dir)

            # Generate GLBs
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

            # Persist to cache
            _save_reconstruction_cache(
                scene_name, timing, full_qk, sparse_qk, sparse_mask,
                model_info, glb_full, glb_sparse,
            )

            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s")

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()

        finally:
            gc.collect()
            torch.cuda.empty_cache()

    total_elapsed = time.time() - t_start_all
    print(f"\nFinished {total} scenes in {total_elapsed/60:.1f} min.")
    print(f"Cache directory: {DEMO_CACHE_DIR}")


if __name__ == "__main__":
    main()
