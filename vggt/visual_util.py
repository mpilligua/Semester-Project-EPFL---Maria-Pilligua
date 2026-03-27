# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Visualization utilities for VGGT predictions.
Provides functions to convert predictions to GLB format for 3D visualization.
"""

import torch
import numpy as np
import trimesh
import os
from pathlib import Path


def predictions_to_glb(
    predictions,
    conf_thres=50.0,
    filter_by_frames="All",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    target_dir=None,
    prediction_mode="Depthmap and Camera Branch",
):
    """
    Convert VGGT predictions to a trimesh Scene for GLB export.
    
    Args:
        predictions: Dictionary containing model predictions (numpy arrays)
        conf_thres: Confidence threshold for filtering points (0-100)
        filter_by_frames: Frame index to filter by, or "All" for all frames
        mask_black_bg: Whether to mask black background pixels
        mask_white_bg: Whether to mask white background pixels
        show_cam: Whether to include camera positions in the scene
        mask_sky: Whether to mask sky pixels
        target_dir: Target directory for saving (used for sky/bg detection)
        prediction_mode: "Depthmap and Camera Branch" or "Pointmap Branch"
    
    Returns:
        trimesh.Scene object that can be exported to GLB/GLB format
    """
    
    # Normalize confidence threshold to [0, 1]
    conf_thres_norm = conf_thres / 100.0
    
    # Extract predictions
    if "world_points" in predictions:
        world_points = predictions["world_points"]  # [S, H, W, 3]
        world_points_conf = predictions.get("world_points_conf", None)  # [S, H, W]
    elif "depth" in predictions:
        # Reconstruct world points from depth
        world_points = predictions.get("world_points_from_depth", None)
        world_points_conf = predictions.get("depth_conf", None)
    else:
        raise ValueError("Predictions must contain 'world_points' or 'depth'")
    
    if world_points is None:
        raise ValueError("Could not find valid 3D points in predictions")
    
    images = predictions.get("images", None)
    if images is not None:
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
    
    # Ensure numpy arrays
    if isinstance(world_points, torch.Tensor):
        world_points = world_points.cpu().numpy()
    if isinstance(world_points_conf, torch.Tensor):
        world_points_conf = world_points_conf.cpu().numpy()
    
    # Remove batch dimension if present
    if world_points.ndim == 5:
        world_points = world_points[0]  # [S, H, W, 3]
    if world_points_conf is not None and world_points_conf.ndim == 4:
        world_points_conf = world_points_conf[0]  # [S, H, W]
    if images is not None and images.ndim == 5:
        images = images[0]  # [S, 3, H, W]
    
    S, H, W = world_points.shape[0], world_points.shape[1], world_points.shape[2]
    
    # Handle frame filtering
    frame_indices = range(S)
    if filter_by_frames != "All" and filter_by_frames is not None:
        try:
            if ":" in filter_by_frames:
                # Parse "0: image_name.png" format
                frame_idx = int(filter_by_frames.split(":")[0])
                frame_indices = [frame_idx]
        except (ValueError, IndexError):
            pass
    
    # Build point cloud
    points_list = []
    colors_list = []
    
    for frame_idx in frame_indices:
        if frame_idx >= S:
            continue
        
        pts_frame = world_points[frame_idx]  # [H, W, 3]
        
        # Get colors from images
        if images is not None and frame_idx < images.shape[0]:
            img_frame = images[frame_idx]  # [3, H, W]
            if img_frame.dtype != np.uint8:
                img_frame = np.clip(img_frame, 0, 1)
                if img_frame.max() <= 1.0:
                    img_frame = (img_frame * 255).astype(np.uint8)
            img_hw3 = np.transpose(img_frame, (1, 2, 0))  # [H, W, 3]
        else:
            img_hw3 = np.ones((H, W, 3), dtype=np.uint8) * 128
        
        # Build validity mask
        valid = np.isfinite(pts_frame).all(axis=-1)
        
        # Apply confidence threshold
        if world_points_conf is not None and frame_idx < world_points_conf.shape[0]:
            conf_frame = world_points_conf[frame_idx]  # [H, W]
            valid = valid & (conf_frame >= conf_thres_norm)
        
        # Apply background masking
        if mask_black_bg or mask_white_bg:
            if mask_black_bg:
                black_mask = (img_hw3.mean(axis=-1) < 0.1)
                valid = valid & ~black_mask
            if mask_white_bg:
                white_mask = (img_hw3.mean(axis=-1) > 0.9)
                valid = valid & ~white_mask
        
        # Flatten and collect
        pts_flat = pts_frame.reshape(-1, 3)
        cols_flat = img_hw3.reshape(-1, 3)
        valid_flat = valid.reshape(-1)
        
        # Filter by validity
        pts_valid = pts_flat[valid_flat]
        cols_valid = cols_flat[valid_flat]
        
        if pts_valid.shape[0] > 0:
            points_list.append(pts_valid)
            colors_list.append(cols_valid)
    
    if len(points_list) == 0:
        # Return empty scene if no valid points
        return trimesh.Scene()
    
    # Combine all points
    all_points = np.vstack(points_list).astype(np.float32)
    all_colors = np.vstack(colors_list).astype(np.uint8)
    
    # Clip colors to valid range
    all_colors = np.clip(all_colors, 0, 255).astype(np.uint8)
    
    # Convert Y coordinate for upright visualization
    all_points[:, 1] *= -1.0
    
    # Create point cloud and scene
    pc = trimesh.points.PointCloud(vertices=all_points, colors=all_colors)
    scene = trimesh.Scene([pc])
    
    # Add cameras if requested
    if show_cam and "extrinsic" in predictions and "intrinsic" in predictions:
        try:
            extrinsic = predictions["extrinsic"]  # [S, 4, 4]
            intrinsic = predictions["intrinsic"]  # [S, 3, 3]
            
            if isinstance(extrinsic, torch.Tensor):
                extrinsic = extrinsic.cpu().numpy()
            if isinstance(intrinsic, torch.Tensor):
                intrinsic = intrinsic.cpu().numpy()
            
            # Remove batch dimension if present
            if extrinsic.ndim == 4:
                extrinsic = extrinsic[0]
            if intrinsic.ndim == 3:
                intrinsic = intrinsic[0]
            
            # Add simple camera meshes (small pyramids at camera positions)
            for i, (extr, intr) in enumerate(zip(extrinsic, intrinsic)):
                if i not in frame_indices:
                    continue
                
                # Extract camera center
                cam_center = -extr[:3, :3].T @ extr[:3, 3]
                
                # Create a small pyramid to represent camera
                cam_size = 0.1 * np.linalg.norm(all_points.max(axis=0) - all_points.min(axis=0))
                camera_mesh = _create_camera_mesh(cam_center, extr[:3, :3], cam_size)
                if camera_mesh is not None:
                    scene.add_geometry(camera_mesh)
        except Exception as e:
            print(f"Warning: Could not add cameras to scene: {e}")
    
    return scene


def _create_camera_mesh(position, rotation, size=0.1):
    """
    Create a simple camera frustum mesh.
    
    Args:
        position: Camera center position [3]
        rotation: Camera rotation matrix [3, 3]
        size: Size of the camera mesh
    
    Returns:
        trimesh.Trimesh object representing the camera
    """
    try:
        # Create a simple pyramid for camera visualization
        vertices = np.array([
            [0, 0, 0],
            [1, 1, 2],
            [-1, 1, 2],
            [-1, -1, 2],
            [1, -1, 2],
        ], dtype=np.float32) * size
        
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [1, 2, 3],
            [1, 3, 4],
        ])
        
        # Transform to world coordinates
        vertices = (rotation @ vertices.T).T + position
        
        color = np.array([0, 255, 0])  # Green for cameras
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh.visual.face_colors = color
        
        return mesh
    except Exception as e:
        print(f"Warning: Could not create camera mesh: {e}")
        return None
