# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True, enable_track=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log", conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp", conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """        
        print("\n" + "="*80)
        print("[VGGT.forward] INPUT IMAGES")
        print(f"  Shape: {images.shape}, dtype: {images.dtype}, device: {images.device}")
        print(f"  Range: [{images.min():.4f}, {images.max():.4f}]")
        
        # If without batch dimension, add it
        if len(images.shape) == 4:
            print("  → Adding batch dimension")
            images = images.unsqueeze(0)
            print(f"  New shape: {images.shape}")
            
        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)
            print(f"[VGGT.forward] Query points shape after unsqueeze: {query_points.shape}")

        print("\n[VGGT.forward] Calling Aggregator...")
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        print(f"[VGGT.forward] Aggregator output: {len(aggregated_tokens_list)} token lists, patch_start_idx={patch_start_idx}")

        print("\n[VGGT.forward] Calling Aggregator...")
        aggregated_tokens_list, patch_start_idx = self.aggregator(images)
        print(f"[VGGT.forward] Aggregator output: {len(aggregated_tokens_list)} token lists, patch_start_idx={patch_start_idx}")

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.camera_head is not None:
                print("\n[VGGT.forward] → CameraHead")
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                print(f"  Output: List of {len(pose_enc_list)} tensors, shape {pose_enc_list[-1].shape}, range [{pose_enc_list[-1].min():.4f}, {pose_enc_list[-1].max():.4f}]")
                predictions["pose_enc"] = pose_enc_list[-1]  # pose encoding of the last iteration
                predictions["pose_enc_list"] = pose_enc_list
                
            if self.depth_head is not None:
                print("\n[VGGT.forward] → DepthHead")
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                print(f"  Depth shape: {depth.shape}, range [{depth.min():.4f}, {depth.max():.4f}]")
                print(f"  Depth_conf shape: {depth_conf.shape}, range [{depth_conf.min():.4f}, {depth_conf.max():.4f}]")
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                print("\n[VGGT.forward] → PointHead (3D points)")
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                print(f"  Points3D shape: {pts3d.shape}, range [{pts3d.min():.4f}, {pts3d.max():.4f}]")
                print(f"  Points3D_conf shape: {pts3d_conf.shape}, range [{pts3d_conf.min():.4f}, {pts3d_conf.max():.4f}]")
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            print("\n[VGGT.forward] → TrackHead")
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            print(f"  Track shape: {track_list[-1].shape}, range [{track_list[-1].min():.4f}, {track_list[-1].max():.4f}]")
            print(f"  Visibility shape: {vis.shape}, range [{vis.min():.4f}, {vis.max():.4f}]")
            print(f"  Confidence shape: {conf.shape}, range [{conf.min():.4f}, {conf.max():.4f}]")
            predictions["track"] = track_list[-1]  # track of the last iteration
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images  # store the images for visualization during inference

        print("\n[VGGT.forward] COMPLETE - All predictions ready")
        print("="*80 + "\n")
        return predictions

