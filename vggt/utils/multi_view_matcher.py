"""
Multi-view token correspondence matcher for sparse attention in VGGT.

This module computes which tokens from different views correspond to each other
based on camera geometry (depth maps, intrinsics, extrinsics), and generates
sparse attention masks for efficient cross-view attention.

The key insight: instead of O(N²) full attention, each token only attends to
~V tokens (one per view), reducing complexity to O(N·V).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np


class MultiViewMatcher(nn.Module):
    """
    Computes token-level correspondences across multiple views and generates
    sparse attention masks for multi-view attention.
    
    Attributes:
        patch_size (int): Size of patches (e.g., 14 for ViT)
        img_size (int): Input image size
        num_patches_per_side (int): Number of patches per side (img_size // patch_size)
    """
    
    def __init__(self, patch_size: int = 14, img_size: int = 518):
        """
        Initialize the matcher.
        
        Args:
            patch_size: Size of each patch (default: 14 for ViT)
            img_size: Input image size (default: 518)
        """
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches_per_side = img_size // patch_size
        # Support non-square images: separate H and W patch counts
        self.num_patches_h = self.num_patches_per_side
        self.num_patches_w = self.num_patches_per_side
        self.num_tokens_per_view = self.num_patches_h * self.num_patches_w
        
        # Cache for computed masks
        self._cached_mask = None
        self._cached_key = None
    
    def pixel_to_token_coords(self, pixel_coords: torch.Tensor) -> torch.Tensor:
        """
        Convert pixel coordinates to token coordinates.
        
        Args:
            pixel_coords: Pixel coordinates [N, 2] as (x, y)
            
        Returns:
            Token coordinates [N, 2] (integer indices)
        """
        token_coords = (pixel_coords / self.patch_size).long()
        # Clamp to valid token range
        token_coords = torch.clamp(
            token_coords, 
            min=0, 
            max=self.num_patches_per_side - 1
        )
        return token_coords
    
    def token_coords_to_linear_idx(self, token_coords: torch.Tensor) -> torch.Tensor:
        """
        Convert 2D token coordinates to linear token indices.
        
        Args:
            token_coords: Token coordinates [N, 2] as (row, col)
            
        Returns:
            Linear indices [N] in range [0, num_tokens_per_view)
        """
        return token_coords[..., 0] * self.num_patches_per_side + token_coords[..., 1]
    
    def forward(
        self,
        depths: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
        extrinsics: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        matching_gt: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute sparse attention masks from camera geometry or matching ground truth.
        
        Args:
            depths: Depth maps [B, S, H, W]
            intrinsics: Camera intrinsics [B, S, 3, 3]
            extrinsics: Camera extrinsics [B, S, 4, 4]
            masks: Valid region masks [B, S, H, W] (optional)
            matching_gt: Pre-computed matching ground truth tensor (optional)
                If provided, uses this directly instead of computing from camera params.
                Format: sparse tensor with nonzero entries at [b, view1, y1, x1, view2, y2, x2]
        
        Returns:
            attn_mask: Sparse attention mask [B, S*T, S*T] where T is tokens per view
                        Values: 1.0 where attention is allowed, 0.0 where masked
            info: Dictionary with debug information
                - num_valid_correspondences: Number of valid token pairs
                - sparsity_ratio: Ratio of non-masked entries
        """
        info = {
            'num_valid_correspondences': 0,
            'sparsity_ratio': 0.0,
        }
        
        # If matching_gt is provided, use it directly
        if matching_gt is not None:
            # Get batch and sequence size from matching_gt shape
            B = matching_gt.shape[0]
            S = matching_gt.shape[1]
            device = matching_gt.device
            dtype = torch.float32
            
            attn_mask = self._mask_from_matching_gt(
                matching_gt, B, S, device, dtype, info
            )
            return attn_mask, info
        
        # Otherwise compute from camera parameters
        if depths is None:
            raise ValueError("Either matching_gt or depths must be provided")
        
        B, S, H, W = depths.shape
        device = depths.device
        dtype = depths.dtype
        
        # Compute from camera geometry
        attn_mask = self._mask_from_camera_geometry(
            depths, intrinsics, extrinsics, masks, B, S, device, dtype, info
        )
        
        return attn_mask, info
    
    def _mask_from_matching_gt(
        self,
        matching_gt: torch.Tensor,
        B: int,
        S: int,
        device: torch.device,
        dtype: torch.dtype,
        info: Dict,
    ) -> torch.Tensor:
        """
        Create attention mask from pre-computed matching ground truth.
        
        Args:
            matching_gt: Sparse matching tensor [B, S, H, W, S, H, W] or similar
            B, S: Batch size and number of views
            device, dtype: Target device and dtype
            info: Info dictionary to update
            
        Returns:
            Sparse attention mask [B, S*T, S*T]
        """
        total_tokens = S * self.num_tokens_per_view
        
        # Initialize: set all to 0 (masked out initially)
        attn_mask = torch.zeros(
            (B, total_tokens, total_tokens),
            device=device,
            dtype=dtype
        )
        
        # Process each batch
        for b in range(B):
            # Get all nonzero correspondences for this batch
            coords = torch.nonzero(matching_gt[b])
            
            if len(coords) == 0:
                # No correspondences, keep full attention as fallback
                attn_mask[b] = torch.ones_like(attn_mask[b])
                continue
            
            # coords shape: [N, 7] with values [view1, y1, x1, view2, y2, x2, ...]
            # (may have additional dimensions depending on matching_gt format)
            
            view1_idx = coords[:, 0].long()
            y1 = coords[:, 1].long()
            x1 = coords[:, 2].long()
            view2_idx = coords[:, 3].long()
            y2 = coords[:, 4].long()
            x2 = coords[:, 5].long()
            
            # matching_gt is already in token coordinates (not pixel coords)
            token_y1 = y1
            token_x1 = x1
            token_y2 = y2
            token_x2 = x2
            
            # Clamp to valid range (non-square: separate h and w bounds)
            token_y1 = torch.clamp(token_y1, 0, self.num_patches_h - 1)
            token_x1 = torch.clamp(token_x1, 0, self.num_patches_w - 1)
            token_y2 = torch.clamp(token_y2, 0, self.num_patches_h - 1)
            token_x2 = torch.clamp(token_x2, 0, self.num_patches_w - 1)
            
            # Convert to linear token indices within each view
            # Row-major: index = row * num_cols + col
            token_idx1 = token_y1 * self.num_patches_w + token_x1
            token_idx2 = token_y2 * self.num_patches_w + token_x2
            
            # Convert to global token indices (accounting for which view)
            global_idx1 = view1_idx * self.num_tokens_per_view + token_idx1
            global_idx2 = view2_idx * self.num_tokens_per_view + token_idx2
            
            # Vectorized mask assignment (no Python loop)
            valid = (
                (global_idx1 >= 0) & (global_idx1 < total_tokens) &
                (global_idx2 >= 0) & (global_idx2 < total_tokens)
            )
            g1 = global_idx1[valid]
            g2 = global_idx2[valid]
            attn_mask[b, g1, g2] = 1.0
            attn_mask[b, g2, g1] = 1.0  # Bidirectional
            
            info['num_valid_correspondences'] = int(valid.sum().item())
        
        # Compute sparsity ratio
        total_entries = B * total_tokens * total_tokens
        nonzero_entries = (attn_mask > 0).sum().item()
        info['sparsity_ratio'] = nonzero_entries / total_entries if total_entries > 0 else 0.0
        
        return attn_mask
    
    def _mask_from_camera_geometry(
        self,
        depths: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        masks: Optional[torch.Tensor],
        B: int,
        S: int,
        device: torch.device,
        dtype: torch.dtype,
        info: Dict,
    ) -> torch.Tensor:
        """
        Compute attention mask from camera geometry.
        
        For now, returns full attention as placeholder.
        TODO: Implement proper 3D-to-2D projection for each pixel
        """
        total_tokens = S * self.num_tokens_per_view
        
        # Placeholder: return full attention
        # TODO: Implement
        # 1. For each pixel in each view, project to 3D using depth
        # 2. Project 3D point to all other views using extrinsics/intrinsics
        # 3. Mark corresponding tokens as allowed
        
        attn_mask = torch.ones(
            (B, total_tokens, total_tokens),
            device=device,
            dtype=dtype
        )
        
        info['sparsity_ratio'] = 1.0  # Full attention
        
        return attn_mask
    
    def create_sparse_attention_mask(
        self,
        attn_mask: torch.Tensor,
        num_heads: int = 16,
    ) -> torch.Tensor:
        """
        Expand attention mask to head dimension for multi-head attention.
        
        Args:
            attn_mask: Mask [B, S*T, S*T]
            num_heads: Number of attention heads
            
        Returns:
            Mask suitable for use in attention: [B, num_heads, S*T, S*T]
        """
        B, N, _ = attn_mask.shape
        # Replicate mask for all heads
        mask = attn_mask.unsqueeze(1).expand(B, num_heads, N, N)
        return mask
    
    def apply_mask_to_attention(
        self,
        attn_logits: torch.Tensor,
        attn_mask: torch.Tensor,
        mask_value: float = float('-inf'),
    ) -> torch.Tensor:
        """
        Apply mask to attention logits before softmax.
        
        Args:
            attn_logits: Attention logits [B, H, N, N]
            attn_mask: Binary mask [B, H, N, N] or broadcastable shape
            mask_value: Value to fill masked positions (default: -inf)
            
        Returns:
            Masked attention logits
        """
        return attn_logits.masked_fill(attn_mask == 0, mask_value)
