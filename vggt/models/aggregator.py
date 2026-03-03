# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # hardcoded to False

        self._capture_global_attention_enabled = False
        self._capture_global_attention_block_idx = None

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        print("\n[Aggregator.forward] BEGIN")
        print(f"  Input images shape: {images.shape}")
        
        B, S, C_in, H, W = images.shape
        print(f"  Batch={B}, Sequence={S}, Channels={C_in}, Height={H}, Width={W}")

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        print(f"\n  [Step 1] Normalizing images (ResNet mean/std)")
        images = (images - self._resnet_mean) / self._resnet_std
        print(f"    Normalized range: [{images.min():.4f}, {images.max():.4f}]")

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        print(f"\n  [Step 2] Patch embedding")
        print(f"    Reshaped to: {images.shape}")
        patch_tokens = self.patch_embed(images)
        print(f"    Patch tokens shape: {patch_tokens.shape if isinstance(patch_tokens, torch.Tensor) else type(patch_tokens)}")

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
            print(f"    Extracted 'x_norm_patchtokens': {patch_tokens.shape}")

        _, P, C = patch_tokens.shape
        print(f"    P (num patches)={P}, C (embed_dim)={C}")

        # Expand camera and register tokens to match batch size and sequence length
        print(f"\n  [Step 3] Special tokens (camera & register)")
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        print(f"    Camera token shape: {camera_token.shape}")
        print(f"    Register token shape: {register_token.shape}")

        # Concatenate special tokens with patch tokens
        print(f"\n  [Step 4] Concatenate special + patch tokens")
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        print(f"    Combined tokens shape: {tokens.shape}")

        pos = None
        if self.rope is not None:
            print(f"\n  [Step 5] Rotary position embeddings (RoPE)")
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)
            print(f"    Pos shape: {pos.shape}")

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            print(f"    Masking positions for special tokens (first {self.patch_start_idx} tokens)")
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
            print(f"    Final pos shape: {pos.shape}")

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        print(f"\n  [Step 6] Alternating Attention ({self.aa_block_num} blocks)")
        print(f"    Attention order: {self.aa_order}, block_size: {self.aa_block_size}")

        for block_num in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    print(f"    Block {block_num}, Frame attention (block {frame_idx})...")
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    print(f"    Block {block_num}, Global attention (block {global_idx})...")
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)
                if i == 0:
                    print(f"      Output intermediate shape: {concat_inter.shape}, range [{concat_inter.min():.4f}, {concat_inter.max():.4f}]")

        del concat_inter
        del frame_intermediates
        del global_intermediates
        
        print(f"\n  [Aggregator.forward] COMPLETE")
        print(f"    Total outputs: {len(output_list)}, patch_start_idx: {self.patch_start_idx}")
        print("="*80 + "\n")
        return output_list, self.patch_start_idx

    def enable_global_attention_capture(
        self,
        block_idx: Optional[int] = None,
        query_indices: Optional[Union[List[int], torch.Tensor]] = None,
    ) -> None:
        if block_idx is None:
            block_idx = len(self.global_blocks) - 1

        if block_idx < 0 or block_idx >= len(self.global_blocks):
            raise ValueError(f"block_idx out of range: {block_idx}")

        blk = self.global_blocks[block_idx]
        if not hasattr(blk, "attn"):
            raise RuntimeError("Target block has no attention module")

        attn_mod = blk.attn
        if not hasattr(attn_mod, "capture_attention"):
            raise RuntimeError("Attention module does not support capture_attention")

        attn_mod.capture_attention = True
        attn_mod.attention_query_indices = query_indices
        attn_mod.captured_attention = None

        self._capture_global_attention_enabled = True
        self._capture_global_attention_block_idx = block_idx

    def disable_global_attention_capture(self) -> None:
        if not self._capture_global_attention_enabled or self._capture_global_attention_block_idx is None:
            return

        blk = self.global_blocks[self._capture_global_attention_block_idx]
        attn_mod = blk.attn
        if hasattr(attn_mod, "capture_attention"):
            attn_mod.capture_attention = False
            attn_mod.attention_query_indices = None
            attn_mod.captured_attention = None

        self._capture_global_attention_enabled = False
        self._capture_global_attention_block_idx = None

    def get_captured_global_attention(self) -> Optional[torch.Tensor]:
        if not self._capture_global_attention_enabled or self._capture_global_attention_block_idx is None:
            return None
        blk = self.global_blocks[self._capture_global_attention_block_idx]
        attn_mod = blk.attn
        return getattr(attn_mod, "captured_attention", None)

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for block_iter in range(self.aa_block_size):
            print(f"        [Frame Block {frame_idx}, iter {block_iter}] input shape: {tokens.shape}, range [{tokens.min():.4f}, {tokens.max():.4f}]")
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            print(f"        [Frame Block {frame_idx}, iter {block_iter}] output shape: {tokens.shape}, range [{tokens.min():.4f}, {tokens.max():.4f}]")
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for block_iter in range(self.aa_block_size):
            print(f"        [Global Block {global_idx}, iter {block_iter}] input shape: {tokens.shape}, range [{tokens.min():.4f}, {tokens.max():.4f}]")
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            print(f"        [Global Block {global_idx}, iter {block_iter}] output shape: {tokens.shape}, range [{tokens.min():.4f}, {tokens.max():.4f}]")
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined
