# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = True,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope

        self.capture_attention = False
        self.attention_query_indices = None
        self.captured_attention = None

        self.capture_qk = False
        self.captured_q = None
        self.captured_k = None

    def forward(self, x: Tensor, pos=None, attn_mask: Tensor = None) -> Tensor:
        """
        Forward pass with optional sparse attention masking.
        
        Args:
            x: Input tensor [B, N, C]
            pos: Positional embeddings (for RoPE)
            attn_mask: Optional sparse attention mask [B, N, N]
                      Values should be 1.0 where attention is allowed,
                      0.0 where it should be masked out.
        
        Returns:
            Output tensor [B, N, C]
        """
        B, N, C = x.shape
        
        # QKV projection and reshape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: [B, H, N, D]
        
        # Q, K normalization (if enabled)
        q, k = self.q_norm(q), self.k_norm(k)

        # Apply RoPE if available
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # Capture Q/K after RoPE for later on-the-fly attention recomputation
        if self.capture_qk:
            self.captured_q = q.detach().cpu().half()
            self.captured_k = k.detach().cpu().half()

        # Attention computation
        if self.fused_attn and attn_mask is None:
            # Use fused attention when no mask
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            # Use sparse attention computation when mask is provided
            if attn_mask is not None:
                # Sparse attention: only compute attention for valid token pairs
                x = self._sparse_attention(q, k, v, attn_mask)
            else:
                # Full attention without mask
                q = q * self.scale
                attn = q @ k.transpose(-2, -1)
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                if self.capture_attention:
                    if self.attention_query_indices is None:
                        self.captured_attention = attn.detach()
                    else:
                        qi = self.attention_query_indices
                        qi = torch.as_tensor(qi, device=attn.device, dtype=torch.long)
                        self.captured_attention = attn.index_select(dim=-2, index=qi).detach()
                x = attn @ v

        if self.capture_attention and self.attention_query_indices is not None:
            qi = torch.as_tensor(self.attention_query_indices, device=q.device, dtype=torch.long)
            q_sel = q.index_select(dim=-2, index=qi)
            attn_sel = (q_sel * self.scale) @ k.transpose(-2, -1)
            if attn_mask is not None:
                # Index the mask at query dimension: [B, N, N] -> [B, len(qi), N]
                if attn_mask.dim() == 3:
                    mask_sel = attn_mask[:, qi, :]  # [B, len(qi), N]
                    mask_sel = mask_sel.unsqueeze(1)  # [B, 1, len(qi), N]
                else:
                    mask_sel = attn_mask[:, :, qi, :]  # [B, H, len(qi), N]
                attn_sel = attn_sel.masked_fill(mask_sel == 0, float('-inf'))
            attn_sel = attn_sel.softmax(dim=-1)
            attn_sel = torch.nan_to_num(attn_sel, nan=0.0, posinf=0.0, neginf=0.0)
            self.captured_attention = attn_sel.detach()

        # Reshape and project back
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def _sparse_attention(self, q: Tensor, k: Tensor, v: Tensor, attn_mask: Tensor) -> Tensor:
        B, H, N, D = q.shape
        
        if attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1).expand(B, H, N, N)
        
        scale = self.scale
        q_scaled = q * scale  # [B, H, N, D]
        
        q_bh = q_scaled.reshape(B*H, N, D)
        k_bh = k.reshape(B*H, N, D)
        v_bh = v.reshape(B*H, N, D)
        mask_bh = attn_mask.reshape(B*H, N, N)
        
        attn_logits = q_bh @ k_bh.transpose(-2, -1)  # [B*H, N, N]
        
        attn_logits = attn_logits.masked_fill(mask_bh == 0, float('-inf'))
        
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        
        attn_weights = self.attn_drop(attn_weights)
        
        out_bh = attn_weights @ v_bh
        
        out = out_bh.reshape(B, H, N, D)
        
        return out
    
    
    
    
    def _apply_attention_mask(self, attn: Tensor, attn_mask: Tensor) -> Tensor:
        """
        Apply attention mask to logits.
        
        Args:
            attn: Attention logits [B, H, N, N]
            attn_mask: Binary mask [B, H, N, N] or [B, N, N]
                      1.0 = allow attention, 0.0 = mask out
        
        Returns:
            Masked attention logits
        """
        # Expand mask to match attn shape if needed
        if attn_mask.dim() == 3:
            # [B, N, N] -> [B, 1, N, N]
            attn_mask = attn_mask.unsqueeze(1)
        
        # Apply mask: set masked positions to -inf
        return attn.masked_fill(attn_mask == 0, float('-inf'))


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None, attn_mask: Tensor = None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, pos=pos, attn_mask=attn_mask)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
