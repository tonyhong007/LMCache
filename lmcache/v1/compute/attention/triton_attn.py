# SPDX-License-Identifier: Apache-2.0
"""Triton attention backend for LMCache.

Uses PyTorch's scaled_dot_product_attention for contiguous Q, K, V
operations (blend/recompute mini-forward). This is correct because
Triton's own attention kernel is for paged KV cache only, while
LMCache's forward_contiguous receives already-extracted contiguous
tensors.

PyTorch SDPA dispatches to Flash Attention 2 on CUDA, so performance
is equivalent to the Flash Attention backend for small token counts.
"""
from typing import TYPE_CHECKING

import logging
import torch
import torch.nn.functional as F
from vllm.attention.layer import Attention

from lmcache.v1.compute.attention.abstract import AttentionInterface
from lmcache.v1.compute.attention.metadata import LMCFlashAttnMetadata

if TYPE_CHECKING:
    from lmcache.v1.compute.attention.metadata import LMCAttnMetadata

logger = logging.getLogger(__name__)


class LMCTritonAttnBackend(AttentionInterface):
    """Triton-compatible attention backend for LMCache.

    Uses PyTorch SDPA for contiguous attention, which dispatches to
    the best available kernel on the device.
    """

    def __init__(self, vllm_attn: Attention):
        self.vllm_attn = vllm_attn
        self.scale = vllm_attn.impl.scale
        self.num_heads = vllm_attn.num_heads
        self.num_kv_heads = vllm_attn.num_kv_heads
        self.head_size = vllm_attn.head_size

        idx = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{idx}")

    def forward_contiguous(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: "LMCAttnMetadata",
        **kwargs,
    ) -> torch.Tensor:
        # Input shapes: [M, num_heads, head_size]
        # SDPA expects: [batch, num_heads, seq_len, head_size]
        q = query.transpose(0, 1).unsqueeze(0)
        k = key.transpose(0, 1).unsqueeze(0)
        v = value.transpose(0, 1).unsqueeze(0)

        # GQA: expand KV heads to match Q heads.
        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            scale=self.scale,
        )
        # [1, H, M, D] -> [M, H, D]
        out = out.squeeze(0).transpose(0, 1).contiguous()
        output[:] = out
        return output

    def init_attn_metadata(
        self,
        input_ids: torch.Tensor,
        num_queries: int = None,
        **kwargs,
    ) -> "LMCAttnMetadata":
        seq_len = input_ids.shape[0]
        device = input_ids.device
        q_len = num_queries if num_queries is not None else seq_len
        return LMCFlashAttnMetadata(
            query_start_loc=torch.tensor(
                [0, q_len], dtype=torch.int32, device=device,
            ),
            seq_lens=torch.tensor([seq_len], device=device),
            cu_seqlens_k=torch.tensor(
                [0, seq_len], dtype=torch.int32, device=device,
            ),
            max_query_len=q_len,
            max_seq_len=seq_len,
        )
