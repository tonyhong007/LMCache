# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import List, Optional

# Third Party
import torch


@dataclass
class LMCBlendCommonMetadata:
    """
    CommonMetadata (fixed hyperparams) for blending operations in LMCache.
    """

    check_layers: List[int]
    recomp_ratios: Optional[List[float]] = None
    thresholds: Optional[List[float]] = None


@dataclass
class LMCBlendMetadata:
    """
    Metadata (determined during runtime) for blending operations in LMCache.
    """

    imp_indices: Optional[torch.Tensor] = None
    attn_mask: Optional[torch.Tensor] = None
    positions: Optional[torch.Tensor] = None
    # Optional per-step override for recompute ratio used by incremental blending.
    step_recompute_ratio: Optional[float] = None
    # Whether to capture a full token ranking during this blend run.
    capture_full_ranking: bool = False
    # Number of suffix tokens (e.g., query) that should always be recomputed
    # and excluded from diff_k selection. 0 means no suffix handling.
    suffix_len: int = 0
    include_suffix: bool = False
    # Pass 2 only: pre-selected indices (1D long tensor) to use directly
    # instead of scoring. process_qkv slices + writes back at every layer
    # without running the magnet scorer.
    magnet_preselected_indices: Optional[torch.Tensor] = None

    def clean(self):
        self.imp_indices = None
        self.attn_mask = None
        self.positions = None
        self.step_recompute_ratio = None
        self.capture_full_ranking = False
        self.suffix_len = 0
        self.include_suffix = False
        self.magnet_preselected_indices = None
