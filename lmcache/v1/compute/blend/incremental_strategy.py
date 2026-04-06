# SPDX-License-Identifier: Apache-2.0
# Standard
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from lmcache.logging import init_logger

logger = init_logger(__name__)


def _normalize_ratio(value: float) -> float:
    """
    Accept either ratio-style values (0.0-1.0) or percent-style values
    (0.0-100.0) and normalize to [0.0, 1.0].
    """
    ratio = float(value)
    if ratio < 0.0:
        return 0.0
    if ratio > 1.0:
        ratio = ratio / 100.0
    return min(ratio, 1.0)


@dataclass
class IncrementalBlendState:
    decode_step: int = 0
    cumulative_ratio: float = 0.0
    # Frozen descending ranking of candidate token indices.
    frozen_importance_ranking: Optional[Any] = None
    # Consumed ratio cursor for disjoint slice selection on frozen ranking.
    frozen_ratio_cursor: float = 0.0
    # Tokens already recomputed before TTFT. Used to adjust the budget so
    # the total recompute (pre_ttft + incremental) reaches max_ratio.
    pre_ttft_recomputed_tokens: int = 0

    # ── Layer-wise incremental state ──
    # Saved hidden_states tensor at the layer-group boundary [M, hidden_dim].
    layerwise_saved_hidden: Optional[Any] = None
    # Saved residual tensor at the layer-group boundary [M, hidden_dim].
    layerwise_saved_residual: Optional[Any] = None
    # Global positions of the selected tokens (set once during pre-TTFT).
    layerwise_selected_indices: Optional[Any] = None
    # Next layer to recompute (exclusive upper bound of the last completed range).
    layerwise_current_layer: int = 0
    # Deferred diff-k params saved from prefill when SAGE_LAYERWISE_DEFER_DIFF_K=True.
    # Scoring runs in decode step 1 instead of pre-TTFT to reduce TTFT latency.
    layerwise_deferred_blend_params: Optional[Any] = None
    # Mean diff_k from scoring layer (per-token average).
    # Used by adaptive layerwise schedule to front-load layers when cache is stale.
    layerwise_diff_k_mean: Optional[float] = None


@dataclass
class IncrementalBlendStep:
    should_blend: bool
    recompute_ratio: Optional[float] = None
    reason: str = ""
    # Layer-wise only: (start_layer, end_layer) range for this step.
    layer_range: Optional[tuple[int, int]] = None


class IncrementalBlendStrategy(ABC):
    @abstractmethod
    def should_continue(self, state: IncrementalBlendState) -> bool:
        raise NotImplementedError

    @abstractmethod
    def next_step(self, state: IncrementalBlendState, num_layers: int) -> IncrementalBlendStep:
        raise NotImplementedError


class TokenWiseIncrementalBlendStrategy(IncrementalBlendStrategy):
    """
    Token-wise incremental blending.

    Each decode step recomputes ``step_percent`` fraction of tokens,
    up to ``max_ratio`` cumulative budget.

    Example: step_percent=0.10, max_ratio=0.60
      decode step 1 -> cumulative 0.50 (0.40 pre-TTFT + 0.10)
      decode step 2 -> cumulative 0.60 (done)
    """

    def __init__(self, step_percent: float, max_ratio: float):
        self.step_percent = max(0.0, _normalize_ratio(step_percent))
        self.max_ratio = _normalize_ratio(max_ratio)
        logger.info(
            "[SAGE_INCREMENTAL] Token-wise strategy initialized: "
            "step_percent=%.4f, max_ratio=%.4f",
            self.step_percent,
            self.max_ratio,
        )

    def should_continue(self, state: IncrementalBlendState) -> bool:
        should_continue = state.cumulative_ratio + 1e-8 < self.max_ratio
        if not should_continue:
            logger.info(
                "[SAGE_INCREMENTAL] Recompute budget exhausted: "
                "decode_step=%d, cumulative_ratio=%.4f, max_ratio=%.4f",
                state.decode_step,
                state.cumulative_ratio,
                self.max_ratio,
            )
        return should_continue

    def next_step(self, state: IncrementalBlendState, num_layers: int) -> IncrementalBlendStep:
        del num_layers
        state.decode_step += 1
        if not self.should_continue(state):
            step = IncrementalBlendStep(
                should_blend=False,
                reason="incremental token-wise budget exhausted",
            )
            logger.info("[SAGE_INCREMENTAL] %s", step.reason)
            return step

        next_ratio = min(self.max_ratio, state.cumulative_ratio + self.step_percent)
        if next_ratio <= state.cumulative_ratio + 1e-8:
            step = IncrementalBlendStep(
                should_blend=False,
                reason="token-wise ratio did not increase",
            )
            logger.info("[SAGE_INCREMENTAL] %s", step.reason)
            return step

        state.cumulative_ratio = next_ratio
        step = IncrementalBlendStep(
            should_blend=True,
            recompute_ratio=state.cumulative_ratio,
            reason=(
                f"token-wise decode_step={state.decode_step}, "
                f"delta={self.step_percent:.4f}, cumulative_ratio={state.cumulative_ratio:.4f}"
            ),
        )
        logger.info("[SAGE_INCREMENTAL] %s", step.reason)
        return step


class LayerWiseIncrementalBlendStrategy(IncrementalBlendStrategy):
    """
    Layer-wise incremental blending.

    Tokens are selected once (at pre-TTFT time via diff_k at layer 1).
    Each decode step recomputes those selected tokens through the next
    ``step_num_layers`` transformer layers, writing corrected K/V only
    for those layers.

    Example with num_layers=36, step_num_layers=6, pre_ttft_layers=18:
      decode step 1 -> layers 18-23
      decode step 2 -> layers 24-29
      decode step 3 -> layers 30-35  (done)
    """

    def __init__(self, step_num_layers: int, num_model_layers: int):
        if step_num_layers <= 0:
            raise ValueError(
                f"step_num_layers must be positive, got {step_num_layers}"
            )
        self.step_num_layers = step_num_layers
        self.num_model_layers = num_model_layers
        # Adaptive schedule: front-load more layers when diff_k exceeds threshold.
        self.adaptive_threshold = float(
            os.environ.get("SAGE_LAYERWISE_ADAPTIVE_THRESHOLD", "0")
        )
        self.adaptive_aggressive_layers = int(
            os.environ.get("SAGE_LAYERWISE_ADAPTIVE_AGGRESSIVE_LAYERS", str(step_num_layers * 3))
        )
        self.adaptive_aggressive_steps = int(
            os.environ.get("SAGE_LAYERWISE_ADAPTIVE_AGGRESSIVE_STEPS", "2")
        )
        logger.info(
            "[SAGE_INCREMENTAL] Layer-wise strategy initialized: "
            "step_num_layers=%d, num_model_layers=%d, "
            "adaptive_threshold=%.2f, aggressive_layers=%d, aggressive_steps=%d",
            self.step_num_layers,
            self.num_model_layers,
            self.adaptive_threshold,
            self.adaptive_aggressive_layers,
            self.adaptive_aggressive_steps,
        )

    def should_continue(self, state: IncrementalBlendState) -> bool:
        should_cont = state.layerwise_current_layer < self.num_model_layers
        if not should_cont:
            logger.info(
                "[SAGE_INCREMENTAL] Layer-wise budget exhausted: "
                "decode_step=%d, current_layer=%d, num_model_layers=%d",
                state.decode_step,
                state.layerwise_current_layer,
                self.num_model_layers,
            )
        return should_cont

    def next_step(
        self, state: IncrementalBlendState, num_layers: int
    ) -> IncrementalBlendStep:
        state.decode_step += 1
        if not self.should_continue(state):
            return IncrementalBlendStep(
                should_blend=False,
                reason="incremental layer-wise budget exhausted",
            )

        # Adaptive: front-load more layers when diff_k exceeds threshold.
        # Use layerwise_recompute_steps (counts only actual recompute steps,
        # excluding the deferred diff-k scoring step).
        recompute_step = getattr(state, "_layerwise_recompute_steps", 0) + 1
        state._layerwise_recompute_steps = recompute_step
        diff_k_mean = state.layerwise_diff_k_mean
        use_aggressive = (
            self.adaptive_threshold > 0
            and diff_k_mean is not None
            and diff_k_mean > self.adaptive_threshold
            and recompute_step <= self.adaptive_aggressive_steps
        )
        layers_this_step = (
            self.adaptive_aggressive_layers if use_aggressive
            else self.step_num_layers
        )

        start_layer = state.layerwise_current_layer
        end_layer = min(
            start_layer + layers_this_step,
            self.num_model_layers,
        )
        # Advance the state cursor so should_continue() reflects progress.
        state.layerwise_current_layer = end_layer
        _dk_str = f", diff_k_mean={diff_k_mean:.2f}" if diff_k_mean is not None else ""
        step = IncrementalBlendStep(
            should_blend=True,
            layer_range=(start_layer, end_layer),
            reason=(
                f"layer-wise decode_step={state.decode_step}, "
                f"layers=[{start_layer}, {end_layer}) "
                f"[{'aggressive' if use_aggressive else 'normal'}{_dk_str}]"
            ),
        )
        logger.info("[SAGE_INCREMENTAL] %s", step.reason)
        return step


def build_incremental_blend_strategy(config) -> Optional[IncrementalBlendStrategy]:
    strategy_name = (
        getattr(config, "blend_incremental_strategy", None) or "none"
    ).strip().lower()

    if strategy_name in ("none", "off", "disabled"):
        logger.info("[SAGE_INCREMENTAL] Incremental blending disabled.")
        return None

    max_ratio = getattr(config, "blend_incremental_max_ratio", None)
    if max_ratio is None:
        # Token-wise defaults to a full 100% cumulative budget unless explicitly
        # capped (e.g., 40% with blend_incremental_max_ratio=0.4).
        max_ratio = 1.0
    max_ratio = _normalize_ratio(float(max_ratio))

    if strategy_name in ("token", "token_wise", "token-wise"):
        k = getattr(config, "tokenwise_step_ratio", None)
        step_percent = _normalize_ratio(float(k)) if k is not None else 0.05
        return TokenWiseIncrementalBlendStrategy(
            step_percent=step_percent,
            max_ratio=float(max_ratio),
        )

    if strategy_name in ("layer", "layer_wise", "layer-wise"):
        step_num_layers = getattr(config, "layerwise_step_num_layers", None)
        if step_num_layers is None:
            step_num_layers = 6
        step_num_layers = int(step_num_layers)
        # num_model_layers is not available from config alone; it will be
        # patched by the adapter after model initialization.  Use a sentinel.
        return LayerWiseIncrementalBlendStrategy(
            step_num_layers=step_num_layers,
            num_model_layers=0,  # patched by adapter
        )

    logger.error(
        "[SAGE_INCREMENTAL] Unsupported incremental strategy '%s'. "
        "Supported values: none, token_wise, layer_wise.",
        strategy_name,
    )
    raise ValueError(
        f"Unknown blend incremental strategy '{strategy_name}'. "
        "Supported values: none, token_wise, layer_wise."
    )
