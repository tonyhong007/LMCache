# SPDX-License-Identifier: Apache-2.0
# Standard
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Generator, Optional, Union
import os
import threading
import time
import traceback

import numpy as np

# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import get_pp_group
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus
from vllm.version import __version__ as VLLM_VERSION
import torch
try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func as _vllm_flash_attn_varlen_func
except Exception:
    _vllm_flash_attn_varlen_func = None

# First Party
# Use LMCache's own math utilities instead of vllm's
# (avoids dependency on vllm internal changes like https://github.com/vllm-project/vllm/pull/27188)
from lmcache import utils
from lmcache.integration.vllm.utils import (
    ENGINE_NAME,
    apply_mm_hashes_to_token_ids,
    extract_mm_features,
    lmcache_get_or_create_config,
)
from lmcache.logging import init_logger
from lmcache.observability import LMCStatsMonitor, PrometheusLogger
from lmcache.utils import CacheStoreEvent, _lmcache_nvtx_annotate, cdiv
from lmcache.v1.cache_engine import LMCacheEngine
from lmcache.v1.compute.blend import LMCBlenderBuilder
from lmcache.v1.compute.blend.incremental_strategy import (
    IncrementalBlendState,
    IncrementalBlendStrategy,
    LayerWiseIncrementalBlendStrategy,
    MagnetIncrementalBlendStrategy,
    TokenWiseIncrementalBlendStrategy,
    build_incremental_blend_strategy,
)
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryFormat

try:
    import lmcache.c_ops as lmc_ops
except ImportError:
    lmc_ops = None
from lmcache.v1.config_base import validate_and_set_config_value
from lmcache.v1.manager import LMCacheManager

if TYPE_CHECKING:
    # Third Party
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.sched.output import NewRequestData
    from vllm.v1.request import Request

    # First Party
    from lmcache.v1.lookup_client.abstract_client import LookupClientInterface

logger = init_logger(__name__)


def _normalize_ratio(value: float) -> float:
    ratio = float(value)
    if ratio < 0.0:
        return 0.0
    if ratio > 1.0:
        ratio = ratio / 100.0
    return min(ratio, 1.0)


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in LMCache
    lmcache_cached_tokens: int
    # Whether the scheduler allow us to load the tokens
    can_load: bool


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allow us to save the tokens
    can_save: bool


@dataclass
class DisaggSpec:
    req_id: str
    receiver_id: str
    receiver_host: str
    receiver_init_port: int
    receiver_alloc_port: int
    is_last_prefill: bool = False
    num_transferred_tokens: int = 0


tmp_disagg_tracker: dict[str, DisaggSpec] = {}


def extract_request_configs(sampling_params: SamplingParams) -> Optional[dict]:
    request_configs = None
    if sampling_params.extra_args is not None:
        if kv_transfer_params := sampling_params.extra_args.get("kv_transfer_params"):
            for k, v in kv_transfer_params.items():
                if k.startswith("lmcache."):
                    if request_configs is None:
                        request_configs = {}
                    request_configs[k] = v
    return request_configs


@dataclass
class RequestTracker:
    # Request id
    req_id: str

    # Total prompt token length
    prompt_len: int

    # The token ids that has been scheduled so far
    token_ids: list[int]

    # The block ids that has been allocated so far
    # NOTE: allocated blocks could be more than the number of tokens
    allocated_block_ids: list[int]

    # The number of tokens that has been saved
    num_saved_tokens: int = 0

    # Disagg spec for the request
    disagg_spec: Optional[DisaggSpec] = None

    # Multimodal hashes and positions
    mm_hashes: Optional[list[str]] = None
    mm_positions: Optional[list["PlaceholderRange"]] = None

    # The configs of the request, includes tags and other configs
    request_configs: Optional[dict] = None

    # Whether the request is in decode phase
    is_decode_phase = False

    # Whether the request cache should be saved
    skip_save: bool = False

    @_lmcache_nvtx_annotate
    @staticmethod
    def from_new_request(
        lmcache_config: LMCacheEngineConfig,
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
        lmcache_cached_tokens: int,
        skip_save: bool,
    ) -> "RequestTracker":
        """Create the request tracker from a new request.

        Args:
            lmcache_config (LMCacheEngineConfig): the LMCache engine config.
            new_request (NewRequestData): the new request data.
            num_tokens_to_compute (int): the number of tokens that will
                be 'computed', including the `num_computed_tokens` (vLLM's
                local cache hit) and new tokens that will be scheduled.
            lmcache_cached_tokens (int): the number of tokens that are
                cached in LMCache.
            request_priority (int): the priority of the request
            skip_save (bool): whether the request cache should be saved
        """
        # vLLM 0.9.0 update: request.block_ids changed from list[int] to
        # tuple[list[int]]
        # Need to check the type of request.block_ids

        unfolded_block_ids = []

        if not isinstance(new_request.block_ids[0], list):
            unfolded_block_ids = new_request.block_ids.copy()
        else:
            # According to the vLLM code
            # (https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/
            # sched/scheduler.py#L943),
            # only one KVCacheGroup is supported in connector for now.

            # TODO: Please support multiple KVCacheGroup in connector.
            # NOTE: Also, `update` method in RequestTracker should be
            # updated accordingly.
            unfolded_block_ids = new_request.block_ids[0].copy()

        # NOTE: Initialized in `update_state_after_alloc`
        disagg_spec = tmp_disagg_tracker.pop(new_request.req_id, None)

        request_configs = extract_request_configs(new_request.sampling_params)

        mm_hashes, mm_positions = extract_mm_features(new_request, modify=True)

        return RequestTracker(
            req_id=new_request.req_id,
            prompt_len=len(new_request.prompt_token_ids),
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=unfolded_block_ids,
            num_saved_tokens=lmcache_cached_tokens,
            disagg_spec=disagg_spec,
            mm_hashes=mm_hashes,
            mm_positions=mm_positions,
            skip_save=skip_save,
            request_configs=request_configs,
        )

    def update(
        self,
        new_token_ids: list[int],
        new_block_ids: Union[Optional[tuple[list[int], ...]], list[int]],
        preempted: bool = False,
        lmcache_cached_tokens: int = 0,
        vllm_cached_tokens: int = 0,
        all_token_ids: Optional[list[int]] = None,
    ) -> None:
        """Update the request tracker when a running request is
        scheduled again

        vllm_cached_tokens: the number of tokens that are cached in vLLM
        is only used for preempted requests
        all_token_ids: the full token list from the vLLM request, used to
        restore token_ids for preempted requests to ensure chunk keys match
        """

        if new_block_ids is None:
            # https://github.com/vllm-project/vllm/commit/
            # b029de9902aa3ac58806c8c17776c7074175b6db#
            # diff-cafd89ce8a698a56acb24ada62831cbc7a980782f78a52d1742ba238031f296cL94
            new_block_ids = []
        elif len(new_block_ids) == 0:
            new_block_ids = []
        elif isinstance(new_block_ids, tuple):
            new_block_ids = new_block_ids[0]
        elif isinstance(new_block_ids, list):
            pass
        else:
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")

        if preempted:
            assert all_token_ids is not None, (
                f"Preempted request {self.req_id} has no all_token_ids"
            )
            # the block ids will change after preemption
            self.allocated_block_ids = new_block_ids
            # reset the number of saved tokens
            self.num_saved_tokens = lmcache_cached_tokens
            num_computed_tokens = max(lmcache_cached_tokens, vllm_cached_tokens)

            # For preempted requests, restore token_ids from the full token list.
            # We need enough tokens for retrieve - tokens[:lmcache_cached_tokens]
            # will give us the prompt tokens which match the stored cache keys.
            num_tokens_needed = max(
                num_computed_tokens + len(new_token_ids),
                lmcache_cached_tokens,
            )
            self.token_ids = all_token_ids[:num_tokens_needed]
        else:
            self.allocated_block_ids.extend(new_block_ids)
            self.token_ids.extend(new_token_ids)

        # When a request is scheduled again, and the number of new tokens
        # is 1 (excluding chunked prefill), the request is in decode phase.
        # TODO: Need to further exclude the case of chunked prefill with 1 token.
        if len(new_token_ids) == 1:
            self.is_decode_phase = True


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: list[int]  # torch.Tensor
    # Slot mapping
    slot_mapping: torch.Tensor
    # Full allocated block IDs (may exceed ceil(len(token_ids)/block_size)
    # under SAGE zero-copy when chunks contribute partial-last-block extras).
    allocated_block_ids: Optional[torch.Tensor] = None

    # Whether is last prefill or not
    is_last_prefill: bool = False

    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None
    # disagg spec
    disagg_spec: Optional[DisaggSpec] = None
    # the configs of the request
    request_configs: Optional[dict] = None
    # Sage zero-copy: True if chunk blocks were transferred to this request.
    # LMCache may still run incremental blending/recompute on top of transferred KV.
    sage_blocks_transferred: bool = False
    # Sage chunk boundary positions for GPU-direct blending (e.g., [0, 2770, 4615, ...])
    sage_chunk_boundaries: Optional[list[int]] = None
    # Number of query/suffix tokens that should always be recomputed during blending.
    query_token_count: int = 0
    # Number of emitted output tokens observed by scheduler for this request.
    # Used to detect pre-TTFT step (num_output_tokens <= 0).
    num_output_tokens: int = 0

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        lmcache_chunk_size: int = 256,
        load_spec: Optional[LoadSpec] = None,
        discard_partial_chunks: bool = True,
        save_decode_cache: bool = False,
        num_output_tokens: Optional[int] = None,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker.

        Args:
            tracker (RequestTracker): the request tracker.
            block_size (int): the block size in vLLM.
            lmcache_chunk_size (int): the chunk size for LMCache.
            load_spec (Optional[LoadSpec]): the load spec for KV cache loading.
            discard_partial_chunks (bool): whether to discard partial chunks.
            save_decode_cache (bool): whether to save the cache in decode phase.

        Returns:
            the request metadata if we need to perform load/save
            operations, None otherwise.
        """
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)

        is_last_prefill = False
        if input_token_len >= tracker.prompt_len:
            is_last_prefill = True

        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        # 3. if save_decode_cache is False and it is in decode phase

        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = (
            cdiv(tracker.num_saved_tokens + 1, lmcache_chunk_size) * lmcache_chunk_size
        )

        # NOTE(vladnosiv): for disagg, you cannot skip saving, as saving is a transfer
        # Check if request_configs has lmcache.skip_save set to True
        request_skip = (tracker.request_configs or {}).get("lmcache.skip_save", False)

        skip_save = tracker.disagg_spec is None and (
            tracker.skip_save
            or (tracker.num_saved_tokens > 0 and input_token_len < chunk_boundary)
            or (tracker.is_decode_phase and not save_decode_cache)
            or request_skip
        )

        if skip_save and load_spec is None:
            return None

        # Calculate number of tokens to save based on discard_partial_chunks
        # setting

        # NOTE(vladnosiv): for the input_token_len chunk prefill,
        # we are required to discard partial chunks,
        # as new tokens will be added in the next iteration.
        if not is_last_prefill or discard_partial_chunks:
            num_tokens_to_save = (
                input_token_len // lmcache_chunk_size * lmcache_chunk_size
            )
        else:
            num_tokens_to_save = input_token_len

        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)

        # Calculate the token ids and slot mappings for load and save
        token_ids = input_token_ids[:num_tokens_to_save]

        # If the request has multimodal hashes, apply them to the token ids
        if tracker.mm_hashes:
            # TODO: Optimize this
            token_ids = torch.tensor(token_ids)
            assert tracker.mm_positions is not None, (
                "tracker got mm_hashes but no mm_positions"
            )
            apply_mm_hashes_to_token_ids(
                token_ids, tracker.mm_hashes, tracker.mm_positions
            )
            token_ids = token_ids.tolist()

        num_blocks = len(tracker.allocated_block_ids)

        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks"
                " for request %s. "
                "Something might be wrong in scheduling logic!",
                tracker.req_id,
            )
            logger.error(
                "Num tokens: %d, num blocks: %d, block size: %d",
                len(token_ids),
                num_blocks,
                block_size,
            )

        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids.reshape((num_blocks, 1)) * block_size
        )

        slot_mapping = slot_mapping.flatten()[: len(token_ids)]
        assert slot_mapping.dtype == torch.long  # TODO: this could be removed

        # For load operation: check whether the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(
                "Scheduled to load %d tokens (%d cached in vLLM) for request %s",
                load_spec.lmcache_cached_tokens,
                load_spec.vllm_cached_tokens,
                tracker.req_id,
            )
        else:
            # Do not load if not in `can_load` state
            load_spec = None

        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            allocated_block_ids=block_ids,
            is_last_prefill=is_last_prefill,
            save_spec=save_spec,
            load_spec=load_spec,
            disagg_spec=tracker.disagg_spec,
            request_configs=tracker.request_configs,
            num_output_tokens=(
                max(0, int(num_output_tokens)) if num_output_tokens is not None else 0
            ),
        )


@dataclass
class LMCacheConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta] = field(default_factory=list)
    @_lmcache_nvtx_annotate
    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata.

        Args:
            req_meta (ReqMeta): the request metadata.
        """
        self.requests.append(req_meta)


class SageMethod(str, Enum):
    """Which SAGE recomputation method is active for this connector."""
    NONE = "none"                          # baseline — no SAGE recompute
    CACHEBLEND = "cacheblend"              # sync CacheBlend
    CACHEBLEND_TOKENWISE = "cacheblend-tokenwise"
    CACHEBLEND_LAYERWISE = "cacheblend-layerwise"
    MAGNET = "magnet"                      # query-aware adaptive (ProphetKV-style)


class LMCacheConnectorV1Impl:
    # Shared across all adapter instances in the same process (WORKER + SCHEDULER).
    # Used to propagate early_blend_done from sage_process_layer (WORKER adapter)
    # to the blend_from_gpu check (SCHEDULER adapter).
    _shared_early_blend_done: dict[str, bool] = {}

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        parent: KVConnectorBase_V1,
    ):
        self._parent = parent
        self._vllm_config = vllm_config
        self._role = role
        self.device = vllm_config.device_config.device
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        self.worker_count = vllm_config.parallel_config.tensor_parallel_size

        # Load and configure LMCache config
        config = lmcache_get_or_create_config()
        assert isinstance(config, LMCacheEngineConfig), (
            "LMCache v1 configuration is should be passed for vLLM v1."
        )
        self._apply_extra_config(config, vllm_config)
        self.config = config

        # Initialize LMCacheManager to handle internal components
        self._manager = LMCacheManager(
            config=config,
            vllm_config=vllm_config,
            role=role.name.lower(),
            connector=self,
        )

        # Start services managed by LMCacheManager
        self._manager.start_services()

        # Initialize connector-specific state
        self._init_connector_state(role, vllm_config, config)

        # Setup metrics for monitoring data structures
        self._setup_metrics()

        logger.info(
            "LMCache initialized for role %s with version %s, "
            "vllm version %s, lmcache cache_engine metadata: %s",
            role,
            utils.get_version(),
            VLLM_VERSION,
            getattr(self.lmcache_engine, "metadata", None),
        )

    def _apply_extra_config(
        self, config: LMCacheEngineConfig, vllm_config: "VllmConfig"
    ) -> None:
        """Apply extra config from vLLM to LMCache config."""
        kv_connector_extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
        )
        if kv_connector_extra_config:
            for key, value in kv_connector_extra_config.items():
                if key.startswith("lmcache."):
                    config_key = key[8:]  # Remove "lmcache." prefix
                    if validate_and_set_config_value(config, config_key, value):
                        logger.info(
                            "Updated config %s from vLLM extra config: %s",
                            config_key,
                            value,
                        )

    def _init_connector_state(
        self,
        role: KVConnectorRole,
        vllm_config: "VllmConfig",
        config: LMCacheEngineConfig,
    ) -> None:
        """Initialize connector-specific state variables."""
        self.async_loading = config.enable_async_loading
        self.layerwise_retrievers: list[
            Generator[Optional[torch.Tensor], None, None]
        ] = []
        self.layerwise_storers: list[Generator[Optional[torch.Tensor], None, None]] = []
        self._stats_monitor = LMCStatsMonitor.GetOrCreate()
        self.enable_sage = os.environ.get("ENABLE_SAGE", "False").lower() == "true"
        self.enable_blending = config.enable_blending
        self._incremental_blend_strategy: Optional[IncrementalBlendStrategy] = (
            build_incremental_blend_strategy(config)
            if self.enable_blending
            else None
        )
        # Fraction of tokens to recompute synchronously before TTFT
        # (tokenwise only). e.g. 0.2 means 20% of top-diff tokens pre-TTFT.
        self._tokenwise_pre_ttft_ratio: float = _normalize_ratio(
            float(os.getenv("SAGE_TOKENWISE_PRE_TTFT_RATIO", "0.0"))
        )
        # Fraction of layers to recompute pre-TTFT (layerwise only, 0-1).
        # e.g. 0.5 means recompute layers 0-17 of 36 before TTFT.
        self._layerwise_pre_ttft_ratio: float = _normalize_ratio(
            float(os.getenv("SAGE_LAYERWISE_PRE_TTFT_RATIO", "0.5"))
        )
        # When True, skip diff-k scoring pre-TTFT and defer it to decode step 1.
        self._layerwise_defer_diff_k: bool = (
            os.getenv("SAGE_LAYERWISE_DEFER_DIFF_K", "").lower() in ("1", "true")
        )
        # Derive SageMethod: use incremental strategy if configured,
        # otherwise default to sync cacheblend.
        if not self.enable_sage:
            self._sage_method = SageMethod.NONE
        elif isinstance(self._incremental_blend_strategy, LayerWiseIncrementalBlendStrategy):
            self._sage_method = SageMethod.CACHEBLEND_LAYERWISE
        elif isinstance(self._incremental_blend_strategy, TokenWiseIncrementalBlendStrategy):
            self._sage_method = SageMethod.CACHEBLEND_TOKENWISE
        elif isinstance(self._incremental_blend_strategy, MagnetIncrementalBlendStrategy):
            self._sage_method = SageMethod.MAGNET
        else:
            self._sage_method = SageMethod.CACHEBLEND
        self._sync_cacheblend = (
            self._sage_method == SageMethod.CACHEBLEND
        )
        logger.info(
            "[SAGE_CONFIG] sage_method=%s",
            self._sage_method.value,
        )
        # req_id → (slot_mapping, num_tokens) for deferred hash firing
        self._sage_suffix_rerecompute_pending: set[str] = set()
        # req_id → {positions, token_ids, slots} for next-step injection
        self._fused_inject_pending: dict[str, dict] = {}
        # req_ids that had injection in the current model forward step
        self._fused_inject_active: set[str] = set()
        # PyTorch hook handles registered during injection; cleaned up after forward.
        self._sage_inject_hook_handles: list = []
        self._sage_incremental_states: dict[str, IncrementalBlendState] = {}
        # req_id -> cached full-prompt tensors (token_ids, slot_mapping) for injection.
        self._sage_cached_prompt_inputs: dict[str, dict[str, Any]] = {}
        self._sage_pending_blend_launch_t0: dict[str, float] = {}
        # Requests that have already had RoPE adjusted once for transferred KV.
        self._sage_rope_adjusted_requests: set[str] = set()
        # Requests that have fully consumed incremental recompute budget.
        self._sage_incremental_completed_requests: set[str] = set()

        # Role-specific initialization
        if role == KVConnectorRole.SCHEDULER:
            self._unfinished_requests: dict[str, "Request"] = {}
        else:
            self.use_layerwise = config.use_layerwise

            if self.enable_blending:
                assert self.lmcache_engine is not None
                assert self.lmcache_engine.gpu_connector is not None, (
                    "GPU connector must be available for blending"
                )
                self.blender = LMCBlenderBuilder.get_or_create(
                    ENGINE_NAME,
                    self.lmcache_engine,
                    self.lmcache_engine.gpu_connector,
                    config,
                )
                # Patch LayerWiseIncrementalBlendStrategy with actual model
                # layer count (only known after blender builds the model).
                if isinstance(
                    self._incremental_blend_strategy,
                    LayerWiseIncrementalBlendStrategy,
                ):
                    self._incremental_blend_strategy.num_model_layers = (
                        self.blender.num_layers
                    )
                    logger.info(
                        "[SAGE_LAYERWISE] Patched num_model_layers=%d "
                        "step_num_layers=%d",
                        self.blender.num_layers,
                        self._incremental_blend_strategy.step_num_layers,
                    )
                if self._incremental_blend_strategy is not None:
                    logger.info(
                        "[SAGE_INCREMENTAL] Enabled strategy=%s, "
                        "scoring_method=diff_k",
                        type(self._incremental_blend_strategy).__name__,
                    )
                elif self.enable_sage and self.enable_blending:
                    logger.info(
                        "[SAGE_INCREMENTAL] Incremental prefill disabled; "
                        "using synchronous full blending before first token.",
                    )

        # Legacy compatibility check
        self._check_legacy_register_kv_caches()

        self.kv_caches: dict[str, torch.Tensor] = {}
        self._block_size = vllm_config.cache_config.block_size
        self.load_specs: dict[str, LoadSpec] = {}
        self.kv_cache_manager: Optional["KVCacheManager"] = None
        self._request_trackers: dict[str, RequestTracker] = {}

        self._discard_partial_chunks = (
            vllm_config.kv_transfer_config.get_from_extra_config(
                "discard_partial_chunks", False
            )
            or not config.save_unfull_chunk
        )

        self._lmcache_chunk_size = config.chunk_size
        self._save_decode_cache = config.save_decode_cache

        self.skip_last_n_tokens = vllm_config.kv_transfer_config.get_from_extra_config(
            "skip_last_n_tokens", 0
        )

        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.current_layer = 0

        self.force_skip_save = bool(os.environ.get("LMCACHE_FORCE_SKIP_SAVE", False))
        self._requests_priority: dict[str, int] = {}
        self._invalid_block_ids: set[int] = set()

    def _check_legacy_register_kv_caches(self) -> None:
        """Check for legacy connector without register_kv_caches implementation."""
        if self.lmcache_engine is None:
            return

        child_class = self._parent.__class__
        parent_class = KVConnectorBase_V1
        child_method = getattr(child_class, "register_kv_caches", None)
        parent_method = getattr(parent_class, "register_kv_caches", None)

        if child_method is None or parent_method is None:
            implements = False
        else:
            implements = child_method is not parent_method

        if not implements:
            logger.warning(
                "Please use the latest lmcache connector, otherwise some "
                "features may not work, such as DSA"
            )
            self._manager.post_init()

    # ==================== Property Accessors ====================

    @property
    def lmcache_engine(self) -> Optional[LMCacheEngine]:
        """Get the LMCache engine instance from manager."""
        return self._manager.lmcache_engine

    @property
    def lmcache_engine_metadata(self):
        """Get the LMCache engine metadata from manager."""
        return self._manager.lmcache_engine_metadata

    @property
    def lookup_client(self) -> Optional["LookupClientInterface"]:
        """Get the lookup client from manager."""
        return self._manager.lookup_client

    @property
    def lookup_server(self):
        """Get the lookup server from manager."""
        return self._manager.lookup_server

    def _setup_metrics(self):
        """Setup metrics for monitoring data structures in the connector."""
        prometheus_logger = PrometheusLogger.GetInstanceOrNone()
        if prometheus_logger is None:
            logger.warning(
                "PrometheusLogger is not initialized, "
                "connector metrics will not be collected"
            )
            return

        # Set up metrics for scheduler-specific and general data structures
        metrics_map = {
            "_unfinished_requests": "scheduler_unfinished_requests_count",
            "load_specs": "connector_load_specs_count",
            "_request_trackers": "connector_request_trackers_count",
            "kv_caches": "connector_kv_caches_count",
            "layerwise_retrievers": "connector_layerwise_retrievers_count",
            "_invalid_block_ids": "connector_invalid_block_ids_count",
            "_requests_priority": "connector_requests_priority_count",
        }

        for attr_name, metric_name in metrics_map.items():
            if hasattr(self, attr_name):
                metric = getattr(prometheus_logger, metric_name)
                # Use a default argument in the lambda to capture
                # the current value of `attr_name`
                # to avoid issues with late binding in closures.
                metric.set_function(lambda name=attr_name: len(getattr(self, name)))

    def get_inference_info(self) -> dict:
        """Get inference information including vLLM config and related details.

        Returns:
            dict: Dictionary containing inference information
        """
        # Get vLLM config information
        vllm_config = self._vllm_config

        # Use vLLM config's string representation and add specific configs
        inference_info = {
            "vllm_version": VLLM_VERSION,
            "lmcache_version": utils.get_version(),
            "vllm_config": str(vllm_config),
            "model_config": {
                "model": getattr(vllm_config.model_config, "model", None),
                "dtype": str(getattr(vllm_config.model_config, "dtype", None)),
                "max_model_len": getattr(
                    vllm_config.model_config, "max_model_len", None
                ),
                "vocab_size": getattr(vllm_config.model_config, "vocab_size", None),
                "num_layers": getattr(
                    vllm_config.model_config, "get_num_layers", lambda _: None
                )(vllm_config.parallel_config),
                "num_attention_heads": getattr(
                    vllm_config.model_config, "get_num_attention_heads", lambda _: None
                )(vllm_config.parallel_config),
                "num_kv_heads": getattr(
                    vllm_config.model_config, "get_num_kv_heads", lambda _: None
                )(vllm_config.parallel_config),
                "head_size": getattr(
                    vllm_config.model_config, "get_head_size", lambda: None
                )(),
            },
            "cache_config": {
                "block_size": getattr(vllm_config.cache_config, "block_size", None),
                "cache_dtype": str(
                    getattr(vllm_config.cache_config, "cache_dtype", None)
                ),
                "gpu_memory_utilization": getattr(
                    vllm_config.cache_config, "gpu_memory_utilization", None
                ),
                "swap_space": getattr(vllm_config.cache_config, "swap_space", None),
                "enable_prefix_caching": getattr(
                    vllm_config.cache_config, "enable_prefix_caching", None
                ),
            },
        }

        return inference_info

    def get_inference_version(self) -> str:
        """Get vLLM version information.

        Returns:
            str: vLLM version string
        """
        return VLLM_VERSION

    def _build_kv_layer_groups(self):
        # Build KV layer groups structure if not already built
        if self.lmcache_engine is not None:
            assert len(self.kv_caches) > 0
            kv_layer_groups_manager = (
                self.lmcache_engine.metadata.kv_layer_groups_manager
            )
            kv_layer_groups_manager.build_kv_layer_groups(self.kv_caches)

    def _get_or_create_sage_incremental_state(self, req_id: str) -> Optional[IncrementalBlendState]:
        if req_id not in self._sage_incremental_states:
            self._sage_incremental_states[req_id] = IncrementalBlendState()
            logger.info(
                "[SAGE_INCREMENTAL] Created state for request=%s",
                req_id,
            )
        return self._sage_incremental_states[req_id]

    def _should_schedule_incremental_blend(
        self, req_id: str, create_if_missing: bool = True
    ) -> bool:
        if self._incremental_blend_strategy is None:
            logger.debug(
                "[SAGE_INCREMENTAL] incremental blending disabled for request=%s",
                req_id,
            )
            return False
        if req_id in self._sage_incremental_completed_requests:
            logger.debug(
                "[SAGE_INCREMENTAL] request=%s already completed incremental recompute; "
                "skip scheduling",
                req_id,
            )
            return False
        strategy = getattr(self, "_incremental_blend_strategy", None)
        if strategy is None:
            return False
        state = self._sage_incremental_states.get(req_id)
        if state is None:
            if not create_if_missing:
                logger.debug(
                    "[SAGE_INCREMENTAL] request=%s has no state yet; "
                    "allow scheduling without creating state",
                    req_id,
                )
                return True
            state = self._get_or_create_sage_incremental_state(req_id)
        assert state is not None
        should_continue = strategy.should_continue(state)
        logger.debug(
            "[SAGE_INCREMENTAL] request=%s should_schedule=%s "
            "(decode_step=%d cumulative_ratio=%.4f)",
            req_id,
            should_continue,
            state.decode_step,
            state.cumulative_ratio,
        )
        return should_continue

    def _cleanup_sage_incremental_state(self, req_id: str) -> None:
        state = self._sage_incremental_states.pop(req_id, None)
        self._sage_pending_blend_launch_t0.pop(req_id, None)
        self._sage_cached_prompt_inputs.pop(req_id, None)
        LMCacheConnectorV1Impl._shared_early_blend_done.pop(req_id, None)
        self._sage_rope_adjusted_requests.discard(req_id)
        self._sage_incremental_completed_requests.discard(req_id)
        self._fused_inject_pending.pop(req_id, None)
        self._fused_inject_active.discard(req_id)

    def _mark_sage_recompute_completed(
        self,
        req_id: str,
        state: Any,
        ratio: float,
    ) -> None:
        """Mark a request's SAGE recomputation as completed and clean up."""
        self._fused_inject_pending.pop(req_id, None)
        if state is not None:
            state.frozen_ratio_cursor = ratio
            state.cumulative_ratio = ratio
        self._sage_incremental_completed_requests.add(req_id)

    def _compute_virtual_slot_mapping(
        self,
        contiguous_slot_mapping: torch.Tensor,
        chunk_boundaries: list[int],
        num_tokens: int,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        """Remap contiguous slot_mapping to virtual layout for chunk boundaries.

        After zero-copy, each chunk starts at a new block. The contiguous
        slot_mapping assumes continuous block filling, which is wrong at chunk
        boundaries (partially-filled last blocks create gaps). This computes
        the correct virtual slot for each token.

        Computed on CPU to avoid many small GPU kernel launches, then
        transferred to GPU as a single tensor.
        """
        block_size = self._block_size
        device = contiguous_slot_mapping.device

        # Build cumulative block offsets on CPU (tiny: num_chunks elements).
        boundaries = list(chunk_boundaries)
        if boundaries[-1] != num_tokens:
            boundaries.append(num_tokens)
        num_chunks = len(boundaries) - 1
        cum_blocks = [0] * (num_chunks + 1)
        for i in range(num_chunks):
            chunk_len = boundaries[i + 1] - boundaries[i]
            cum_blocks[i + 1] = cum_blocks[i] + (chunk_len + block_size - 1) // block_size
        total_virtual_blocks = cum_blocks[num_chunks]

        # Get physical block IDs from block table (may be on GPU or CPU).
        bt_cpu = block_table[:total_virtual_blocks].cpu().to(torch.int64)

        # Compute virtual slots on CPU — one pass over N tokens.
        import numpy as np
        positions = np.arange(num_tokens, dtype=np.int64)
        chunk_id = np.searchsorted(boundaries[1:], positions, side="right")
        boundary_arr = np.array(boundaries, dtype=np.int64)
        pos_in_chunk = positions - boundary_arr[chunk_id]
        chunk_local_block = pos_in_chunk // block_size
        cum_blocks_arr = np.array(cum_blocks, dtype=np.int64)
        parent_block_idx = cum_blocks_arr[chunk_id] + chunk_local_block
        np.clip(parent_block_idx, 0, total_virtual_blocks - 1, out=parent_block_idx)
        bt_np = bt_cpu.numpy()
        actual_block_id = bt_np[parent_block_idx]
        pos_in_block = pos_in_chunk % block_size
        result_np = actual_block_id * block_size + pos_in_block

        return torch.from_numpy(result_np).to(device=device, dtype=torch.int64)

    def _run_rope_prepass_and_remap(
        self,
        req_id: str,
        chunk_boundaries: list[int],
        slot_mapping: torch.Tensor,
        num_tokens: int,
        rope_block_table: torch.Tensor,
        kvcaches: list,
        layer_range: tuple[int, int] | None = None,
        rope_only: bool = False,
    ) -> None:
        """RoPE correction + virtual→contiguous KV remap.

        1. Compute virtual slot mapping from chunk boundaries.
        2. Correct RoPE at virtual slots (chunk-local → absolute positions).
        3. Copy KV from virtual to contiguous slots.

        If layer_range is None, processes all layers and marks the request
        as RoPE-adjusted. If layer_range is (start, end), only processes
        layers [start, end) and does NOT mark the request as adjusted
        (caller must call again for remaining layers later).

        This partial mode is used by early-blend to RoPE-correct only the
        blend layers (0-1) while drain threads are still writing higher
        layers. The remaining layers are corrected later during
        _launch_final_request.

        rope_block_table is the per-request block_id list, derived from
        the request's slot_mapping (NOT from attn_metadata.block_table[idx]
        which uses the wrong indexing for multi-request batches).
        """
        assert rope_block_table is not None, (
            f"RoPE prepass: no block_table for {req_id}"
        )
        rope_block_table = rope_block_table.to(dtype=torch.int64).contiguous()

        virtual_sm = self._compute_virtual_slot_mapping(
            contiguous_slot_mapping=slot_mapping[:num_tokens],
            chunk_boundaries=chunk_boundaries,
            num_tokens=num_tokens,
            block_table=rope_block_table,
        )

        layer_start = 0
        layer_end = len(kvcaches)
        if layer_range is not None:
            layer_start, layer_end = layer_range

        # RoPE correction: the fused kernel operates on all layers at
        # once via key_cache_ptrs. For partial mode, we temporarily set
        # the gpu_connector's kvcaches to just the target layers so the
        # fused kernel only corrects those layers' KV. We save and
        # restore the original kvcaches to avoid corrupting the
        # connector's state.
        gc = self.blender.gpu_connector
        saved_kvcaches = gc.kvcaches
        saved_num_layers = getattr(gc, "num_layers", None)
        target_kvcaches = kvcaches[layer_start:layer_end]
        if target_kvcaches:
            # Temporarily set num_layers to match the target subset so
            # the fused kernel's key_cache_ptrs loop stays in bounds.
            gc.num_layers = len(target_kvcaches)
            self.blender.gpu_connector.rope_correction_inplace(
                slot_mapping=virtual_sm,
                chunk_boundaries=chunk_boundaries,
                num_tokens=num_tokens,
                kvcaches=target_kvcaches,
            )
        # Restore original state on the connector.
        gc.kvcaches = saved_kvcaches
        if saved_num_layers is not None:
            gc.num_layers = saved_num_layers

        # Copy KV from virtual to contiguous slots (skip if rope_only).
        # rope_only mode is used by early-blend: it only needs the RoPE
        # correction (so the blend reads correctly-positioned K values)
        # but should NOT remap because the blend's slot_mapping points
        # to virtual slots. The full remap happens later during the
        # parent's first forward step.
        if not rope_only:
            contiguous_sm = slot_mapping[:num_tokens].to(
                device=virtual_sm.device, dtype=torch.long,
            )
            gc2 = self.blender.gpu_connector
            gc2._lazy_initialize_buffer(kvcaches)
            buf = gc2.gpu_buffer_allocator.allocate(
                gc2.get_shape(num_tokens), gc2.dtype, MemoryFormat.KV_2TD,
            )
            for li in range(layer_start, layer_end):
                lmc_ops.single_layer_kv_transfer(
                    buf.tensor, kvcaches[li], virtual_sm,
                    True, False, gc2.vllm_two_major, gc2.use_mla,
                )
                lmc_ops.single_layer_kv_transfer(
                    buf.tensor, kvcaches[li], contiguous_sm,
                    False, False, gc2.vllm_two_major, gc2.use_mla,
                )
            buf.ref_count_down()
        if layer_range is None:
            self._sage_rope_adjusted_requests.add(req_id)

    def _run_tokenwise_pre_ttft(
        self,
        req_id: str,
        blend_tokens: torch.Tensor,
        sage_token_mask: torch.Tensor,
        blend_slot_mapping: torch.Tensor,
        blend_block_table: Optional[torch.Tensor],
        kvcaches: list,
        sage_zero_copy: bool,
        state: Any,
        suffix_len: int = 0,
    ) -> None:
        """Tokenwise incremental pre-TTFT: score + partial recompute.

        1. Run diff-k scoring at check layer, capture full ranking.
        2. Recompute top pre_ttft_ratio fraction through all layers.
        3. Store remaining ranked tokens on state for decode-step slicing.
        """
        pre_ttft_ratio = self._tokenwise_pre_ttft_ratio

        _check_layers = self.blender.common_metadata.check_layers or [1]
        _max_layers = (
            max(_check_layers) + 1 if pre_ttft_ratio == 0
            else self.blender.num_layers
        )
        self.blender.blend_from_gpu(
            blend_tokens,
            sage_token_mask,
            step_recompute_ratio=pre_ttft_ratio if pre_ttft_ratio > 0 else None,
            capture_full_ranking=True,
            save_boundary=True,
            suffix_len=suffix_len,
            include_suffix=(pre_ttft_ratio >= 1.0),
            kvcaches=kvcaches,
            slot_mapping=blend_slot_mapping,
            block_table=blend_block_table,
            sage_zero_copy=sage_zero_copy,
            max_layers_to_process=_max_layers,
        )

        full_ranking = self.blender._last_full_importance_ranking
        if full_ranking is None or full_ranking.numel() == 0:
            logger.warning(
                "[SAGE_CB_INC_FUSED] request=%s diff-k produced no ranking; "
                "falling back to completed",
                req_id,
            )
            self._mark_sage_recompute_completed(req_id, state, pre_ttft_ratio)
            return

        N = int(blend_tokens.shape[0])
        pre_ttft_count = max(1, int(N * pre_ttft_ratio))
        remaining_ranking = full_ranking[pre_ttft_count:]

        if remaining_ranking.numel() == 0:
            self._mark_sage_recompute_completed(req_id, state, pre_ttft_ratio)
            return

        # Update state so decode-step scheduler picks up the frozen ranking.
        if state is not None:
            state.frozen_importance_ranking = remaining_ranking
            state.frozen_ratio_cursor = 0.0
            state.cumulative_ratio = float(pre_ttft_ratio)
            state.pre_ttft_recomputed_tokens = pre_ttft_count
            state.decode_step = max(state.decode_step, 1)
            state.query_token_count = suffix_len
            state._tw_cursor = 0
            state._tw_tokens_emitted = 0
            state._tw_prev_ratio = float(pre_ttft_ratio)
            # Snapshot the scoring boundary into per-request state so that
            # later decode-step injection reads the right tensors even if
            # other concurrent requests overwrite the shared blender
            # `_scoring_boundary_*` fields between now and then. Cloned
            # because the blender owns the underlying buffers.
            _bnd_h = self.blender._scoring_boundary_hidden
            _bnd_r = self.blender._scoring_boundary_residual
            _bnd_l = self.blender._scoring_boundary_layer
            _bnd_p2r = self.blender._scoring_boundary_pos_to_row
            if _bnd_h is not None:
                state.tokenwise_boundary_hidden = _bnd_h.clone()
            if _bnd_r is not None:
                state.tokenwise_boundary_residual = _bnd_r.clone()
            state.tokenwise_boundary_layer = _bnd_l
            if _bnd_p2r is not None:
                state.tokenwise_boundary_pos_to_row = _bnd_p2r.clone()

        logger.info(
            "[SAGE_CB_INC_FUSED] request=%s pre-TTFT complete: "
            "N=%d pre_ttft_ratio=%.4f pre_ttft_tokens=%d "
            "remaining_ranked=%d total_ranked=%d",
            req_id, N, pre_ttft_ratio, pre_ttft_count,
            int(remaining_ranking.numel()), int(full_ranking.numel()),
        )

    def _run_magnet_pre_ttft(
        self,
        req_id: str,
        blend_tokens: torch.Tensor,
        sage_token_mask: torch.Tensor,
        blend_slot_mapping: torch.Tensor,
        blend_block_table: Optional[torch.Tensor],
        kvcaches: list,
        sage_zero_copy: bool,
        state: Any,
        suffix_len: int = 0,
        chunk_boundaries: Optional[list[int]] = None,
    ) -> None:
        """Magnet pre-TTFT: ProphetKV Algorithm 1 (two-pass).

        Pass 1 (Stage I — lightweight query-only scoring): runs ONLY the
        suffix/query tokens through all L layers via
        magnet_query_only_score(). At each layer, computes
        α_l(t) = column_sum(softmax(Q_s · K'_context / √d)) using the
        cached context K'. Per-layer cost is O(|Q_s| × s) instead of
        O(s^2) of a full-context pass. Returns fused ᾱ(t); caller
        applies the threshold.

        Pass 2 (Stage II — targeted recompute): runs blend_from_gpu with
        magnet_preselected_indices set. process_qkv slices at layer 0 to
        the selected tokens and writes back K/V at every layer.

        Per-request state is initialized so decode-time hooks + retries
        know which tokens are already recomputed.
        """
        threshold = getattr(
            self._incremental_blend_strategy, "threshold", 0.01
        )
        N = int(blend_tokens.shape[0])
        ctx_len = N - suffix_len
        device = blend_tokens.device

        # Edge: if no suffix or no context, skip scoring entirely.
        if suffix_len <= 0 or ctx_len <= 0:
            logger.info(
                "[SAGE_MAGNET] request=%s pre-TTFT skip: suffix_len=%d ctx_len=%d",
                req_id, suffix_len, ctx_len,
            )
            if state is not None:
                state.decode_step = 0
                state.query_token_count = suffix_len
            self._sage_incremental_completed_requests.add(req_id)
            return

        # ── Pass 1: lightweight query-only scoring (ProphetKV Stage I) ──
        query_tokens = blend_tokens[ctx_len:].to(device)
        context_slot_mapping = blend_slot_mapping[:ctx_len]
        fused_alpha = self.blender.layerwise_model.magnet_query_only_score(
            query_token_ids=query_tokens,
            context_slot_mapping=context_slot_mapping,
            context_len=ctx_len,
            query_position_start=ctx_len,
            kvcaches=kvcaches,
            gpu_connector=self.blender.gpu_connector,
        )

        selected: Optional[torch.Tensor] = None
        # Apply candidate mask (first-chunk exclusion).
        if (
            sage_token_mask is not None
            and sage_token_mask.numel() >= ctx_len
        ):
            ctx_mask = sage_token_mask[:ctx_len].to(device)
            fused_alpha = fused_alpha.masked_fill(~ctx_mask, 0.0)

        # Log quantile distribution for threshold tuning.
        if fused_alpha.numel() > 0:
            _fs, _ = torch.sort(fused_alpha, descending=True)
            _n = _fs.numel()
            _qs = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80]
            _q_str = ", ".join(
                f"top{int(q*100)}%≥{float(_fs[int(q*_n)].item()):.4f}"
                for q in _qs
            )
            logger.info(
                "[MAGNET_DIST] req=%s n=%d mean=%.4f max=%.4f min=%.4f "
                "threshold=%.4f  %s",
                req_id, _n, float(fused_alpha.mean().item()),
                float(_fs[0].item()), float(_fs[-1].item()),
                threshold, _q_str,
            )
            if os.environ.get("SAGE_MAGNET_DUMP_ALPHA", "0") == "1":
                _dump_dir = os.environ.get(
                    "SAGE_MAGNET_DUMP_DIR", "/tmp/magnet_alpha",
                )
                os.makedirs(_dump_dir, exist_ok=True)
                torch.save({
                    "alpha": fused_alpha.detach().cpu(),
                    "req_id": req_id,
                    "context_len": _n,
                }, os.path.join(_dump_dir, f"{req_id}.pt"))

        # Selection mode: "threshold" (ProphetKV-adaptive, fraction
        # depends on context) or "topk" (fixed fraction, matches the
        # CacheBlend paper and ProphetKV original exactly — importance
        # still comes from magnet's fused_alpha).
        _sel_mode = os.environ.get(
            "SAGE_MAGNET_SELECTION_MODE", "threshold",
        ).strip().lower()
        if _sel_mode == "topk":
            _topk_ratio = float(
                os.environ.get("SAGE_MAGNET_TOPK_RATIO", "0.15")
            )
            _k = max(1, int(round(ctx_len * _topk_ratio)))
            _k = min(_k, int(fused_alpha.numel()))
            _topk_result = torch.topk(fused_alpha, k=_k)
            _topk_idx = _topk_result.indices
            _topk_vals = _topk_result.values
            context_selected, _ = torch.sort(_topk_idx)
            # Diagnostic: log top-10 selected indices and their α scores
            # so we can correlate with the needle position.
            _top10 = torch.topk(fused_alpha, k=min(10, int(fused_alpha.numel())))
            _top10_idx = _top10.indices.cpu().tolist()
            _top10_vals = [round(v, 4) for v in _top10.values.cpu().tolist()]
            logger.info(
                "[SAGE_MAGNET] selection_mode=topk k=%d ratio=%.4f "
                "top10_idx=%s top10_alpha=%s ctx_len=%d",
                _k, _topk_ratio, _top10_idx, _top10_vals, ctx_len,
            )
        else:
            context_selected = torch.nonzero(
                fused_alpha > threshold, as_tuple=False
            ).flatten()
            logger.info(
                "[SAGE_MAGNET] selection_mode=threshold p=%.4f "
                "selected=%d/%d",
                threshold, int(context_selected.numel()), ctx_len,
            )
        suffix_indices = torch.arange(
            ctx_len, N, device=device, dtype=torch.long,
        )
        selected = torch.cat([context_selected.to(torch.long), suffix_indices])
        selected = torch.unique(selected, sorted=True)

        selected_count = int(selected.numel()) if selected is not None else 0

        # ── Pass 2: targeted recompute using pre-selected indices ──
        if selected is not None and selected.numel() > 0:
            self.blender.metadata.magnet_preselected_indices = selected.to(
                device=device, dtype=torch.long,
            )
            try:
                self.blender.blend_from_gpu(
                    blend_tokens,
                    sage_token_mask,
                    step_recompute_ratio=None,
                    capture_full_ranking=False,
                    save_boundary=False,
                    suffix_len=suffix_len,
                    include_suffix=True,
                    kvcaches=kvcaches,
                    slot_mapping=blend_slot_mapping,
                    block_table=blend_block_table,
                    sage_zero_copy=sage_zero_copy,
                )
            finally:
                self.blender.metadata.magnet_preselected_indices = None

        if state is not None:
            state.cumulative_ratio = float(selected_count) / max(1, N)
            state.pre_ttft_recomputed_tokens = selected_count
            state.decode_step = 0
            state.query_token_count = suffix_len

        imp_mean = (
            float(fused_alpha.mean().item())
            if fused_alpha is not None and fused_alpha.numel() > 0 else 0.0
        )
        imp_max = (
            float(fused_alpha.max().item())
            if fused_alpha is not None and fused_alpha.numel() > 0 else 0.0
        )
        logger.info(
            "[SAGE_MAGNET] request=%s pre-TTFT complete (ProphetKV 2-pass): "
            "N=%d suffix=%d selected=%d threshold=%.6f "
            "mean_fused=%.6f max_fused=%.6f",
            req_id, N, suffix_len, selected_count,
            threshold, imp_mean, imp_max,
        )

        # Mark completed from the incremental-blend scheduler's perspective
        # so it doesn't re-enter start_load_kv. Decode-time recomputation
        # happens entirely via the hook + fused-inject pipeline, not via
        # scheduler-triggered blend passes.
        self._sage_incremental_completed_requests.add(req_id)

    def _run_layerwise_pre_ttft(
        self,
        req_id: str,
        blend_tokens: torch.Tensor,
        sage_token_mask: torch.Tensor,
        blend_slot_mapping: torch.Tensor,
        blend_block_table: Optional[torch.Tensor],
        kvcaches: list,
        sage_zero_copy: bool,
        state: Any,
        suffix_len: int = 0,
    ) -> None:
        """Layerwise incremental pre-TTFT: score + save boundary.

        1. Run layers [0, pre_ttft_layers) with diff_k scoring.
        2. Capture selected indices and boundary hidden_states.
        3. Store on state for decode-step layer-group injection.
        """
        N = int(blend_tokens.shape[0])
        num_layers = self.blender.num_layers
        pre_ttft_layers = max(
            2,  # minimum: layers 0-1 for diff_k scoring
            int(num_layers * self._layerwise_pre_ttft_ratio),
        )

        import time as _time
        _t_blend = _time.perf_counter()
        _all_layers_pre_ttft = (pre_ttft_layers >= num_layers)
        self.blender.blend_from_gpu(
            blend_tokens,
            sage_token_mask,
            capture_full_ranking=True,
            save_boundary=True,
            suffix_len=suffix_len,
            include_suffix=_all_layers_pre_ttft,
            kvcaches=kvcaches,
            slot_mapping=blend_slot_mapping,
            block_table=blend_block_table,
            sage_zero_copy=sage_zero_copy,
            max_layers_to_process=pre_ttft_layers,
        )
        logger.info(
            "[SAGE_TIMING] pre_ttft_blend elapsed=%.3fms layers=%d N=%d",
            (_time.perf_counter() - _t_blend) * 1000,
            pre_ttft_layers,
            int(blend_tokens.shape[0]),
        )

        # Capture selected indices and boundary state.
        imp_indices = self.blender._last_imp_indices
        if imp_indices is None or imp_indices.numel() == 0:
            logger.warning(
                "[SAGE_LAYERWISE] request=%s diff-k produced no selection",
                req_id,
            )
            return

        selected_indices = imp_indices.to(
            device=blend_tokens.device, dtype=torch.long
        ).flatten()
        selected_indices, _ = torch.sort(selected_indices)
        M = int(selected_indices.numel())

        # Read boundary state from blender (M tokens at pre_ttft_layers).
        out_hidden = self.blender._scoring_boundary_hidden
        out_residual = self.blender._scoring_boundary_residual
        _p2r = self.blender._scoring_boundary_pos_to_row
        if out_hidden is not None and _p2r is not None:
            _rows = _p2r[selected_indices]
            boundary_hidden = out_hidden[_rows].clone()
            boundary_residual = (
                out_residual[_rows].clone()
                if out_residual is not None else None
            )
        else:
            boundary_hidden = None
            boundary_residual = None

        # Update state for decode-step layer-wise scheduling.
        if state is not None:
            state.layerwise_selected_indices = selected_indices
            state.layerwise_saved_hidden = boundary_hidden
            state.layerwise_saved_residual = boundary_residual
            state.layerwise_diff_k_mean = getattr(
                self.blender, "_last_diff_k_mean", None
            )
            state.layerwise_current_layer = pre_ttft_layers
            state.decode_step = max(state.decode_step, 1)
            state.query_token_count = suffix_len

        logger.info(
            "[SAGE_LAYERWISE] request=%s pre-TTFT complete: "
            "N=%d M=%d pre_ttft_layers=%d/%d",
            req_id, N, M, pre_ttft_layers, num_layers,
        )

    def sage_warmup_blender(self, device: Any, kvcaches: list) -> None:
        """Pre-warm blender kernels with a tiny dummy blend.

        Runs blend_from_gpu over a single block worth of dummy tokens
        so that JIT compilation, cuBLAS handle creation, and the
        torch caching allocator are all warmed up before the first
        real request. The wall cost is paid once at engine init.
        """
        if not kvcaches or self.blender is None:
            return
        try:
            block_size = self._block_size
            N = block_size  # one block worth of tokens
            dummy_tokens = torch.zeros(N, dtype=torch.long, device=device)
            dummy_mask = torch.ones(N, dtype=torch.bool, device=device)
            # Slot mapping into block 0 — this block is unused / about
            # to be overwritten by real requests, so writing dummy KV
            # into it is harmless and resets to zero before first use.
            slot_mapping = torch.arange(N, dtype=torch.long, device=device)
            block_table = torch.tensor(
                [0], dtype=torch.long, device=device,
            )
            import time as _time
            _t0 = _time.perf_counter()
            try:
                self._run_layerwise_pre_ttft(
                    req_id="__sage_warmup__",
                    blend_tokens=dummy_tokens,
                    sage_token_mask=dummy_mask,
                    blend_slot_mapping=slot_mapping,
                    blend_block_table=block_table,
                    kvcaches=kvcaches,
                    sage_zero_copy=True,
                    state=None,
                    suffix_len=0,
                )
            except Exception as e:
                logger.warning(
                    "[SAGE_WARMUP] blender warmup failed (non-fatal): %s",
                    e,
                )
                return
            logger.info(
                "[SAGE_WARMUP] Blender pre-warmed in %.1fms",
                (_time.perf_counter() - _t0) * 1000,
            )
        finally:
            # Discard any state the dummy left behind on the blender so
            # the first real request gets a clean start.
            try:
                self.blender._last_imp_indices = None
                self.blender._scoring_boundary_hidden = None
                self.blender._scoring_boundary_residual = None
                self.blender._scoring_boundary_pos_to_row = None
                self.blender._scoring_boundary_layer = None
            except Exception:
                pass

    def sage_process_layer(
        self,
        parent_id: str,
        layer_idx: int,
        kvcaches: list,
        slot_mapping: torch.Tensor,
        token_ids: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        tokenwise_pre_ttft_ratio: Optional[float] = None,
    ) -> bool:
        """Process a single layer for pipelined recompute.

        Handles ALL layer types:
        - L0 (pre-check): N tokens, forward pass, no scoring
        - L1 (check layer): N tokens, diff-k scoring, reduce to M
        - L2+ (recompute): M tokens, forward pass, KV update

        Stateless w.r.t. the blender — uses per-request state only,
        so calls can be interleaved across requests.

        Returns True if processing ran, False if skipped.
        """
        state = self._get_or_create_sage_incremental_state(parent_id)
        if state is None:
            return False

        lmc_model = self.blender.layerwise_model
        vllm_model = lmc_model.vllm_model
        layers = vllm_model.model.layers
        if layer_idx >= len(layers):
            return False
        layer = layers[layer_idx]

        # Determine check layers and recompute ratio from blender config.
        # For tokenwise, use the pre-TTFT ratio at the check layer
        # (matching SG's _run_tokenwise_pre_ttft which passes
        # step_recompute_ratio=pre_ttft_ratio to blend_from_gpu).
        check_layers = self.blender.common_metadata.check_layers
        recomp_ratio = (
            self.blender.common_metadata.recomp_ratios[0]
            if self.blender.common_metadata.recomp_ratios
            else 0.15
        )
        if tokenwise_pre_ttft_ratio is not None and tokenwise_pre_ttft_ratio > 0:
            recomp_ratio = tokenwise_pre_ttft_ratio
        is_check_layer = layer_idx in check_layers
        is_pre_check = layer_idx < min(check_layers)

        # Get or initialize hidden states.
        hidden = state.layerwise_saved_hidden
        residual = state.layerwise_saved_residual

        if layer_idx == 0:
            # First layer: embed tokens.
            hidden = vllm_model.embed_input_ids(token_ids.cuda())
            residual = None
            # Positions for all N tokens.
            positions = torch.arange(
                token_ids.shape[0], device=hidden.device, dtype=torch.long,
            )
            state._pipelined_positions = positions
        elif hidden is None or residual is None:
            logger.warning(
                "[SAGE_PIPELINED] parent=%s L%d SKIP: hidden=%s residual=%s",
                parent_id, layer_idx,
                "None" if hidden is None else f"shape={list(hidden.shape)}",
                "None" if residual is None else f"shape={list(residual.shape)}",
            )
            return False
        else:
            positions = state._pipelined_positions

        # After check layer, only process M selected tokens.
        selected = state.layerwise_selected_indices
        if selected is not None and not is_pre_check and not is_check_layer:
            positions = selected.to(device=hidden.device, dtype=torch.long)

        num_kv_heads = lmc_model.vllm_attn_layers[layer_idx].num_kv_heads
        num_heads = lmc_model.vllm_attn_layers[layer_idx].num_heads
        head_size = lmc_model.vllm_attn_layers[layer_idx].head_size
        N = int(positions.shape[0])

        # Slot mapping for active tokens.
        active_slots = slot_mapping[positions]

        with torch.no_grad():
            # === LayerNorm ===
            if residual is None:
                residual = hidden
                hidden = layer.input_layernorm(hidden)
            else:
                hidden, residual = layer.input_layernorm(hidden, residual)

            # === QKV projection ===
            qkv, _ = layer.self_attn.qkv_proj(hidden)
            q, k, v = qkv.split(
                [
                    layer.self_attn.q_size,
                    layer.self_attn.kv_size,
                    layer.self_attn.kv_size,
                ],
                dim=-1,
            )
            q, k, v = lmc_model._process_qkv(q, k, v, layer)

            # === RoPE ===
            q, k = layer.self_attn.rotary_emb(positions, q, k)

            # === Diff-k scoring at check layer ===
            if is_check_layer:
                # Read old K from paged cache for comparison.
                kv_cache = kvcaches[layer_idx]
                _is_flash_ck = kv_cache.dim() >= 3 and kv_cache.shape[0] == 2
                if _is_flash_ck:
                    block_size = kv_cache.shape[2]
                    blk_idx = active_slots // block_size
                    blk_off = active_slots % block_size
                    old_k = kv_cache[0, blk_idx, blk_off].view(N, -1)
                elif kv_cache.dim() >= 4 and kv_cache.shape[1] == 2:
                    block_size = kv_cache.shape[2]
                    blk_idx = active_slots // block_size
                    blk_off = active_slots % block_size
                    old_k = kv_cache[blk_idx, 0, blk_off].view(N, -1)
                else:
                    block_size = kv_cache.shape[1]
                    blk_idx = active_slots // block_size
                    blk_off = active_slots % block_size
                    old_k = kv_cache[blk_idx, blk_off, :num_kv_heads * head_size]

                k_flat = k.view(N, -1)

                # Compute suffix exclusion first (needed by debug hash).
                _cached = self._sage_cached_prompt_inputs.get(parent_id, {})
                suffix_len = min(
                    int(_cached.get("query_token_count", 0) or 0),
                    N,
                )
                non_suffix_len = N - suffix_len

                _kf = k_flat[:non_suffix_len].float()
                _ok = old_k[:non_suffix_len].float()
                diff_k = torch.sum((_kf - _ok) ** 2, dim=[1])
                # Build candidate mask excluding first chunk (causal LM
                # optimization: first chunk has no left context, so its
                # prefill KV is already correct). Uses masking to match
                # SG's blend_from_gpu candidate_indices path exactly.
                chunk_bounds = self._sage_cached_prompt_inputs.get(
                    parent_id, {},
                ).get("chunk_boundaries")
                candidate_mask = torch.ones(
                    non_suffix_len, dtype=torch.bool, device=diff_k.device,
                )
                if chunk_bounds and len(chunk_bounds) > 1 and chunk_bounds[0] == 0:
                    first_chunk_end = min(chunk_bounds[1], non_suffix_len)
                    candidate_mask[:first_chunk_end] = False
                ns_candidates = torch.nonzero(
                    candidate_mask, as_tuple=False,
                ).flatten()

                # Select top-M from candidates only (matching blend path).
                topk_num = max(1, int(non_suffix_len * recomp_ratio))
                effective_len = int(ns_candidates.numel())
                topk_num = min(topk_num, effective_len)
                if ns_candidates.numel() > 0:
                    top_local = torch.topk(
                        diff_k[ns_candidates], k=topk_num,
                    ).indices
                    top_indices = ns_candidates[top_local]
                else:
                    top_indices = torch.topk(diff_k, k=topk_num).indices

                # Store full ranking for layerwise strategy (non-suffix only).
                if ns_candidates.numel() > 0:
                    score_view = diff_k[ns_candidates]
                    ranked_local = torch.argsort(score_view, descending=True)
                    full_ranking = ns_candidates[ranked_local]
                else:
                    full_ranking = torch.argsort(
                        diff_k[:non_suffix_len], descending=True,
                    )
                state.layerwise_full_importance_ranking = full_ranking

                # Append suffix when all layers are processed
                # pre-TTFT (ptt=1.0 or sync cacheblend). For incremental
                # (ptt<1.0), suffix is deferred to the final step.
                _strategy_env = os.environ.get(
                    "LMCACHE_BLEND_INCREMENTAL_STRATEGY", "layer_wise"
                )
                _all_layers_mg = (
                    self._layerwise_pre_ttft_ratio >= 1.0
                    or _strategy_env == "none"
                )
                if suffix_len > 0 and _all_layers_mg:
                    suffix_indices = torch.arange(
                        non_suffix_len, non_suffix_len + suffix_len,
                        device=top_indices.device,
                    )
                    top_indices = torch.cat([top_indices, suffix_indices])
                top_indices, _ = torch.sort(top_indices)

                # Reduce to M tokens.
                q = q[top_indices]
                k = k[top_indices]
                v = v[top_indices]
                residual = residual[top_indices]
                positions = positions[top_indices]
                active_slots = slot_mapping[positions]
                N = int(top_indices.shape[0])

                state.layerwise_selected_indices = top_indices
                state._pipelined_positions = positions
                state.layerwise_diff_k_mean = float(diff_k.mean().item())
                state.query_token_count = suffix_len
                # Mark scoring as done so blend_from_gpu skips it
                # when the parent enters the scheduler.
                # Use class-level dict so SCHEDULER adapter can see it
                # (MG has separate WORKER and SCHEDULER adapter instances).
                state.early_blend_done = True
                LMCacheConnectorV1Impl._shared_early_blend_done[parent_id] = True

            # === Write new K,V to paged cache ===
            # Pre-check layers (L0): skip write — blend_from_gpu writes
            # original data back (no-op). Check layer (L1): skip write —
            # the incremental decode path handles it. Recompute layers
            # (L2+): write all M positions directly.
            kv_cache = kvcaches[layer_idx]
            k_for_cache = k.view(-1, num_kv_heads, head_size)
            v_for_cache = v.view(-1, num_kv_heads, head_size)
            _is_flash = kv_cache.dim() >= 3 and kv_cache.shape[0] == 2
            if is_pre_check:
                pass  # Don't write L0 — matches blend_from_gpu (no-op write)
            elif _is_flash:
                # Flash: [2, blocks, block_size, H, D]
                block_size = kv_cache.shape[2]
                blk_idx = active_slots // block_size
                blk_off = active_slots % block_size
                kv_cache[0, blk_idx, blk_off] = k_for_cache
                kv_cache[1, blk_idx, blk_off] = v_for_cache
            elif kv_cache.dim() >= 4 and kv_cache.shape[1] == 2:
                # Triton: [blocks, 2, block_size, H, D]
                block_size = kv_cache.shape[2]
                blk_idx = active_slots // block_size
                blk_off = active_slots % block_size
                kv_cache[blk_idx, 0, blk_off] = k_for_cache
                kv_cache[blk_idx, 1, blk_off] = v_for_cache
            else:
                # FlashInfer: [blocks, block_size, H, D]
                block_size = kv_cache.shape[1]
                blk_idx = active_slots // block_size
                blk_off = active_slots % block_size
                kv_cache[blk_idx, blk_off] = torch.cat(
                    [k_for_cache, v_for_cache], dim=-1,
                )

            # === Attention ===
            # Pre-check layers (L0): use fresh K,V (N×N self-attention).
            # Check/recompute layers (L1+): M queries × N keys from paged
            # cache, matching blend_from_gpu's M×N attention.
            if is_pre_check:
                # N×N with fresh K,V — same as compute_layer
                q_attn = q.view(-1, num_heads, head_size)
                k_attn = k.view(-1, num_kv_heads, head_size)
                v_attn = v.view(-1, num_kv_heads, head_size)
                attn_metadata = lmc_model.lmc_attn_layers[layer_idx].init_attn_metadata(
                    input_ids=positions,
                )
                attn_output = torch.empty(
                    N, num_heads, head_size, dtype=q.dtype, device=q.device,
                )
                attn_output = lmc_model.lmc_attn_layers[layer_idx].forward_contiguous(
                    q_attn, k_attn, v_attn, attn_output, attn_metadata,
                )
            else:
                # M×N: read full context K,V from paged cache
                _total_N = int(token_ids.shape[0])
                _all_slots = slot_mapping[:_total_N]
                kv_cache_attn = kvcaches[layer_idx]
                _is_flash_a = kv_cache_attn.dim() >= 3 and kv_cache_attn.shape[0] == 2
                if _is_flash_a:
                    _bs = kv_cache_attn.shape[2]
                    _bi = _all_slots // _bs
                    _bo = _all_slots % _bs
                    full_k = kv_cache_attn[0, _bi, _bo].view(_total_N, num_kv_heads, head_size)
                    full_v = kv_cache_attn[1, _bi, _bo].view(_total_N, num_kv_heads, head_size)
                elif kv_cache_attn.dim() >= 4 and kv_cache_attn.shape[1] == 2:
                    _bs = kv_cache_attn.shape[2]
                    _bi = _all_slots // _bs
                    _bo = _all_slots % _bs
                    full_k = kv_cache_attn[_bi, 0, _bo].view(_total_N, num_kv_heads, head_size)
                    full_v = kv_cache_attn[_bi, 1, _bo].view(_total_N, num_kv_heads, head_size)
                else:
                    _bs = kv_cache_attn.shape[1]
                    _bi = _all_slots // _bs
                    _bo = _all_slots % _bs
                    _kv = kv_cache_attn[_bi, _bo]
                    full_k = _kv[..., :num_kv_heads * head_size].view(_total_N, num_kv_heads, head_size)
                    full_v = _kv[..., num_kv_heads * head_size:].view(_total_N, num_kv_heads, head_size)

                q = q.view(-1, num_heads, head_size)
                attn_metadata = lmc_model.lmc_attn_layers[layer_idx].init_attn_metadata(
                    input_ids=torch.arange(_total_N, device=q.device),
                    num_queries=N,
                )
                attn_output = torch.empty(
                    N, num_heads, head_size, dtype=q.dtype, device=q.device,
                )
                attn_output = lmc_model.lmc_attn_layers[layer_idx].forward_contiguous(
                    q, full_k, full_v, attn_output, attn_metadata,
                )

            # === O projection + FFN ===
            attn_output = attn_output.view(-1, num_heads * head_size)
            hidden, _ = layer.self_attn.o_proj(attn_output)
            if hasattr(layer, "pre_feedforward_layernorm"):
                # Gemma 3: 4-norm structure
                hidden = layer.post_attention_layernorm(hidden)
                hidden, residual = layer.pre_feedforward_layernorm(
                    hidden, residual,
                )
                hidden = layer.mlp(hidden)
                hidden = layer.post_feedforward_layernorm(hidden)
            else:
                # Llama/Qwen: 2-norm structure
                hidden, residual = layer.post_attention_layernorm(
                    hidden, residual,
                )
                hidden = layer.mlp(hidden)

        # Save state for next layer.
        state.layerwise_saved_hidden = hidden.detach()
        state.layerwise_saved_residual = residual.detach()
        state.layerwise_current_layer = layer_idx + 1

        return True

    def _get_or_create_cached_prompt_inputs(
        self,
        req_id: str,
        tokens: Union[torch.Tensor, list[int]],
        prompt_len: int,
        slot_mapping: torch.Tensor,
    ) -> dict[str, Any]:
        cached = self._sage_cached_prompt_inputs.get(req_id)
        if cached is not None and int(cached["prompt_len"]) == int(prompt_len):
            return cached
        if isinstance(tokens, torch.Tensor):
            prompt_tokens = tokens[:prompt_len]
            if prompt_tokens.device != self.device or prompt_tokens.dtype != torch.long:
                prompt_tokens = prompt_tokens.to(device=self.device, dtype=torch.long)
        else:
            prompt_tokens = torch.tensor(
                tokens[:prompt_len],
                device=self.device,
                dtype=torch.long,
            )
        prompt_slot_mapping = slot_mapping[:prompt_len]
        if prompt_slot_mapping.device != self.device:
            prompt_slot_mapping = prompt_slot_mapping.to(device=self.device)
        prompt_slot_mapping = prompt_slot_mapping.clone()
        cached_prompt = {
            "prompt_tokens": prompt_tokens,
            "prompt_slot_mapping": prompt_slot_mapping,
            "full_token_mask": torch.ones(
                int(prompt_len),
                dtype=torch.bool,
                device=self.device,
            ),
            "prompt_len": int(prompt_len),
        }
        self._sage_cached_prompt_inputs[req_id] = cached_prompt
        return cached_prompt

    # TODO(chunxiaozheng): in the latest lmcache_connector, we use `register_kv_caches`
    #  to init self.kv_caches, we keep it in order to be compatible with old versions
    #  and will be removed in the future.
    @_lmcache_nvtx_annotate
    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug("The layer %s does not have kv_cache, skip it", layer_name)
                continue

            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[
                    forward_context.virtual_engine
                ]

        self._build_kv_layer_groups()

    ####################
    # Worker side APIs
    ####################
    @_lmcache_nvtx_annotate
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        logger.info("Registering KV caches")
        # TODO(chunxiaozheng): `_init_kv_caches_from_forward_context` is
        #  not called, we should consider removing it.
        assert len(self.kv_caches) == 0 and len(kv_caches) > 0
        self.kv_caches = kv_caches
        self._build_kv_layer_groups()
        self._manager.post_init()

    @_lmcache_nvtx_annotate
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """Start loading the KV cache from the connector buffer to vLLM's
        paged KV buffer.

        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation

        Note:
            The number of elements in kv_caches and layer_names should be
            the same.
        """
        self.current_layer = 0

        if len(self.kv_caches) == 0:
            logger.warning(
                "Please update LMCacheConnector, "
                "use register_kv_caches to init kv_caches"
            )
            self._init_kv_caches_from_forward_context(forward_context)

        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        # Helper: extract block_table tensor from forward_context.attn_metadata,
        # which is a dict[str, AttentionMetadata] (PerLayerAttnMetadata), not a
        # single AttentionMetadata.  getattr(dict, "block_table") always returns
        # None; we need to pull the value from one of the per-layer entries.
        def _extract_block_table_from_attn_metadata(
            attn_meta: Any,
        ) -> Optional[torch.Tensor]:
            if attn_meta is None:
                return None
            if isinstance(attn_meta, dict):
                single = next(iter(attn_meta.values()), None)
            elif isinstance(attn_meta, list):
                single = None
                for _d in attn_meta:
                    if isinstance(_d, dict) and _d:
                        single = next(iter(_d.values()), None)
                        break
            else:
                single = attn_meta
            return getattr(single, "block_table", None)

        assert self.lmcache_engine is not None

        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.debug("In connector.start_load_kv, but the attn_metadata is None")
            return
        self.layerwise_retrievers = []

        last_idx = -1
        for idx, request in enumerate(metadata.requests):
            if request.load_spec is not None:
                continue
            last_idx = idx

        for idx, request in enumerate(metadata.requests):
            if request.load_spec is None:
                continue

            tokens = request.token_ids
            slot_mapping = request.slot_mapping.to(self.device)
            assert len(tokens) == len(slot_mapping), f"tokens={len(tokens)}, slot_mapping={len(slot_mapping)}"

            # Keep mask on GPU when blending is enabled to avoid CPU->GPU copies
            # and satisfy strict device checks in blender.process_qkv.
            mask_device = self.device if self.enable_blending else "cpu"
            token_mask = torch.ones(len(tokens), dtype=torch.bool, device=mask_device)
            masked_token_count = (
                request.load_spec.vllm_cached_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )
            token_mask[:masked_token_count] = False

            lmcache_cached_tokens = request.load_spec.lmcache_cached_tokens
            if self.use_layerwise:
                if idx == last_idx:
                    sync = True
                else:
                    sync = False
                # NOTE(Jiayi): Perform blending before layerwise prefix caching
                if self.enable_blending:
                    # TODO(Jiayi): Need to make prefix caching and blending compatible
                    sage_enable_timing = os.getenv("SAGE_ENABLE_TIMING", "0").lower() == "1"
                    if sage_enable_timing:
                        start_time = time.time()
                        logger.info(
                            "Start blending %d tokens for request %s at time %.3f",
                            lmcache_cached_tokens,
                            request.req_id,
                            start_time,
                        )

                    # SAGE ZERO-COPY: For requests with transferred blocks, use
                    # GPU-direct blending since KV is already in GPU paged memory.
                    use_gpu_blend = (
                        request.sage_blocks_transferred and self.enable_sage
                    )

                    if use_gpu_blend:
                        logger.debug(
                            "[SAGE_BLEND] Using GPU-direct blending for request %s "
                            "(sage_blocks_transferred=%s)",
                            request.req_id,
                            request.sage_blocks_transferred,
                        )
                        state: Optional[IncrementalBlendState] = None
                        step_recompute_ratio: Optional[float] = None
                        precomputed_imp_indices: Optional[torch.Tensor] = None
                        should_blend_this_step = True
                        # RoPE prepass + virtual slot mapping — runs for ALL
                        # SAGE methods (sync CacheBlend, tokenwise, layerwise)
                        # before any method-specific blending.
                        _chunk_bounds = getattr(
                            request, "sage_chunk_boundaries", None
                        )
                        _already_adj = request.req_id in self._sage_rope_adjusted_requests
                        if not _already_adj:
                            assert _chunk_bounds is not None and len(_chunk_bounds) >= 2, (
                                f"SAGE request {request.req_id} missing chunk_boundaries"
                            )
                            # Use the full allocated block list (may exceed
                            # ceil(N/block_size) when SAGE zero-copy chunks
                            # contribute partial-last-block extras). Falling
                            # back to slot_mapping-striding would drop the
                            # virtual-only blocks and break the remap math.
                            _req_block_ids = getattr(
                                request, "allocated_block_ids", None
                            )
                            # Contiguous-order block list (slot_mapping-derived):
                            # may MISS virtual-only blocks (partial-last-block
                            # extras from SAGE chunked prefill).
                            _rope_block_table_from_sm = (
                                slot_mapping[:lmcache_cached_tokens:self._block_size]
                                // self._block_size
                            ).clone().to(dtype=torch.long)
                            if _req_block_ids is not None:
                                _rope_block_table = _req_block_ids.to(
                                    device=self.device, dtype=torch.long,
                                ).contiguous()
                            else:
                                _rope_block_table = _rope_block_table_from_sm
                            # If pipelined recompute already did partial
                            # RoPE+remap, skip those layers.
                            _partial_done = None
                            _state_check = self._sage_incremental_states.get(
                                request.req_id
                            )
                            if _state_check is not None:
                                _partial_done = getattr(
                                    _state_check, "layerwise_current_layer", 0
                                ) or None
                            _layer_range = None
                            if _partial_done is not None and _partial_done > 0:
                                if _partial_done >= len(kvcaches):
                                    # All layers already processed.
                                    logger.info(
                                        "[SAGE_TIMING] rope_prepass skipped "
                                        "(all %d layers done by pipelined recompute)",
                                        len(kvcaches),
                                    )
                                    self._sage_rope_adjusted_requests.add(
                                        request.req_id
                                    )
                                    # Skip to blend logic below.
                                    _already_adj = True
                                else:
                                    _layer_range = (_partial_done, len(kvcaches))
                                    logger.info(
                                        "[SAGE_TIMING] rope_prepass skipping "
                                        "L0..%d, processing L%d..%d",
                                        _partial_done - 1, _partial_done,
                                        len(kvcaches) - 1,
                                    )
                            if not _already_adj:
                                self._run_rope_prepass_and_remap(
                                    request.req_id,
                                    chunk_boundaries=_chunk_bounds,
                                    slot_mapping=slot_mapping,
                                    num_tokens=int(lmcache_cached_tokens),
                                    rope_block_table=_rope_block_table,
                                    kvcaches=kvcaches,
                                    layer_range=_layer_range,
                                )
                                # If we did a partial pass (remaining layers
                                # after early-blend), mark as fully adjusted.
                                if _layer_range is not None:
                                    self._sage_rope_adjusted_requests.add(
                                        request.req_id
                                    )
                        strategy = getattr(self, "_incremental_blend_strategy", None)
                        if (
                            strategy is not None
                            and should_blend_this_step
                        ):
                            num_output_tokens = int(
                                getattr(request, "num_output_tokens", 0)
                            )
                            is_pre_ttft_step = num_output_tokens <= 0
                        if strategy is not None and should_blend_this_step:
                            state = self._get_or_create_sage_incremental_state(
                                request.req_id
                            )
                            assert state is not None
                            incremental_active = (
                                request.req_id
                                not in self._sage_incremental_completed_requests
                            )
                            if not incremental_active:
                                should_blend_this_step = False
                            logger.info(
                                "[SAGE_PATH] request=%s method=%s "
                                "is_pre_ttft=%s incremental_active=%s "
                                "decode_step=%d should_blend=%s",
                                request.req_id,
                                self._sage_method.value,
                                is_pre_ttft_step,
                                incremental_active,
                                state.decode_step,
                                should_blend_this_step,
                            )

                            # ── Tokenwise MG: wire ranking from sage_process_layer ──
                            # sage_process_layer (MG drain) may have scored + recomputed
                            # all layers. Detect this via early_blend_done and set up
                            # the frozen ranking for decode-step scheduling. This check
                            # runs regardless of is_pre_ttft_step because in MG the
                            # suffix prefill may have already incremented num_output_tokens.
                            _eb_tw_early = (
                                self._sage_method == SageMethod.CACHEBLEND_TOKENWISE
                                and getattr(state, "early_blend_done", False)
                                and getattr(state, "frozen_importance_ranking", None) is None
                                and should_blend_this_step
                            )
                            if _eb_tw_early:
                                _full_rank = getattr(state, "layerwise_full_importance_ranking", None)
                                if _full_rank is not None and _full_rank.numel() > 0:
                                    _tw_ptt = self._tokenwise_pre_ttft_ratio
                                    N_tw = int(lmcache_cached_tokens)
                                    _pre_ttft_count = max(1, int(N_tw * _tw_ptt))
                                    _remaining = _full_rank[_pre_ttft_count:]
                                    if _remaining.numel() > 0:
                                        state.frozen_importance_ranking = _remaining
                                        state.frozen_ratio_cursor = 0.0
                                        state.cumulative_ratio = float(_tw_ptt)
                                        state.pre_ttft_recomputed_tokens = _pre_ttft_count
                                        state.decode_step = max(state.decode_step, 1)
                                        state._tw_cursor = 0
                                        state._tw_tokens_emitted = 0
                                        state._tw_prev_ratio = float(_tw_ptt)
                                    else:
                                        self._mark_sage_recompute_completed(
                                            request.req_id, state, _tw_ptt,
                                        )
                                        should_blend_this_step = False
                                    logger.info(
                                        "[SAGE_PATH] request=%s → tokenwise_early_blend "
                                        "ranking=%d remaining=%d ptt=%.2f",
                                        request.req_id,
                                        int(_full_rank.numel()),
                                        int(_remaining.numel()) if _remaining.numel() > 0 else 0,
                                        _tw_ptt,
                                    )
                                    # Hash pre-TTFT state for tokenwise (MG)
                            # ── Layer-wise incremental scheduling ──
                            # Deferred diff-k: run scoring in decode step 1 instead of pre-TTFT.
                            if not should_blend_this_step:
                                pass  # Hash handled by universal check below
                            elif (
                                self._sage_method == SageMethod.CACHEBLEND_LAYERWISE
                                and self._layerwise_defer_diff_k
                                and state.layerwise_selected_indices is None
                                and state.layerwise_deferred_blend_params is not None
                            ):
                                params = state.layerwise_deferred_blend_params
                                state.layerwise_deferred_blend_params = None
                                logger.info(
                                    "[SAGE_PATH] request=%s → SCHED:layerwise_deferred_scoring",
                                    request.req_id,
                                )
                                self._run_layerwise_pre_ttft(
                                    req_id=request.req_id,
                                    blend_tokens=params["blend_tokens"],
                                    sage_token_mask=params["sage_token_mask"],
                                    blend_slot_mapping=params["blend_slot_mapping"],
                                    blend_block_table=params["blend_block_table"],
                                    kvcaches=params["kvcaches"],
                                    sage_zero_copy=params["sage_zero_copy"],
                                    state=state,
                                    suffix_len=request.query_token_count,
                                )
                                # Immediately set up first injection in the same
                                # step (same fix as the immediate pre-TTFT path).
                                if (
                                    state.layerwise_selected_indices is not None
                                    and state.layerwise_saved_hidden is not None
                                ):
                                    _first_step = strategy.next_step(
                                        state, self.blender.num_layers
                                    )
                                    if _first_step.should_blend and _first_step.layer_range is not None:
                                        _lw_s, _lw_e = _first_step.layer_range
                                        _sel = state.layerwise_selected_indices
                                        _cp = self._get_or_create_cached_prompt_inputs(
                                            req_id=request.req_id,
                                            tokens=tokens,
                                            prompt_len=int(lmcache_cached_tokens),
                                            slot_mapping=slot_mapping,
                                        )
                                        self._fused_inject_pending[request.req_id] = {
                                            "token_ids": _cp["prompt_tokens"][_sel],
                                            "positions": _sel.clone(),
                                            "slots": _cp["prompt_slot_mapping"][_sel],
                                            "hidden_states": state.layerwise_saved_hidden,
                                            "residual": state.layerwise_saved_residual,
                                            "inject_layer": _lw_s,
                                            "extract_layer": _lw_e,
                                        }
                                        self._sage_pending_blend_launch_t0[
                                            request.req_id
                                        ] = time.perf_counter()
                                        logger.info(
                                            "[SAGE_LAYERWISE] request=%s "
                                            "deferred_scoring+inject layers=[%d,%d) M=%d",
                                            request.req_id, _lw_s, _lw_e,
                                            int(_sel.numel()),
                                        )
                                should_blend_this_step = False
                            elif (
                                self._sage_method == SageMethod.CACHEBLEND_LAYERWISE
                                and incremental_active
                                and state.layerwise_selected_indices is not None
                            ):
                                step = strategy.next_step(state, self.blender.num_layers)
                                if step.should_blend and step.layer_range is not None:
                                    logger.info(
                                        "[SAGE_PATH] request=%s → SCHED:layerwise_decode_inject",
                                        request.req_id,
                                    )
                                    lw_start, lw_end = step.layer_range
                                    selected_indices = state.layerwise_selected_indices
                                    saved_hidden = state.layerwise_saved_hidden
                                    saved_residual = state.layerwise_saved_residual
                                    if saved_hidden is None or saved_residual is None:
                                        raise RuntimeError(
                                            f"Layerwise decode step {state.decode_step}: "
                                            f"missing saved hidden/residual at layer {lw_start}"
                                        )
                                    cached_prompt = self._get_or_create_cached_prompt_inputs(
                                        req_id=request.req_id,
                                        tokens=tokens,
                                        prompt_len=int(lmcache_cached_tokens),
                                        slot_mapping=slot_mapping,
                                    )
                                    prompt_tokens = cached_prompt["prompt_tokens"]
                                    prompt_slot_mapping = cached_prompt["prompt_slot_mapping"]

                                    # Stage layerwise injection payload — same approach
                                    # as tokenwise but with extract_layer to remove M
                                    # replay tokens after the target layer range.
                                    self._fused_inject_pending[request.req_id] = {
                                        "token_ids": prompt_tokens[selected_indices],
                                        "positions": selected_indices.clone(),
                                        "slots": prompt_slot_mapping[selected_indices],
                                        "hidden_states": saved_hidden,
                                        "residual": saved_residual,
                                        "inject_layer": lw_start,
                                        "extract_layer": lw_end,
                                    }
                                    self._sage_pending_blend_launch_t0[
                                        request.req_id
                                    ] = time.perf_counter()
                                    should_blend_this_step = False
                                    logger.info(
                                        "[SAGE_LAYERWISE] request=%s decode_step=%d "
                                        "staged injection layers=[%d,%d) M=%d "
                                        "num_output_tokens=%d lmcache_cached=%d",
                                        request.req_id,
                                        state.decode_step,
                                        lw_start,
                                        lw_end,
                                        int(selected_indices.numel()),
                                        request.num_output_tokens,
                                        int(lmcache_cached_tokens),
                                    )
                                elif not step.should_blend:
                                    should_blend_this_step = False
                                    # Layer-wise budget exhausted.
                                    self._sage_incremental_completed_requests.add(
                                        request.req_id
                                    )
                                    state.layerwise_selected_indices = None
                                    state.layerwise_saved_hidden = None
                                    state.layerwise_saved_residual = None
                                    logger.info(
                                        "[SAGE_PATH] request=%s → SCHED:layerwise_completed "
                                        "layer=%d/%d",
                                        request.req_id,
                                        state.layerwise_current_layer,
                                        self.blender.num_layers,
                                    )
                                    # Re-recompute suffix after all
                                    # incremental layers are done. The
                                    # suffix was prefilled pre-TTFT with
                                    # stale KV at unrecomputed layers.
                                    # Skip if suffix was already included
                                    # in the pre-TTFT recompute (ptt=1.0).
                                    _qtc_raw = getattr(
                                        state, "query_token_count",
                                        getattr(
                                            request, "query_token_count", 0
                                        ),
                                    )
                                    _suffix_len_final = min(
                                        int(_qtc_raw or 0),
                                        int(lmcache_cached_tokens),
                                    )
                                    _pre_ttft_layers = max(
                                        2,
                                        int(
                                            self.blender.num_layers
                                            * self._layerwise_pre_ttft_ratio
                                        ),
                                    )
                                    _already_included = (
                                        _pre_ttft_layers
                                        >= self.blender.num_layers
                                    )
                                    logger.info(
                                        "[SAGE_SUFFIX_CHECK] req=%s "
                                        "qtc_raw=%s suffix_len=%d "
                                        "pre_ttft_layers=%d "
                                        "already_included=%s",
                                        request.req_id,
                                        _qtc_raw,
                                        _suffix_len_final,
                                        _pre_ttft_layers,
                                        _already_included,
                                    )
                                    if (
                                        _suffix_len_final > 0
                                        and not _already_included
                                    ):
                                        _non_suffix_final = int(
                                            lmcache_cached_tokens
                                        ) - _suffix_len_final
                                        _suffix_pos = torch.arange(
                                            _non_suffix_final,
                                            int(lmcache_cached_tokens),
                                            device=self.device,
                                            dtype=torch.long,
                                        )
                                        _cp_final = (
                                            self._get_or_create_cached_prompt_inputs(
                                                req_id=request.req_id,
                                                tokens=tokens,
                                                prompt_len=int(
                                                    lmcache_cached_tokens
                                                ),
                                                slot_mapping=slot_mapping,
                                            )
                                        )
                                        _suffix_tids = _cp_final[
                                            "prompt_tokens"
                                        ][_suffix_pos]
                                        _suffix_emb = (
                                            self.blender.layerwise_model
                                            .vllm_model.embed_input_ids(
                                                _suffix_tids.to(self.device)
                                            )
                                        )
                                        self._fused_inject_pending[
                                            request.req_id
                                        ] = {
                                            "token_ids": _suffix_tids,
                                            "positions": _suffix_pos.clone(),
                                            "slots": _cp_final[
                                                "prompt_slot_mapping"
                                            ][_suffix_pos],
                                            "hidden_states": _suffix_emb,
                                            "residual": None,
                                            "inject_layer": 0,
                                            "extract_layer": (
                                                self.blender.num_layers
                                            ),
                                        }
                                        self._sage_suffix_rerecompute_pending.add(
                                            request.req_id
                                        )
                                        logger.info(
                                            "[SAGE_SUFFIX_RERECOMPUTE] "
                                            "req=%s suffix=%d inject=[0,%d)",
                                            request.req_id,
                                            _suffix_len_final,
                                            self.blender.num_layers,
                                        )
                            elif (
                                self._sage_method == SageMethod.CACHEBLEND_TOKENWISE
                                and not is_pre_ttft_step
                            ):
                                if state.frozen_importance_ranking is None:
                                    if request.req_id in self._sage_suffix_rerecompute_pending:
                                        # Suffix rerecompute is in flight — don't
                                        # re-enter tokenwise scheduling. The fused
                                        # inject lifecycle will handle completion.
                                        should_blend_this_step = False
                                    elif (
                                        request.req_id
                                        in self._sage_incremental_completed_requests
                                    ):
                                        # Already completed (suffix rerecomp done).
                                        should_blend_this_step = False
                                    else:
                                        # Scoring not done — should not happen
                                        # (pre-TTFT always scores).
                                        should_blend_this_step = False
                                        self._sage_incremental_completed_requests.add(
                                            request.req_id
                                        )
                                        logger.error(
                                            "[SAGE_PATH] request=%s → SCHED:tokenwise_missing_ranking "
                                            "(BUG: pre-TTFT scoring should have set this)",
                                            request.req_id,
                                        )
                                else:
                                    step = strategy.next_step(state, self.blender.num_layers)
                                    if not step.should_blend:
                                        # Budget exhausted — check if suffix
                                        # needs rerecompute (same as layerwise).
                                        state.frozen_importance_ranking = None
                                        _suffix_n_tw = int(getattr(
                                            state, "query_token_count",
                                            getattr(request, "query_token_count", 0),
                                        ) or 0)
                                        _tw_ptt_check = self._tokenwise_pre_ttft_ratio
                                        _tw_suffix_already = (_tw_ptt_check >= 1.0)
                                        if (
                                            _suffix_n_tw > 0
                                            and not _tw_suffix_already
                                            and request.req_id
                                            not in self._sage_suffix_rerecompute_pending
                                        ):
                                            _N_tw_final = int(lmcache_cached_tokens)
                                            _non_suffix_tw = _N_tw_final - _suffix_n_tw
                                            _suffix_pos_tw = torch.arange(
                                                _non_suffix_tw, _N_tw_final,
                                                device=self.device, dtype=torch.long,
                                            )
                                            _cp_tw = self._get_or_create_cached_prompt_inputs(
                                                req_id=request.req_id,
                                                tokens=tokens,
                                                prompt_len=_N_tw_final,
                                                slot_mapping=slot_mapping,
                                            )
                                            _suffix_tids_tw = _cp_tw["prompt_tokens"][_suffix_pos_tw]
                                            _suffix_emb_tw = (
                                                self.blender.layerwise_model
                                                .vllm_model.embed_input_ids(
                                                    _suffix_tids_tw.to(self.device)
                                                )
                                            )
                                            self._fused_inject_pending[request.req_id] = {
                                                "token_ids": _suffix_tids_tw,
                                                "positions": _suffix_pos_tw.clone(),
                                                "slots": _cp_tw["prompt_slot_mapping"][_suffix_pos_tw],
                                                "hidden_states": _suffix_emb_tw,
                                                "residual": None,
                                                "inject_layer": 0,
                                                "extract_layer": self.blender.num_layers,
                                            }
                                            self._sage_suffix_rerecompute_pending.add(request.req_id)
                                            should_blend_this_step = True
                                            logger.info(
                                                "[SAGE_SUFFIX_RERECOMPUTE] req=%s "
                                                "suffix=%d inject=[0,%d) (tokenwise)",
                                                request.req_id, _suffix_n_tw,
                                                self.blender.num_layers,
                                            )
                                        else:
                                            should_blend_this_step = False
                                            self._sage_incremental_completed_requests.add(
                                                request.req_id
                                            )
                                        logger.info(
                                            "[SAGE_PATH] request=%s → SCHED:tokenwise_completed "
                                            "suffix_rerecomp=%s",
                                            request.req_id,
                                            request.req_id in self._sage_suffix_rerecompute_pending,
                                        )
                                    else:
                                        logger.info(
                                            "[SAGE_PATH] request=%s → SCHED:tokenwise_decode_slice "
                                            "decode_step=%d",
                                            request.req_id,
                                            state.decode_step,
                                        )
                                        ranking = state.frozen_importance_ranking
                                        total_ranked = int(ranking.numel())
                                        N = int(max(1, lmcache_cached_tokens))

                                        # Fixed tokens per step: use the step
                                        # delta from the strategy (default 5% of N).
                                        _prev_ratio = getattr(state, "_tw_prev_ratio", 0.0)
                                        _step_delta = step.recompute_ratio - _prev_ratio
                                        _tokens_per_step = max(1, int(_step_delta * N))
                                        state._tw_prev_ratio = step.recompute_ratio
                                        # Use cursor to track position in ranking
                                        start_idx = int(getattr(state, "_tw_cursor", 0))
                                        end_idx = min(total_ranked, start_idx + _tokens_per_step)

                                        if end_idx <= start_idx:
                                            should_blend_this_step = False
                                        else:
                                            precomputed_imp_indices = ranking[start_idx:end_idx]
                                            should_blend_this_step = True
                                            state._tw_cursor = end_idx
                                            state._tw_tokens_emitted = int(
                                                getattr(state, "_tw_tokens_emitted", 0)
                                            ) + (end_idx - start_idx)
                                            step_recompute_ratio = (
                                                float(end_idx) / float(max(1, total_ranked))
                                            )
                                            state.cumulative_ratio = max(
                                                state.cumulative_ratio,
                                                float(
                                                    int(getattr(state, "pre_ttft_recomputed_tokens", 0))
                                                    + end_idx
                                                ) / float(N),
                                            )
                                            logger.info(
                                                "[SAGE_INCREMENTAL] request=%s tokenwise slice "
                                                "[%d:%d) of %d M=%d decode_step=%d",
                                                request.req_id,
                                                start_idx, end_idx, total_ranked,
                                                end_idx - start_idx,
                                                state.decode_step,
                                            )
                        if should_blend_this_step:
                            # Get chunk boundaries to exclude first chunk from candidates.
                            chunk_boundaries = getattr(
                                request, "sage_chunk_boundaries", None
                            )

                            # For SAGE zero-copy requests, all prompt KV is already in GPU memory.
                            # Build a recompute candidate mask for blending:
                            # - default: all True
                            # - optimization: exclude first chunk (it has no left context,
                            #   so its standalone prefill KV is already correct for causal LMs).
                            sage_token_mask = torch.ones(
                                lmcache_cached_tokens, dtype=torch.bool, device=self.device
                            )
                            if (
                                chunk_boundaries is not None
                                and len(chunk_boundaries) > 1
                                and chunk_boundaries[0] == 0
                            ):
                                first_chunk_end = chunk_boundaries[1]
                                if 0 < first_chunk_end <= lmcache_cached_tokens:
                                    sage_token_mask[:first_chunk_end] = False

                            blend_tokens = torch.tensor(
                                tokens[:lmcache_cached_tokens],
                                dtype=torch.long,
                                device=self.device,
                            )
                            blend_token_mask = sage_token_mask
                            # After RoPE prepass + remap, all KV is at
                            # contiguous slots. No virtual/contiguous split.
                            blend_slot_mapping = slot_mapping[:lmcache_cached_tokens]
                            # Build the per-request block_table directly
                            # from the request's slot_mapping. Each block
                            # contributes block_size consecutive slots,
                            # so the block_id at position i*block_size in
                            # slot_mapping is `slot // block_size`. Using
                            # `attn_metadata.block_table[idx]` is wrong
                            # here because `idx` is the LMCache loop
                            # index (per metadata.requests), not the
                            # input_batch row index — they may differ
                            # when multiple requests share an engine.
                            blend_block_table = (
                                slot_mapping[:lmcache_cached_tokens:self._block_size]
                                // self._block_size
                            ).clone().to(dtype=torch.long)
                            # Stage-II token-wise optimization: run blending on selected
                            # token slice only (instead of full prompt range).
                            has_precomputed_indices = (
                                precomputed_imp_indices is not None
                                and float(step_recompute_ratio or 0.0) > 0.0
                            )
                            if has_precomputed_indices:
                                logger.info(
                                    "[SAGE_PATH] request=%s → EXEC:tokenwise_fused_inject "
                                    "M=%d ratio=%.4f",
                                    request.req_id,
                                    int(precomputed_imp_indices.numel()),
                                    float(step_recompute_ratio or 0),
                                )
                                cached_prompt = self._get_or_create_cached_prompt_inputs(
                                    req_id=request.req_id,
                                    tokens=tokens,
                                    prompt_len=int(lmcache_cached_tokens),
                                    slot_mapping=slot_mapping,
                                )
                                prompt_tokens = cached_prompt["prompt_tokens"]
                                prompt_slot_mapping = cached_prompt["prompt_slot_mapping"]

                                # Tokenwise injection: each decode-step
                                # batch processes DIFFERENT tokens through
                                # ALL layers (L0-L35). Start from embeddings
                                # at L0 (these tokens haven't been processed
                                # at any layer yet). Extract at the last
                                # layer so the inject rows are removed after
                                # the forward completes.
                                if self._sage_method == SageMethod.CACHEBLEND_TOKENWISE:
                                    _tw_tids = prompt_tokens[precomputed_imp_indices]
                                    _tw_emb = (
                                        self.blender.layerwise_model
                                        .vllm_model.embed_input_ids(
                                            _tw_tids.to(self.device)
                                        )
                                    )
                                    payload = {
                                        "token_ids": _tw_tids,
                                        "positions": precomputed_imp_indices.clone(),
                                        "slots": prompt_slot_mapping[precomputed_imp_indices],
                                        "hidden_states": _tw_emb,
                                        "residual": None,
                                        "inject_layer": 0,
                                        "extract_layer": self.blender.num_layers,
                                    }
                                    self._fused_inject_pending[request.req_id] = payload
                                    self._sage_pending_blend_launch_t0[request.req_id] = time.perf_counter()
                            if request.req_id not in self._fused_inject_pending:
                                # If sage_process_layer already scored (MG path),
                                # skip pre-TTFT entirely — scoring + indices are
                                # already on state, and the layerwise strategy will
                                # pick up from layerwise_current_layer.
                                logger.info(
                                    "[SAGE_PRE_TTFT_STATE] req=%s early_blend=%s "
                                    "selected=%s current_layer=%d state_id=%d",
                                    request.req_id,
                                    getattr(state, "early_blend_done", "MISSING"),
                                    state.layerwise_selected_indices is not None
                                    if state is not None else "no_state",
                                    getattr(state, "layerwise_current_layer", -1),
                                    id(state),
                                )
                                _eb_lw = (
                                    getattr(state, "early_blend_done", False)
                                    or LMCacheConnectorV1Impl._shared_early_blend_done.get(
                                        request.req_id, False
                                    )
                                )
                                if _eb_lw:
                                    logger.info(
                                        "[SAGE_PATH] request=%s → EXEC:layerwise_pre_ttft_skipped "
                                        "(early, selected=%s current_layer=%d)",
                                        request.req_id,
                                        getattr(state, "layerwise_selected_indices", None) is not None,
                                        getattr(state, "layerwise_current_layer", 0),
                                    )
                                elif (
                                    self._sage_method == SageMethod.CACHEBLEND_LAYERWISE
                                    and state.layerwise_selected_indices is None
                                ):
                                    if self._layerwise_defer_diff_k:
                                        logger.info(
                                            "[SAGE_PATH] request=%s → EXEC:layerwise_defer_save",
                                            request.req_id,
                                        )
                                        state.layerwise_deferred_blend_params = {
                                            "blend_tokens": blend_tokens,
                                            "sage_token_mask": sage_token_mask,
                                            "blend_slot_mapping": blend_slot_mapping,
                                            "blend_block_table": blend_block_table,
                                            "kvcaches": kvcaches,
                                            "sage_zero_copy": request.sage_blocks_transferred,
                                        }
                                        logger.info(
                                            "[SAGE_LAYERWISE_PRE_TTFT] request=%s "
                                            "N=%d deferred diff-k to decode step 1",
                                            request.req_id,
                                            int(blend_tokens.shape[0]),
                                        )
                                    else:
                                        # SG path: run layerwise pre-TTFT blend.
                                        # MG path should never reach here — caught
                                        # by _shared_early_blend_done check above.
                                        assert not LMCacheConnectorV1Impl._shared_early_blend_done.get(
                                            request.req_id, False
                                        ), (
                                            f"Double processing: sage_process_layer already "
                                            f"blended {request.req_id} but layerwise "
                                            f"pre-TTFT is about to run again"
                                        )
                                        logger.info(
                                            "[SAGE_PATH] request=%s → EXEC:layerwise_pre_ttft_immediate",
                                            request.req_id,
                                        )
                                        self._run_layerwise_pre_ttft(
                                            req_id=request.req_id,
                                            blend_tokens=blend_tokens,
                                            sage_token_mask=sage_token_mask,
                                            blend_slot_mapping=blend_slot_mapping,
                                            blend_block_table=blend_block_table,
                                            kvcaches=kvcaches,
                                            sage_zero_copy=request.sage_blocks_transferred,
                                            state=state,
                                            suffix_len=request.query_token_count,
                                        )
                                        # Immediately set up first injection in the same
                                        # step so it piggybacks on the first decode token's
                                        # forward — avoids wasting a decode step before
                                        # the first injection (matching MG behavior).
                                        if (
                                            state.layerwise_selected_indices is not None
                                            and state.layerwise_saved_hidden is not None
                                        ):
                                            _first_step = strategy.next_step(
                                                state, self.blender.num_layers
                                            )
                                            if _first_step.should_blend and _first_step.layer_range is not None:
                                                _lw_s, _lw_e = _first_step.layer_range
                                                _sel = state.layerwise_selected_indices
                                                _cp = self._get_or_create_cached_prompt_inputs(
                                                    req_id=request.req_id,
                                                    tokens=tokens,
                                                    prompt_len=int(lmcache_cached_tokens),
                                                    slot_mapping=slot_mapping,
                                                )
                                                self._fused_inject_pending[request.req_id] = {
                                                    "token_ids": _cp["prompt_tokens"][_sel],
                                                    "positions": _sel.clone(),
                                                    "slots": _cp["prompt_slot_mapping"][_sel],
                                                    "hidden_states": state.layerwise_saved_hidden,
                                                    "residual": state.layerwise_saved_residual,
                                                    "inject_layer": _lw_s,
                                                    "extract_layer": _lw_e,
                                                }
                                                self._sage_pending_blend_launch_t0[
                                                    request.req_id
                                                ] = time.perf_counter()
                                                should_blend_this_step = False
                                                logger.info(
                                                    "[SAGE_LAYERWISE] request=%s "
                                                    "pre_ttft+inject layers=[%d,%d) M=%d",
                                                    request.req_id, _lw_s, _lw_e,
                                                    int(_sel.numel()),
                                                )
                                elif (
                                    self._sage_method == SageMethod.CACHEBLEND_TOKENWISE
                                    and is_pre_ttft_step
                                ):
                                        # SG path: run tokenwise pre-TTFT.
                                        # MG path should never reach here.
                                        assert not LMCacheConnectorV1Impl._shared_early_blend_done.get(
                                            request.req_id, False
                                        ), (
                                            f"Double processing: sage_process_layer already "
                                            f"blended {request.req_id} but tokenwise "
                                            f"pre-TTFT is about to run again"
                                        )
                                        logger.info(
                                            "[SAGE_PATH] request=%s → EXEC:tokenwise_pre_ttft_scoring",
                                            request.req_id,
                                        )
                                        self._run_tokenwise_pre_ttft(
                                            req_id=request.req_id,
                                            blend_tokens=blend_tokens,
                                            sage_token_mask=sage_token_mask,
                                            blend_slot_mapping=blend_slot_mapping,
                                            blend_block_table=blend_block_table,
                                            kvcaches=kvcaches,
                                            sage_zero_copy=request.sage_blocks_transferred,
                                            state=state,
                                            suffix_len=request.query_token_count,
                                        )
                                        # Hash pre-TTFT state for tokenwise
                                elif (
                                    self._sage_method == SageMethod.MAGNET
                                    and is_pre_ttft_step
                                ):
                                        # Magnet (query-aware adaptive): score with
                                        # attention-weighted importance and recompute
                                        # tokens exceeding threshold p.
                                        logger.info(
                                            "[SAGE_PATH] request=%s → EXEC:magnet_pre_ttft",
                                            request.req_id,
                                        )
                                        self._run_magnet_pre_ttft(
                                            req_id=request.req_id,
                                            blend_tokens=blend_tokens,
                                            sage_token_mask=sage_token_mask,
                                            blend_slot_mapping=blend_slot_mapping,
                                            blend_block_table=blend_block_table,
                                            kvcaches=kvcaches,
                                            sage_zero_copy=request.sage_blocks_transferred,
                                            state=state,
                                            suffix_len=request.query_token_count,
                                            chunk_boundaries=_chunk_bounds,
                                        )
                                elif (
                                    self._sage_method == SageMethod.MAGNET
                                    and not is_pre_ttft_step
                                ):
                                        # Magnet decode-time: no-op here.
                                        # Recomputation of newly-identified
                                        # anchors happens via fused-inject
                                        # in the NEXT decode forward
                                        # (populated by the decode hook's
                                        # final post-hook). The request is
                                        # marked completed after pre-TTFT,
                                        # so this branch should be rare.
                                        logger.info(
                                            "[SAGE_PATH] request=%s → "
                                            "SKIP:magnet (fused-inject path)",
                                            request.req_id,
                                        )
                                else:
                                    # SG path: blend_from_gpu handles sync cacheblend.
                                    # MG path should never reach here — early_blend_done
                                    # at the layerwise pre-TTFT check (above) skips this.
                                    assert not LMCacheConnectorV1Impl._shared_early_blend_done.get(
                                        request.req_id, False
                                    ), (
                                        f"Double processing: sage_process_layer already "
                                        f"blended {request.req_id} but blend_from_gpu "
                                        f"is about to run again"
                                    )
                                    logger.info(
                                        "[SAGE_PATH] request=%s → EXEC:sync_cacheblend",
                                        request.req_id,
                                    )
                                    self.blender.blend_from_gpu(
                                        blend_tokens,
                                        blend_token_mask,
                                        step_recompute_ratio=step_recompute_ratio,
                                        suffix_len=request.query_token_count,
                                        include_suffix=True,
                                        kvcaches=kvcaches,
                                        slot_mapping=blend_slot_mapping,
                                        block_table=blend_block_table,
                                        sage_zero_copy=request.sage_blocks_transferred,
                                    )
                                    self._sage_incremental_completed_requests.add(
                                        request.req_id
                                    )
                                    # Hash all layers after sync CB (same point for both SKIP and EXEC)
                                    logger.info(
                                        "[SAGE_INCREMENTAL] Ran sync blend "
                                        "request=%s decode_step=%d ratio=%s",
                                        request.req_id,
                                        state.decode_step
                                        if state is not None
                                        else -1,
                                        step_recompute_ratio,
                                    )
                            else:
                                logger.info(
                                    "[SAGE_INCREMENTAL] request=%s skipped blend after "
                                    "safety checks (reason=%s)",
                                    request.req_id,
                                    "n/a",
                                )
                    else:
                        logger.debug(
                            "Using CPU-based blending for request %s",
                            request.req_id,
                        )
                        self.blender.blend(
                            tokens[:lmcache_cached_tokens],
                            token_mask[:lmcache_cached_tokens],
                            kvcaches=kvcaches,
                            slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        )
                    if sage_enable_timing:
                        end_time = time.time()
                        logger.info(
                            "Finished blending for request %s, at time %.3f, "
                            "duration %.3f seconds",
                            request.req_id,
                            end_time,
                            end_time - start_time,
                        )
                else:
                    layerwise_retriever = self.lmcache_engine.retrieve_layer(
                        tokens[:lmcache_cached_tokens],
                        token_mask[:lmcache_cached_tokens],
                        kvcaches=kvcaches,
                        slot_mapping=slot_mapping[:lmcache_cached_tokens],
                        sync=sync,
                    )
                    # NOTE: retrieve for two layers at the first layer
                    next(layerwise_retriever)
                    next(layerwise_retriever)
                    self.layerwise_retrievers.append(layerwise_retriever)
            else:
                ret_token_mask = self.lmcache_engine.retrieve(
                    tokens[:lmcache_cached_tokens],
                    token_mask[:lmcache_cached_tokens],
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping[:lmcache_cached_tokens],
                    request_configs=request.request_configs,
                    req_id=request.req_id,
                    skip_contains_check=True,
                )

                # Check the result
                num_retrieved_tokens = ret_token_mask.sum().item()
                num_expected_tokens = (
                    lmcache_cached_tokens - request.load_spec.vllm_cached_tokens
                )
                if num_retrieved_tokens < num_expected_tokens:
                    logger.error(
                        "Request %s"
                        "The number of retrieved tokens is less than the "
                        "expected number of tokens! This should not happen!",
                        request.req_id,
                    )
                    logger.error(
                        "Num retrieved tokens: %d, num expected tokens: %d",
                        num_retrieved_tokens,
                        num_expected_tokens,
                    )
                    """
                    Report failed block IDs in case of partial failure.
                    """
                    missing_blocks = self.record_failed_blocks(
                        request.req_id,
                        token_mask[:lmcache_cached_tokens],
                        ret_token_mask,
                        slot_mapping[:lmcache_cached_tokens],
                    )
                    self._invalid_block_ids.update(missing_blocks)

            self._stats_monitor.update_interval_vllm_hit_tokens(
                request.load_spec.vllm_cached_tokens
            )
            self._stats_monitor.update_interval_prompt_tokens(len(tokens))

    def cache_prompt_embeddings(self, scheduler_output, embeddings,
                               positions=None):
        """Cache prompt embeddings and M-RoPE positions for multimodal
        fused injection. ~160MB for 23K visual tokens."""
        if not hasattr(self, "_cached_prompt_embeds"):
            self._cached_prompt_embeds: dict[str, torch.Tensor] = {}
        if not hasattr(self, "_cached_prompt_positions"):
            self._cached_prompt_positions: dict[str, torch.Tensor] = {}
        for req in scheduler_output.scheduled_new_reqs:
            num_tokens = scheduler_output.num_scheduled_tokens.get(
                req.req_id, 0
            )
            if num_tokens > 0:
                self._cached_prompt_embeds[req.req_id] = (
                    embeddings[:num_tokens].detach().clone()
                )
                if positions is not None:
                    if positions.dim() == 2:
                        self._cached_prompt_positions[req.req_id] = (
                            positions[:, :num_tokens].detach().clone()
                        )
                    else:
                        self._cached_prompt_positions[req.req_id] = (
                            positions[:num_tokens].detach().clone()
                        )

    def inject_fused_recompute_tokens(
        self,
        model_runner: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_indices: torch.Tensor,
        attn_metadata: Any,
        num_reqs: int,
        inputs_embeds: torch.Tensor = None,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
        """Inject M recompute tokens into the model forward batch.

        For multimodal models (input_ids=None, inputs_embeds!=None),
        uses cached embeddings from prefill to replay visual tokens.
        Returns 4-tuple (input_ids, positions, logits_indices, inputs_embeds)
        when operating in multimodal mode.

        Replay tokens are placed FIRST (sorted by position ascending),
        followed by the decode token LAST.  This ensures causal attention
        with bottom-right alignment gives approximate causal masking for
        replay tokens while the decode token (last) sees all keys.

        Called inside vLLM's execute_model ``with`` block, after start_load_kv
        and before _model_forward.  Modifies attn_metadata in-place and returns
        updated (input_ids, positions, logits_indices) tensors.

        Only active for cacheblend-tokenwise and cacheblend-layerwise methods.
        Silently no-ops if there are no pending injections.
        """
        _is_multimodal = input_ids is None and inputs_embeds is not None
        if not self._fused_inject_pending:
            if _is_multimodal:
                return input_ids, positions, logits_indices, inputs_embeds
            return input_ids, positions, logits_indices

        req_ids: list[str] = list(model_runner.input_batch.req_ids[:num_reqs])
        pending: dict[str, dict[str, torch.Tensor]] = {}
        for rid in req_ids:
            payload = self._fused_inject_pending.pop(rid, None)
            if payload is not None:
                pending[rid] = payload
        if not pending:
            if _is_multimodal:
                return input_ids, positions, logits_indices, inputs_embeds
            return input_ids, positions, logits_indices

        # ── Augment payloads with previously-generated decode tokens ──
        # When enabled, include all decode tokens generated so far in the
        # replay batch.  Their KV was computed with partially-wrong prefix
        # and recomputing them is negligible overhead (D<<M) but ensures
        # future tokens are conditioned on correct KV for the full context.
        # Skip decode token augmentation when late injection is active —
        # the decode token already flows through the forward from layer 0.
        _any_late_inject = any(
            "inject_layer" in p for p in pending.values()
        )
        if self._sage_method == SageMethod.CACHEBLEND_TOKENWISE and not _any_late_inject:
            input_batch = model_runner.input_batch
            block_table_obj = input_batch.block_table[0]  # first KV cache group
            bt_np = block_table_obj.block_table.np
            block_size = block_table_obj.block_size
            max_blocks_per_req = block_table_obj.max_num_blocks_per_req

            for rid in list(pending.keys()):
                cached_prompt = self._sage_cached_prompt_inputs.get(rid)
                if cached_prompt is None:
                    continue
                prompt_len = int(cached_prompt["prompt_len"])
                req_idx = input_batch.req_id_to_index.get(rid)
                if req_idx is None:
                    continue
                num_computed = int(
                    input_batch.num_computed_tokens_cpu[req_idx]
                )
                num_decode_tokens = num_computed - prompt_len
                if num_decode_tokens <= 0:
                    continue

                # Gather decode token IDs, positions, and slot mappings.
                decode_positions_np = np.arange(
                    prompt_len, num_computed, dtype=np.int64
                )
                decode_token_ids_np = input_batch.token_ids_cpu[
                    req_idx, prompt_len:num_computed
                ].copy()
                # Compute slot mappings from block table.
                req_indices_np = np.full(
                    num_decode_tokens, req_idx, dtype=np.int64
                )
                bt_indices = (
                    req_indices_np * max_blocks_per_req
                    + decode_positions_np // block_size
                )
                block_numbers = bt_np.ravel()[bt_indices]
                block_offsets = decode_positions_np % block_size
                decode_slots_np = (
                    block_numbers * block_size + block_offsets
                )

                payload = pending[rid]
                dev = payload["token_ids"].device
                decode_tok = torch.tensor(
                    decode_token_ids_np, device=dev, dtype=torch.long
                )
                decode_pos = torch.tensor(
                    decode_positions_np, device=dev, dtype=torch.long
                )
                decode_slt = torch.tensor(
                    decode_slots_np, device=dev, dtype=torch.long
                )

                # Concatenate decode tokens with replay tokens.
                payload["token_ids"] = torch.cat(
                    [payload["token_ids"], decode_tok]
                )
                payload["positions"] = torch.cat(
                    [payload["positions"], decode_pos]
                )
                payload["slots"] = torch.cat(
                    [payload["slots"], decode_slt]
                )
                logger.info(
                    "[SAGE_CB_INC_FUSED] request=%s augmented payload with "
                    "%d decode tokens (prompt_len=%d computed=%d "
                    "total_replay=%d)",
                    rid,
                    num_decode_tokens,
                    prompt_len,
                    num_computed,
                    int(payload["positions"].shape[0]),
                )

        # Save ORIGINAL query_start_loc values before we modify them.
        # query_start_loc.gpu[:num_reqs+1] = cumulative token offsets.
        orig_starts = [int(model_runner.query_start_loc.gpu[i]) for i in range(num_reqs + 1)]

        # First pass: get first layer's slot_mapping (for decode slots).
        first_meta = next(iter(attn_metadata.values()))
        orig_slot_mapping = first_meta.slot_mapping  # shape [orig_total_tokens]
        per_req_m: list[int] = [0] * num_reqs
        total_M = 0
        for req_idx, req_id in enumerate(req_ids):
            payload = pending.get(req_id)
            if payload is None:
                continue
            m_tokens = int(payload["positions"].shape[0])
            per_req_m[req_idx] = m_tokens
            total_M += m_tokens

        if total_M == 0:
            if _is_multimodal:
                return input_ids, positions, logits_indices, inputs_embeds
            return input_ids, positions, logits_indices

        # ── Per-request late injection (unified path for any num_reqs) ──
        #
        # For each request r in the batch:
        #   - decode tokens (d_r ≥ 1) flow through every layer 0..N-1
        #   - replay tokens (M_r ≥ 0) enter at inject_layer_r and exit at
        #     extract_layer_r (or ride to the final layer for tokenwise)
        #
        # The batch shape changes layer-by-layer based on which requests
        # are currently in their replay window. Hooks at each transition
        # layer reshape the batch to match the next stretch.
        #
        # Layout invariant: at every layer, the batch is laid out as
        #   [span_0, span_1, ..., span_{num_reqs-1}]
        # where each span_r is [replay_r_rows (if active), decode_r_rows].
        # The order of requests matches model_runner.input_batch.req_ids.
        return self._sage_per_request_late_inject(
            model_runner=model_runner,
            input_ids=input_ids,
            positions=positions,
            logits_indices=logits_indices,
            attn_metadata=attn_metadata,
            num_reqs=num_reqs,
            inputs_embeds=inputs_embeds,
            req_ids=req_ids,
            pending=pending,
            per_req_m=per_req_m,
            orig_starts=orig_starts,
            orig_slot_mapping=orig_slot_mapping,
            is_multimodal=_is_multimodal,
        )

    def _sage_per_request_late_inject(
        self,
        model_runner: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        logits_indices: torch.Tensor,
        attn_metadata: Any,
        num_reqs: int,
        inputs_embeds: torch.Tensor,
        req_ids: list,
        pending: dict,
        per_req_m: list,
        orig_starts: list,
        orig_slot_mapping: torch.Tensor,
        is_multimodal: bool,
    ):
        """Per-request late-injection forward setup.

        Builds per-stretch attention metadata and registers PyTorch hooks
        on decoder layers so that each request's M_r replay tokens enter
        at its own inject_layer_r (with the saved boundary hidden state)
        and exit at its own extract_layer_r (saving the new boundary back
        to per-request state).

        For tokenwise (extract_layer is None), the replay tokens ride to
        the final layer alongside the decode tokens.

        Returns the updated (input_ids, positions, logits_indices, inputs_embeds)
        for layer-0 input. Note layer 0 still sees the unmodified
        decode-only batch — the replay rows get spliced in later by hooks.
        """
        from dataclasses import replace as dc_replace
        from vllm.forward_context import get_forward_context as _get_fwd_ctx
        import time as _time

        _t_setup_start = _time.perf_counter()

        device = positions.device
        slot_device = orig_slot_mapping.device

        # Per-step counters for the verify summary log.
        _verify_counters = {
            "transition_hook_calls": 0,
            "transition_hook_total_us": 0.0,
            "inject_count": 0,
            "extract_count": 0,
        }

        # ── Per-request entries: collect M_r, inject/extract layers, and
        # the sorted replay tensors for each pending request.
        # entries: list of dicts keyed by req_idx with all per-request data.
        entries: list[dict] = []
        for req_idx, rid in enumerate(req_ids):
            payload = pending.get(rid)
            if payload is None or per_req_m[req_idx] == 0:
                continue
            inject_layer = payload.get("inject_layer")
            if inject_layer is None or "hidden_states" not in payload:
                # No late-injection metadata; skip this request entirely
                # (it'll fall through to the no-injection path below if
                # nothing remains).
                continue
            extract_layer = payload.get("extract_layer")  # None for tokenwise

            fused_positions = payload["positions"]
            sort_idx = torch.argsort(fused_positions)
            sorted_positions = fused_positions[sort_idx].to(
                device=device, dtype=positions.dtype,
            )
            sorted_hidden = payload["hidden_states"][sort_idx].to(device=device)
            sorted_residual = (
                payload["residual"][sort_idx].to(device=device)
                if payload.get("residual") is not None else None
            )
            sorted_slots = payload["slots"][sort_idx].to(
                device=slot_device, dtype=orig_slot_mapping.dtype,
            )
            sorted_token_ids = None
            if not is_multimodal:
                sorted_token_ids = payload["token_ids"][sort_idx].to(
                    device=device, dtype=input_ids.dtype,
                )

            entries.append({
                "req_idx": req_idx,
                "rid": rid,
                "M_r": per_req_m[req_idx],
                "inject_layer": int(inject_layer),
                "extract_layer": (
                    int(extract_layer) if extract_layer is not None else None
                ),
                "positions": sorted_positions,
                "hidden": sorted_hidden,
                "residual": sorted_residual,
                "slots": sorted_slots,
                "token_ids": sorted_token_ids,
            })

        if not entries:
            # No requests have late-injection metadata; nothing to do.
            if is_multimodal:
                return input_ids, positions, logits_indices, inputs_embeds
            return input_ids, positions, logits_indices

        # ── Resolve decoder layers (works for text and VL models).
        _inner_model = model_runner.model
        if hasattr(_inner_model, "language_model"):
            _inner_model = _inner_model.language_model
        for _attr in ("model", "model"):
            if hasattr(_inner_model, _attr):
                _inner_model = getattr(_inner_model, _attr)
        _model_layers = _inner_model.layers
        num_model_layers = len(_model_layers)

        # ── Compute the active set per layer.
        # active_at_layer[L] = set of req_idx values whose
        #   inject_layer_r ≤ L < extract_layer_r (extract_layer_r=∞ if None).
        active_at_layer: list[set[int]] = [set() for _ in range(num_model_layers + 1)]
        for entry in entries:
            r_idx = entry["req_idx"]
            il = entry["inject_layer"]
            xl = entry["extract_layer"] if entry["extract_layer"] is not None else num_model_layers
            for L in range(il, min(xl, num_model_layers)):
                active_at_layer[L].add(r_idx)

        # ── Compute the per-request decode lengths from orig_starts.
        decode_lens = [
            orig_starts[r + 1] - orig_starts[r] for r in range(num_reqs)
        ]
        # Original seq_lens (KV cache lengths per request) — needed for
        # the per-stretch attn metadata.
        orig_attn_meta = next(iter(attn_metadata.values()))
        orig_seq_lens = orig_attn_meta.seq_lens
        orig_block_table = orig_attn_meta.block_table

        # ── Per-stretch metadata: a "stretch" is a maximal range of
        # consecutive layers with the same active set. Build one
        # FlashAttentionMetadata per distinct active set.
        # Map from frozenset(active_set) -> (slot_mapping, query_start_loc,
        # block_table, num_actual_tokens, max_query_len, sage_per_req_m,
        # row_layout) where row_layout describes which row indices in the
        # batch belong to which (req_idx, "replay"|"decode") group.
        stretch_meta_cache: dict[frozenset, Any] = {}

        def _build_stretch_meta(active_set: frozenset) -> Any:
            """Build per-stretch FlashAttentionMetadata for a given active set."""
            if active_set in stretch_meta_cache:
                return stretch_meta_cache[active_set]

            new_cu = [0]  # cu_seqlens_q for the spans (1 entry per req)
            new_slot_segments: list[torch.Tensor] = []
            new_per_req_m: list[int] = []
            running_offset = 0
            for r in range(num_reqs):
                # Decode rows for r come from orig slot_mapping.
                d_r = decode_lens[r]
                decode_slots = orig_slot_mapping[
                    orig_starts[r]:orig_starts[r + 1]
                ]
                if r in active_set:
                    # Find the entry for r.
                    entry = next(e for e in entries if e["req_idx"] == r)
                    m_r = entry["M_r"]
                    new_slot_segments.append(entry["slots"])
                    new_slot_segments.append(decode_slots)
                    span_len = m_r + d_r
                    new_per_req_m.append(m_r)
                else:
                    new_slot_segments.append(decode_slots)
                    span_len = d_r
                    new_per_req_m.append(0)
                running_offset += span_len
                new_cu.append(running_offset)

            new_slot_mapping = torch.cat(new_slot_segments, dim=0)
            new_query_start_loc = torch.tensor(
                new_cu, dtype=torch.int32, device=slot_device,
            )
            new_num_tokens = running_offset
            new_max_query_len = max(
                new_cu[r + 1] - new_cu[r] for r in range(num_reqs)
            )

            # Split cu_seqlens_q for replay+decode subsequences so
            # causal masking is applied within each subsequence, not
            # across the (replay, decode) boundary. This makes the
            # split backend-agnostic — no SAGE code needed in
            # flash_attn.py, triton_attn.py, etc.
            has_replay = any(m > 0 for m in new_per_req_m)
            if has_replay:
                split_cu = [0]
                split_seq_k: list[int] = []
                split_block_rows: list[int] = []
                seq_k_list = orig_seq_lens.tolist()
                for r in range(num_reqs):
                    span_start = new_cu[r]
                    span_end = new_cu[r + 1]
                    m_r = new_per_req_m[r]
                    d_r = (span_end - span_start) - m_r
                    if m_r > 0:
                        # Cap seqused_k at max_replay_position + 1 so
                        # the bottom-right causal mask does not extend
                        # into decode-token KV entries that were generated
                        # after the prompt.  For contiguous replay
                        # positions (e.g. suffix rerecompute) this gives
                        # the exact correct causal window.
                        _entry_r = next(
                            e for e in entries if e["req_idx"] == r
                        )
                        replay_sk = min(
                            seq_k_list[r],
                            int(_entry_r["positions"].max().item()) + 1,
                        )
                        split_cu.append(span_start + m_r)
                        split_seq_k.append(replay_sk)
                        split_block_rows.append(r)
                    if d_r > 0:
                        split_cu.append(span_start + m_r + d_r)
                        split_seq_k.append(seq_k_list[r])
                        split_block_rows.append(r)
                split_query_start_loc = torch.tensor(
                    split_cu, dtype=torch.int32, device=slot_device,
                )
                split_seqused_k = torch.tensor(
                    split_seq_k, dtype=orig_seq_lens.dtype,
                    device=orig_seq_lens.device,
                )
                split_max_query_len = max(
                    split_cu[i + 1] - split_cu[i]
                    for i in range(len(split_cu) - 1)
                )
                block_row_idx = torch.tensor(
                    split_block_rows, dtype=torch.long,
                    device=orig_block_table.device,
                )
                split_block_table = orig_block_table[block_row_idx].contiguous()
                stretch_meta = dc_replace(
                    orig_attn_meta,
                    slot_mapping=new_slot_mapping,
                    num_actual_tokens=new_num_tokens,
                    max_query_len=split_max_query_len,
                    query_start_loc=split_query_start_loc,
                    seq_lens=split_seqused_k,
                    block_table=split_block_table,
                    scheduler_metadata=None,
                )
            else:
                stretch_meta = dc_replace(
                    orig_attn_meta,
                    slot_mapping=new_slot_mapping,
                    num_actual_tokens=new_num_tokens,
                    max_query_len=new_max_query_len,
                    query_start_loc=new_query_start_loc,
                    seq_lens=orig_seq_lens,
                    block_table=orig_block_table,
                    scheduler_metadata=None,
                )

            stretch_meta_cache[active_set] = stretch_meta
            return stretch_meta

        # ── Build base metadata for "no replays active" (layers before
        # any inject and after any extract). This is the original layout
        # but with sage_per_req_m = [0]*num_reqs so flash_attn doesn't
        # try to apply the late-injection split.
        base_meta = _build_stretch_meta(frozenset())

        # Patch attn_metadata: for each layer, point to the right stretch.
        for layer_name in list(attn_metadata.keys()):
            layer_idx = None
            for part in layer_name.split('.'):
                try:
                    layer_idx = int(part)
                except ValueError:
                    pass
            if layer_idx is None or layer_idx >= num_model_layers:
                continue
            active_set = frozenset(active_at_layer[layer_idx])
            attn_metadata[layer_name] = _build_stretch_meta(active_set)

        # ── Hooks: shape tracker.
        # _current_active is a mutable single-element list holding the
        # frozenset of currently-active req_idx values. Updated by
        # transition hooks.
        _current_active = [frozenset()]
        # _current_positions holds the most recently expanded positions
        # tensor. Non-transition layers in the replay window read from
        # this so they get the same shape as hidden_states (the model
        # loop reuses the same `positions` arg across all layers).
        _current_positions: list[torch.Tensor] = [None]

        # row_indices_for_active(active_set) returns the list of
        # (req_idx, "replay"|"decode") segments in batch order. Used by
        # hooks to know where to splice/extract per-request rows.
        def _layout_for(active_set: frozenset) -> list[tuple]:
            layout: list[tuple] = []
            for r in range(num_reqs):
                if r in active_set:
                    layout.append((r, "replay"))
                layout.append((r, "decode"))
            return layout

        # The set of layers where the active set changes. We register
        # hooks at these layers.
        transition_layers: set[int] = set()
        prev_active: frozenset = frozenset()
        for L in range(num_model_layers):
            cur_active = frozenset(active_at_layer[L])
            if cur_active != prev_active:
                transition_layers.add(L)
            prev_active = cur_active

        # Pre-compute per-request decode-row index ranges within the
        # ORIGINAL (decode-only) batch, used by inject hooks to splice in.
        decode_orig_starts = list(orig_starts)  # cumulative

        _hook_handles: list = []

        # Build per-request mrope expansion helper (works for both text
        # and multimodal).
        def _expand_positions_for_active(
            current_pos: torch.Tensor, active_set: frozenset,
        ) -> torch.Tensor:
            """Construct positions tensor for the new batch shape under
            active_set, given the current decode-only positions tensor.

            For each request: emit M_r replay positions (if active) then
            d_r decode positions for that request.
            """
            is_mrope = current_pos.dim() == 2
            if is_mrope:
                pos_segments = []
                for r in range(num_reqs):
                    if r in active_set:
                        entry = next(e for e in entries if e["req_idx"] == r)
                        replay_pos_indices = entry["positions"].long()
                        # M-RoPE: pos is [3, total_decode_tokens]. Look up
                        # per-request mrope_positions for the M_r replay rows.
                        _req_state = model_runner.requests.get(entry["rid"])
                        if (
                            _req_state is not None
                            and getattr(_req_state, "mrope_positions", None) is not None
                        ):
                            _mrope = _req_state.mrope_positions
                            _idx_cpu = replay_pos_indices.cpu()
                            replay_3d = _mrope[:, _idx_cpu].to(device=current_pos.device)
                        else:
                            replay_3d = entry["positions"].unsqueeze(0).expand(
                                current_pos.shape[0], -1,
                            )
                        pos_segments.append(replay_3d)
                    decode_slice = current_pos[
                        :, decode_orig_starts[r]:decode_orig_starts[r + 1]
                    ]
                    pos_segments.append(decode_slice)
                return torch.cat(pos_segments, dim=1)
            else:
                pos_segments = []
                for r in range(num_reqs):
                    if r in active_set:
                        entry = next(e for e in entries if e["req_idx"] == r)
                        pos_segments.append(entry["positions"])
                    pos_segments.append(
                        current_pos[decode_orig_starts[r]:decode_orig_starts[r + 1]]
                    )
                return torch.cat(pos_segments, dim=0)

        # Helper: row index ranges in the CURRENT batch for layout.
        def _row_ranges_for(active_set: frozenset) -> dict:
            """Return mapping req_idx -> (replay_start, replay_end, decode_start, decode_end)
            for the current active_set. Replay range is None if request
            is not active."""
            ranges: dict = {}
            cursor = 0
            for r in range(num_reqs):
                if r in active_set:
                    entry = next(e for e in entries if e["req_idx"] == r)
                    m_r = entry["M_r"]
                    rep_start = cursor
                    rep_end = cursor + m_r
                    cursor = rep_end
                else:
                    rep_start = rep_end = None
                d_r = decode_lens[r]
                dec_start = cursor
                dec_end = cursor + d_r
                cursor = dec_end
                ranges[r] = (rep_start, rep_end, dec_start, dec_end)
            return ranges

        # ── Register a transition hook on each transition layer.
        # The hook reshapes the input by adding rows for entering
        # requests (with their saved boundary hidden state) and removing
        # rows for exiting requests (saving their new boundary state to
        # the forward context).

        def _make_transition_hook(L_local: int):
            def _hook(module, args):
                _t_hook = _time.perf_counter()
                pos, hs, res = args
                old_active = _current_active[0]
                new_active = frozenset(active_at_layer[L_local])

                if old_active == new_active:
                    return args

                # Compute old and new row ranges.
                old_ranges = _row_ranges_for(old_active)
                new_ranges = _row_ranges_for(new_active)

                exiting = old_active - new_active
                entering = new_active - old_active

                # [VERIFY] Per-transition detail log: layer + active set
                # deltas + entering/exiting request ids. Cheap: just str
                # formatting + dict lookups, no GPU work.
                _entry_by_idx = {e["req_idx"]: e for e in entries}

                # Debug: verify decode row mapping
                _ent_rids = [_entry_by_idx[r]["rid"] for r in sorted(entering)]
                _exit_rids = [_entry_by_idx[r]["rid"] for r in sorted(exiting)]
                _stay_rids = [
                    _entry_by_idx[r]["rid"]
                    for r in sorted(old_active & new_active)
                ]
                # Log per-request row structure at transitions
                _row_detail = []
                _nr = _row_ranges_for(new_active)
                for r in sorted(new_active):
                    _rs, _re, _ds, _de = _nr[r]
                    _rid = _entry_by_idx[r]["rid"] if r in _entry_by_idx else f"req{r}"
                    _row_detail.append(
                        f"{_rid}:replay=[{_rs},{_re})decode=[{_ds},{_de})"
                    )
                logger.info(
                    "[SAGE_FUSED_VERIFY_T] L=%d hs_in=%d "
                    "old=%d new=%d enter=%s exit=%s stay=%s "
                    "rows=[%s] pos=%s",
                    L_local, hs.shape[0],
                    len(old_active), len(new_active),
                    _ent_rids, _exit_rids, _stay_rids,
                    " ".join(_row_detail),
                    pos[:min(8, len(pos))].tolist() if pos is not None else "None",
                )
                # Save extracted hidden/residual for exiting requests.
                if exiting:
                    fwd_ctx = _get_fwd_ctx()
                    if fwd_ctx.sage_extract_result is None:
                        fwd_ctx.sage_extract_result = {}
                    for r in exiting:
                        rep_s, rep_e, _, _ = old_ranges[r]
                        if rep_s is None:
                            continue
                        entry = next(e for e in entries if e["req_idx"] == r)
                        fwd_ctx.sage_extract_result[entry["rid"]] = {
                            "hidden_states": hs[rep_s:rep_e].detach().clone(),
                            "residual": (
                                res[rep_s:rep_e].detach().clone()
                                if res is not None else None
                            ),
                        }
                        _verify_counters["extract_count"] += 1

                # Now build the new hidden_states / residual / positions
                # tensors with the proper row layout for new_active.
                new_hs_segments: list[torch.Tensor] = []
                new_res_segments: list[torch.Tensor] = []
                for r in range(num_reqs):
                    if r in new_active:
                        entry = next(e for e in entries if e["req_idx"] == r)
                        if r in entering:
                            # Inject saved boundary hidden state.
                            new_hs_segments.append(entry["hidden"])
                            if entry["residual"] is not None:
                                new_res_segments.append(entry["residual"])
                            _verify_counters["inject_count"] += 1
                        else:
                            # Continuing — copy from current hs.
                            old_rep_s, old_rep_e, _, _ = old_ranges[r]
                            new_hs_segments.append(hs[old_rep_s:old_rep_e])
                            if res is not None:
                                new_res_segments.append(res[old_rep_s:old_rep_e])
                    # Decode rows: always carried over from current hs.
                    old_dec_s = old_ranges[r][2]
                    old_dec_e = old_ranges[r][3]
                    new_hs_segments.append(hs[old_dec_s:old_dec_e])
                    if res is not None:
                        new_res_segments.append(res[old_dec_s:old_dec_e])

                new_hs = torch.cat(new_hs_segments, dim=0)
                new_res = (
                    torch.cat(new_res_segments, dim=0)
                    if res is not None and new_res_segments else res
                )
                new_pos = _expand_positions_for_active(pos, new_active)

                # Per-call shape sanity check (cheap: just int compare).
                _expected_rows = sum(
                    (e["M_r"] if e["req_idx"] in new_active else 0)
                    + decode_lens[e["req_idx"]]
                    for e in entries
                )
                for r in range(num_reqs):
                    if not any(e["req_idx"] == r for e in entries):
                        _expected_rows += decode_lens[r]
                if new_hs.shape[0] != _expected_rows:
                    logger.error(
                        "[SAGE_VERIFY] L=%d shape mismatch: hs=%d expected=%d",
                        L_local, new_hs.shape[0], _expected_rows,
                    )

                _current_active[0] = new_active
                _current_positions[0] = new_pos
                _verify_counters["transition_hook_calls"] += 1
                _verify_counters["transition_hook_total_us"] += (
                    (_time.perf_counter() - _t_hook) * 1e6
                )
                return (new_pos, new_hs, new_res)
            return _hook

        # Non-transition pre-hook: layers between transitions reuse the
        # current positions tensor (the model loop passes the same
        # original `positions` arg across all layers, but transition
        # hooks have reshaped hidden_states; we need to keep positions
        # in sync).
        def _carry_positions_pre_hook(module, args):
            pos, hs, res = args
            cur_pos = _current_positions[0]
            if cur_pos is None:
                return args
            cur_pos_seqlen = (
                cur_pos.shape[-1] if cur_pos.dim() == 2 else cur_pos.shape[0]
            )
            hs_len = hs.shape[0]
            if cur_pos_seqlen == hs_len:
                import os as _os_cp
                return (cur_pos, hs, res)
            return args

        for L in sorted(transition_layers):
            h = _model_layers[L].register_forward_pre_hook(_make_transition_hook(L))
            _hook_handles.append(h)

        # Carry-positions hook on every layer in the affected range
        # that's NOT a transition layer. This keeps positions in sync
        # with the (possibly reshaped) hidden_states tensor.
        if entries:
            min_inject = min(e["inject_layer"] for e in entries)
            max_extract = max(
                e["extract_layer"] if e["extract_layer"] is not None else num_model_layers
                for e in entries
            )
            for L in range(min_inject + 1, max_extract):
                if L in transition_layers or L >= num_model_layers:
                    continue
                h = _model_layers[L].register_forward_pre_hook(_carry_positions_pre_hook)
                _hook_handles.append(h)

        # Also register a final post-hook on the last layer of any
        # request whose extract_layer is None (tokenwise) — wait,
        # tokenwise rides to the final layer. We don't extract those.
        # We DO need a post-hook on layer (extract_layer - 1) for
        # layerwise requests where extract_layer == num_model_layers
        # (i.e., the request would naturally exit at the end).
        # The transition_layers logic above already handles transitions
        # at layer extract_layer (request exits when L == extract_layer
        # and is removed from active set). But the extraction (saving
        # the boundary) needs to happen on the OUTPUT of the last layer
        # the request is in (extract_layer - 1), not the INPUT of
        # extract_layer. The pre-hook approach above does extraction on
        # the input of layer extract_layer, which corresponds to the
        # output of layer (extract_layer - 1) — semantically equivalent
        # because layers don't modify their input.
        # However if any request's extract_layer == num_model_layers,
        # there's no layer extract_layer to register a pre-hook on.
        # Handle that case with a post-hook on the last layer.
        last_layer_extracts = [
            e for e in entries
            if e["extract_layer"] is not None
            and e["extract_layer"] == num_model_layers
        ]
        if last_layer_extracts:
            def _final_post_hook(module, args, output):
                hs, res = output
                fwd_ctx = _get_fwd_ctx()
                if fwd_ctx.sage_extract_result is None:
                    fwd_ctx.sage_extract_result = {}
                final_active = frozenset(_current_active[0])
                ranges = _row_ranges_for(final_active)
                for entry in last_layer_extracts:
                    r = entry["req_idx"]
                    if r not in final_active:
                        continue
                    rep_s, rep_e, _, _ = ranges[r]
                    if rep_s is None:
                        continue
                    fwd_ctx.sage_extract_result[entry["rid"]] = {
                        "hidden_states": hs[rep_s:rep_e].detach().clone(),
                        "residual": (
                            res[rep_s:rep_e].detach().clone()
                            if res is not None else None
                        ),
                    }
                # Drop the replay rows from the output so the final
                # logits indexing only sees decode rows.
                keep_segments_h: list[torch.Tensor] = []
                keep_segments_r: list[torch.Tensor] = []
                for r in range(num_reqs):
                    _, _, dec_s, dec_e = ranges[r]
                    keep_segments_h.append(hs[dec_s:dec_e])
                    if res is not None:
                        keep_segments_r.append(res[dec_s:dec_e])
                new_hs = torch.cat(keep_segments_h, dim=0)
                new_res = (
                    torch.cat(keep_segments_r, dim=0)
                    if res is not None and keep_segments_r else res
                )
                return (new_hs, new_res)
            h = _model_layers[num_model_layers - 1].register_forward_hook(_final_post_hook)
            _hook_handles.append(h)

        # Store handles for cleanup in wait_for_save.
        self._sage_inject_hook_handles = _hook_handles

        # ── Compute the layer-0 input.
        #
        # Layer 0 sees the original decode-only batch (no replay rows
        # yet). The transition hooks will splice rows in at the right
        # layers. UNLESS layer 0 is itself a transition layer (i.e.,
        # some request has inject_layer == 0). In that case the hook on
        # layer 0 will fire and reshape before layer 0's compute, so we
        # still pass the decode-only input here.
        #
        # The initial _current_active is frozenset() (no replay rows
        # present). We don't actually run with this layout — the first
        # transition hook updates it.

        # ── logits_indices: where the decode token(s) for each request
        # land in the FINAL layer's output.
        #
        # The final layer's output shape depends on whether any requests
        # are still active at layer N-1:
        #   - Layerwise: at the final layer, all requests have exited
        #     their windows, so the batch is just decode rows in order.
        #     logits_indices for request r is the position of its last
        #     decode token in the decode-only batch, which equals
        #     orig_starts[r+1] - 1 (same as the original).
        #   - Tokenwise: at the final layer, all tokenwise requests still
        #     have their replay rows in the batch. logits_indices for
        #     request r must point to the decode row(s) AFTER the replay
        #     rows in r's span.
        # When all entries are layerwise (have extract_layer set), the
        # final batch is decode-only and logits_indices is unchanged.
        any_tokenwise = any(e["extract_layer"] is None for e in entries)
        if any_tokenwise:
            # Final layer's active set: all requests with extract_layer is None.
            final_active = frozenset(
                e["req_idx"] for e in entries if e["extract_layer"] is None
            )
            ranges = _row_ranges_for(final_active)
            new_logits_list = []
            for r in range(num_reqs):
                _, _, dec_s, dec_e = ranges[r]
                # Pick the LAST decode row for each request.
                new_logits_list.append(dec_e - 1)
            new_logits_indices = torch.tensor(
                new_logits_list, device=logits_indices.device,
                dtype=logits_indices.dtype,
            )
        else:
            new_logits_indices = logits_indices

        # Record injected requests for completion tracking in wait_for_save.
        for entry in entries:
            self._fused_inject_active.add(entry["rid"])

        # The transition hooks haven't fired yet — they fire during the
        # forward pass. Stash the counters dict on the adapter so the
        # hooks can update it, and we'll log the totals after the forward
        # pass completes (in wait_for_save).
        self._sage_last_inject_counters = _verify_counters
        self._sage_last_inject_counters["entries"] = len(entries)
        self._sage_last_inject_counters["transitions"] = len(transition_layers)
        self._sage_last_inject_counters["setup_us"] = (
            (_time.perf_counter() - _t_setup_start) * 1e6
        )

        logger.info(
            "[SAGE_FUSED] Per-request late injection: %d entries, "
            "%d transition layers, setup_us=%.0f",
            len(entries),
            len(transition_layers),
            self._sage_last_inject_counters["setup_us"],
        )

        if is_multimodal:
            return input_ids, positions, new_logits_indices, inputs_embeds
        return input_ids, positions, new_logits_indices

    def record_failed_blocks(
        self,
        request_id: str,
        expected_mask: torch.Tensor,
        ret_mask: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> set[int]:
        """Record block IDs associated with failed load attempts.

        Args:
            request_id: request id from vLLM.
            expected_mask: Boolean tensor indicating which tokens were expected to
                be loaded from LMCache. True means the token should be loaded,
                False means the token is already cached in vLLM and does not need
                to be loaded from LMCache.
            ret_mask: Boolean tensor indicating which tokens were actually
                successfully retrieved from LMCache. True means the token was
                successfully loaded. For example, if 256 tokens are expected to be
                loaded, but only 192 tokens are successfully loaded, then the
                ret_mask will be a tensor of 256 items like [T, T, ..., F, F, ...]
                where the first 192 elements are True and the last 64 elements
                are False.
            slot_mapping: Tensor indicating slot IDs for each token. The block
                ID is computed by dividing the slot ID by the block size.

        Example:
            expected_mask = [F, T, T, T] meaning the 1st is in vLLM cache
            ret_mask = [F, T, F, F] meaning failure from loading the 3rd
            missing_mask = expected_mask & ~ret_mask = [F, F, T, T]
            missing_indices = [2, 3]
            then missing_blocks is calculated from slot_mapping and missing_indices

        Returns:
            set[int]: Set of block IDs that failed to load.
        """

        if expected_mask.numel() == 0:
            return set()

        expected_mask_cpu = expected_mask.to(device="cpu", dtype=torch.bool)
        ret_mask_cpu = ret_mask.to(device="cpu", dtype=torch.bool)

        if ret_mask_cpu.shape[0] != expected_mask_cpu.shape[0]:
            logger.debug("expected_mask_cpu.shape[0] != ret_mask_cpu.shape[0]")
            return set()

        missing_mask = expected_mask_cpu & ~ret_mask_cpu
        if not torch.any(missing_mask):
            return set()

        missing_indices = torch.nonzero(missing_mask, as_tuple=False).view(-1)
        if missing_indices.numel() == 0:
            return set()

        slot_mapping_cpu = slot_mapping.to(device="cpu", dtype=torch.long)
        if slot_mapping_cpu.shape[0] > missing_mask.shape[0]:
            slot_mapping_cpu = slot_mapping_cpu[: missing_mask.shape[0]]

        missing_blocks_tensor = torch.unique(
            slot_mapping_cpu[missing_indices] // self._block_size
        )
        missing_blocks = {int(block.item()) for block in missing_blocks_tensor}

        if not missing_blocks:
            return set()

        logger.warning(
            "Request %s failed to load %d tokens across %d blocks",
            request_id,
            missing_indices.numel(),
            len(missing_blocks),
        )
        return missing_blocks

    @_lmcache_nvtx_annotate
    def wait_for_layer_load(self, layer_name: str) -> None:
        """Blocking until the KV for a specific layer is loaded into vLLM's
        paged buffer.

        This interface will be useful for layer-by-layer pipelining.

        Args:
            layer_name: the name of that layer
        """
        if self.layerwise_retrievers:
            logger.debug(f"Waiting for layer {self.current_layer} to be loaded")

        # Wait for the layer to be loaded
        for layerwise_retriever in self.layerwise_retrievers:
            ret_token_mask = next(layerwise_retriever)

            if self.current_layer == self.num_layers - 1:
                assert ret_token_mask is not None
                num_retrieved_tokens = ret_token_mask.sum().item()
                logger.info(f"Retrieved {num_retrieved_tokens} tokens")

        return

    @_lmcache_nvtx_annotate
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        """Start saving the a layer of KV cache from vLLM's paged buffer
        to the connector.

        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current
                layer in vLLM.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        assert self.lmcache_engine is not None

        if not self.use_layerwise:
            return

        # Skip CPU store when Sage is active — KV lives in GPU paged memory
        if self.enable_sage:
            return

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return
        if self._parent._connector_metadata is None:
            logger.warning(
                "In connector.save_kv_layer, but the connector metadata is None"
            )
            return
        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        assert len(self.kv_caches) > 0

        kvcaches = list(self.kv_caches.values())
        if self.current_layer == 0:
            self.layerwise_storers = []

            is_first = True

            for idx, request in enumerate(connector_metadata.requests):
                save_spec = request.save_spec
                if save_spec is None or not save_spec.can_save:
                    continue

                token_ids = request.token_ids
                assert isinstance(token_ids, list)

                slot_mapping = request.slot_mapping
                assert isinstance(slot_mapping, torch.Tensor)
                assert len(slot_mapping) == len(token_ids)

                # TODO: have a pre-allocated buffer to hold the slot_mappings
                slot_mapping = slot_mapping.to(self.device)

                if self.kv_role == "kv_producer":
                    skip_leading_tokens = 0
                else:
                    skip_leading_tokens = save_spec.skip_leading_tokens

                    if skip_leading_tokens == len(token_ids):
                        continue  # skip this request
                    # Align to lmcache chunk size
                    skip_leading_tokens = (
                        skip_leading_tokens
                        // self._lmcache_chunk_size
                        * self._lmcache_chunk_size
                    )

                store_mask = torch.ones(len(token_ids), dtype=torch.bool)
                store_mask[:skip_leading_tokens] = False

                logger.info(
                    "Storing KV cache for %d out of %d tokens "
                    "(skip_leading_tokens=%d) for request %s",
                    len(token_ids) - skip_leading_tokens,
                    len(token_ids),
                    skip_leading_tokens,
                    request.req_id,
                )

                # TODO (Jiayi): need to make layerwise storing
                # compatible with disagg spec
                layerwise_storer = self.lmcache_engine.store_layer(
                    token_ids,
                    mask=store_mask,
                    kvcaches=kvcaches,
                    slot_mapping=slot_mapping,
                    offset=skip_leading_tokens,
                    sync=is_first,
                    req_id=request.req_id,
                )
                self.layerwise_storers.append(layerwise_storer)
                if is_first:
                    is_first = False

        for layerwise_storer in self.layerwise_storers:
            next(layerwise_storer)

        self.current_layer += 1

    @_lmcache_nvtx_annotate
    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""

        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, LMCacheConnectorMetadata)

        if self.kv_role == "kv_consumer":
            # Don't do save if the role is kv_consumer
            return

        # Skip CPU store when Sage is active — KV lives in GPU paged memory
        if self.enable_sage:
            # Process injection path completions.  reshape_and_cache_flash
            # already wrote corrected K,V to the paged cache during the
            # model forward; we just need to update counters and log.
            if self._fused_inject_active:
                for rid in self._fused_inject_active:
                    launch_t0 = self._sage_pending_blend_launch_t0.pop(rid, None)
                    wall_ms = (
                        (time.perf_counter() - launch_t0) * 1000.0
                        if launch_t0 is not None
                        else None
                    )
                    logger.info(
                        "[SAGE_FUSED] Injection completed request=%s "
                        "wall_ms=%s",
                        rid,
                        f"{wall_ms:.3f}" if wall_ms is not None else "n/a",
                    )
                    # Write blend TBT to shared file for benchmark
                    _tbt_file = os.environ.get("SAGE_BLEND_TBT_FILE")
                    if _tbt_file and wall_ms is not None:
                        try:
                            with open(_tbt_file, "a") as _f:
                                _f.write(f"{rid} {wall_ms:.3f}\n")
                        except Exception:
                            pass
                # Check for layerwise extraction result: the per-request
                # extract hooks saved each request's M replay tokens'
                # boundary state after that request's extract layer.
                # Format: dict[req_id -> {hidden_states, residual}].
                from vllm.forward_context import get_forward_context
                try:
                    _fwd_ctx = get_forward_context()
                    _extract = getattr(_fwd_ctx, 'sage_extract_result', None)
                except Exception:
                    _extract = None
                if _extract is not None:
                    for rid, payload in _extract.items():
                        _state = self._sage_incremental_states.get(rid)
                        if _state is not None and payload is not None:
                            _state.layerwise_saved_hidden = payload['hidden_states']
                            _state.layerwise_saved_residual = payload['residual']
                    _fwd_ctx.sage_extract_result = None

                self._fused_inject_active.clear()

                # Remove PyTorch hooks registered for injection/extraction.
                for h in self._sage_inject_hook_handles:
                    h.remove()
                self._sage_inject_hook_handles.clear()

                # Suffix rerecompute completion: when a suffix inject
                # finishes, mark the request as completed.
                _suffix_done = []
                for rid in list(self._sage_suffix_rerecompute_pending):
                    if (
                        rid not in self._fused_inject_pending
                        and rid not in self._fused_inject_active
                    ):
                        _suffix_done.append(rid)
                for rid in _suffix_done:
                    self._sage_suffix_rerecompute_pending.discard(rid)
                    self._sage_incremental_completed_requests.add(rid)
                    logger.info(
                        "[SAGE_SUFFIX_RERECOMPUTE_DONE] req=%s",
                        rid,
                    )

                # Per-step verification summary (cheap: read 6 ints/floats).
                _ctr = getattr(self, "_sage_last_inject_counters", None)
                if _ctr is not None:
                    logger.info(
                        "[SAGE_FUSED_VERIFY] entries=%d transitions=%d "
                        "hook_calls=%d hook_total_us=%.0f "
                        "injects=%d extracts=%d setup_us=%.0f",
                        _ctr.get("entries", 0),
                        _ctr.get("transitions", 0),
                        _ctr["transition_hook_calls"],
                        _ctr["transition_hook_total_us"],
                        _ctr["inject_count"],
                        _ctr["extract_count"],
                        _ctr.get("setup_us", 0.0),
                    )
                    self._sage_last_inject_counters = None

            return

        if self.use_layerwise:
            for layerwise_storer in self.layerwise_storers:
                next(layerwise_storer)

            # unpin the kv caches according to req_id
            for request in connector_metadata.requests:
                self.lmcache_engine.lookup_unpin(request.req_id)
            return

        assert len(self.kv_caches) > 0
        kvcaches = list(self.kv_caches.values())

        assert self.lmcache_engine is not None

        for request in connector_metadata.requests:
            # unpin the kv caches according to req_id
            self.lmcache_engine.lookup_unpin(request.req_id)

            save_spec = request.save_spec
            if (
                save_spec is None or not save_spec.can_save
            ) and self.kv_role != "kv_producer":
                continue

            token_ids = request.token_ids

            slot_mapping = request.slot_mapping
            assert isinstance(slot_mapping, torch.Tensor)
            assert len(slot_mapping) == len(token_ids)

            # TODO: have a pre-allocated buffer to hold the slot_mappings
            slot_mapping = slot_mapping.to(self.device)

            skip_leading_tokens = save_spec.skip_leading_tokens
            # shared storage disaggregation will not have a disagg_spec passed in
            if self.kv_role == "kv_producer" and request.disagg_spec:
                skip_leading_tokens = min(
                    skip_leading_tokens, request.disagg_spec.num_transferred_tokens
                )

            if skip_leading_tokens == len(token_ids):
                continue  # skip this request
            # Align to lmcache chunk size
            skip_leading_tokens = (
                skip_leading_tokens
                // self._lmcache_chunk_size
                * self._lmcache_chunk_size
            )

            store_mask = torch.ones(len(token_ids), dtype=torch.bool)
            store_mask[:skip_leading_tokens] = False

            logger.info(
                "Storing KV cache for %d out of %d tokens "
                "(skip_leading_tokens=%d) for request %s",
                len(token_ids) - skip_leading_tokens,
                len(token_ids),
                skip_leading_tokens,
                request.req_id,
            )

            is_last_prefill = request.is_last_prefill
            if is_last_prefill:
                if request.disagg_spec:
                    request.disagg_spec.is_last_prefill = True
            else:
                if not self.enable_blending:
                    token_len = len(token_ids)
                    aligned_token_len = (
                        token_len // self._lmcache_chunk_size * self._lmcache_chunk_size
                    )
                    token_ids = token_ids[:aligned_token_len]
                    store_mask = store_mask[:aligned_token_len]
                    slot_mapping = slot_mapping[:aligned_token_len]

            self.lmcache_engine.store(
                token_ids,
                mask=store_mask,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
                offset=skip_leading_tokens,
                transfer_spec=request.disagg_spec,
                request_configs=request.request_configs,
                req_id=request.req_id,
            )

            # Update skip_leading_tokens only on last rank to ensure
            # each PP stage stores its own KV cache
            if get_pp_group().is_last_rank:
                # NOTE(Jiayi): We assume all tokens are saved
                save_spec.skip_leading_tokens = len(token_ids)
                if request.disagg_spec:
                    request.disagg_spec.num_transferred_tokens = len(token_ids)

    @_lmcache_nvtx_annotate
    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        for req_id in finished_req_ids:
            self._cleanup_sage_incremental_state(req_id)
        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        invalid_blocks = self._invalid_block_ids.copy()
        self._invalid_block_ids.clear()
        return invalid_blocks

    @_lmcache_nvtx_annotate
    def shutdown(self):
        """Shutdown the connector by delegating to LMCacheManager."""
        logger.info("Starting LMCacheConnector shutdown...")
        self._manager.stop_services()

    ###################
    # Scheduler side APIs
    ####################

    @_lmcache_nvtx_annotate
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Optional[int]:
        """
        Check for external KV cache hit.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request

        Returns:
            the number of tokens that can be loaded from the
            external KV cache beyond what is already computed.
        """
        # Ignore DP attention mock requests
        if request.request_id.startswith("mock_req"):
            return 0
        # Sage: skip LMCache hit check entirely — KV is managed via
        # GPU-direct zero-copy, not CPU-based LMCache retrieval
        if os.environ.get("ENABLE_SAGE", "False").lower() == "true":
            return 0
        # to handle preempted requests, we want `get_num_new_matched_tokens` to be
        # idempotent under the condition that `update_state_after_alloc` is NOT called
        # then the two side-effects that must be idempotent are:
        # 1. lookup_client caches a result
        #     uncached in `update_state_after_alloc` if this request can be scheduled
        # 2. cache engine will pin the KV caches for the request
        #     unpinned in `wait_for_save` if this request can be scheduled
        if self.kv_role == "kv_producer" and not hasattr(
            self.lookup_client, "supports_producer_reuse"
        ):
            return 0

        req_id = request.request_id

        # lookup_client is always initialized for scheduler role
        assert self.lookup_client is not None

        if (
            num_external_hit_tokens := self.lookup_client.lookup_cache(lookup_id=req_id)
        ) != -1:
            # -1 means no result cached
            # None or int means ongoing (async) or cached result
            logger.debug(
                f"Found {num_external_hit_tokens} hit tokens for request"
                f" {req_id} in the lookup cache."
            )
        else:
            logger.debug(f"Looking up cache for the first time for request {req_id}!")
            self._requests_priority[req_id] = getattr(request, "priority", 0)

            # token_ids = request.prompt_token_ids
            # all token ids covers the preemption case
            token_ids = request.all_token_ids

            # If the request has multimodal hashes, apply them to the token ids
            mm_hashes, mm_positions = extract_mm_features(request)
            if mm_hashes and mm_positions:
                # TODO(Jiayi): Optimize this
                token_ids = torch.tensor(request.prompt_token_ids)
                apply_mm_hashes_to_token_ids(token_ids, mm_hashes, mm_positions)
                token_ids = token_ids.tolist()

            request_configs = extract_request_configs(request.sampling_params)
            if self.skip_last_n_tokens > 0:
                token_ids = token_ids[: -self.skip_last_n_tokens]

            num_external_hit_tokens = self.lookup_client.lookup(
                token_ids,
                lookup_id=req_id,
                request_configs=request_configs,
            )

        if num_external_hit_tokens is None:
            logger.debug(
                "Reqid: %s, Total tokens %d, LMCache hit tokens: None.",
                req_id,
                request.num_tokens,
            )
            return None

        # When prompt length is divisible by the block size and all
        # blocks are cached, we need to recompute the last token.
        # This will be removed in the future if vLLM's scheduler provides
        # a better support for this case.
        need_to_allocate = num_external_hit_tokens - num_computed_tokens

        # In, full-prompt-hit case, we need to recompute the last token
        if num_external_hit_tokens == request.num_tokens:
            need_to_allocate -= 1

        # DEBUG: Track request state and token counts
        req_status = getattr(request, 'status', 'UNKNOWN')
        num_computed = getattr(request, 'num_computed_tokens', 0)
        all_token_len = len(request.all_token_ids) if hasattr(request, 'all_token_ids') else -1
        prompt_token_len = len(request.prompt_token_ids) if hasattr(request, 'prompt_token_ids') else -1
        logger.debug(
            "Reqid: %s, Total tokens %d, LMCache hit tokens: %d, need to load: %d, "
            "status=%s, num_computed=%d, all_token_ids_len=%d, prompt_token_ids_len=%d",
            req_id,
            request.num_tokens,
            num_external_hit_tokens,
            need_to_allocate,
            req_status,
            num_computed,
            all_token_len,
            prompt_token_len,
        )

        self.load_specs[req_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            lmcache_cached_tokens=num_external_hit_tokens,
            can_load=False,
        )

        if need_to_allocate <= 0:
            return 0

        # TODO: Align to vLLM block size. Should test whether it can be removed
        # need_to_allocate = need_to_allocate // self._block_size * \
        #        self._block_size

        return need_to_allocate

    @_lmcache_nvtx_annotate
    def update_state_after_alloc(self, request: "Request", num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.

        For SharedStorageConnector, update _request_needs_load
        if the CacheManager this allocated blocks for us.
        """

        # Clear local status in lookup client when a new request is
        # successfully scheduled.
        assert self.lookup_client is not None
        req_status = getattr(request, 'status', 'UNKNOWN')
        logger.debug(
            "[DEBUG] update_state_after_alloc called for %s, status=%s, num_external_tokens=%d, clearing lookup cache",
            request.request_id,
            req_status,
            num_external_tokens,
        )
        self.lookup_client.clear_lookup_status(request.request_id)

        kv_transfer_params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )

        if kv_transfer_params is not None and "disagg_spec" in kv_transfer_params:
            req_disagg_spec = kv_transfer_params["disagg_spec"]

            receiver_id = req_disagg_spec["receiver_host"] + str(
                req_disagg_spec["receiver_init_port"]
            )

            disagg_spec = DisaggSpec(
                req_id=req_disagg_spec["req_id"],
                receiver_id=receiver_id,
                receiver_host=req_disagg_spec["receiver_host"],
                receiver_init_port=req_disagg_spec["receiver_init_port"],
                receiver_alloc_port=req_disagg_spec["receiver_alloc_port"],
            )

            tmp_disagg_tracker[request.request_id] = disagg_spec
        self._unfinished_requests[request.request_id] = request

        if request.request_id not in self.load_specs:
            # No KV tokens from external KV cache, return
            return

        if num_external_tokens == 0:
            # No need to load anything
            self.load_specs[request.request_id].can_load = False
            return

        recalc_last = (
            1
            if (
                self.load_specs[request.request_id].lmcache_cached_tokens
                == request.num_tokens
            )
            else 0
        )
        assert (
            num_external_tokens
            == self.load_specs[request.request_id].lmcache_cached_tokens
            - self.load_specs[request.request_id].vllm_cached_tokens
            - recalc_last
        ), (
            f"Mismatch in tokens to load: {num_external_tokens} vs "
            f"{self.load_specs[request.request_id].lmcache_cached_tokens} "
            "(tokens in lmcache) - "
            f"{self.load_specs[request.request_id].vllm_cached_tokens} "
            "(tokens in vllm) - "
            f"{recalc_last} "
            "(full lmcache hits subtracts last token to recalculate logits)"
            f" for request {request.request_id}"
        )

        self.load_specs[request.request_id].can_load = True

    @_lmcache_nvtx_annotate
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        """Attach the connector metadata to the request object.

        This function should NOT modify other fields in the scheduler_output
        except the `kv_connector_metadata` field.
        Also, calling this function will reset the state of the connector.

        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
        """

        force_skip_save = self.kv_role == "kv_consumer" or self.force_skip_save

        meta = LMCacheConnectorMetadata()

        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
            self._unfinished_requests.pop(finished_req_id, None)
            self._cleanup_sage_incremental_state(finished_req_id)

        # We should load KV for:
        # 1. new requests
        # 2. preempted requests (once per recovery)
        # can_load will only be True if `update_state_after_alloc` has been called
        # which only happens when vLLM's KV manager has space to receive KV from LMCache
        for request in scheduler_output.scheduled_new_reqs:
            # Ignore DP attention mock requests
            if request.req_id.startswith("mock_req"):
                continue
            # Sage: skip chunk requests entirely — they use GPU-direct
            # zero-copy block transfer and don't need LMCache storage,
            # lookup, or metadata bookkeeping.
            if "_chunk_" in request.req_id:
                logger.debug(
                    "[SAGE] Skipping build_connector_meta for "
                    "chunk request %s",
                    request.req_id,
                )
                continue
            load_spec = self.load_specs.pop(request.req_id, None)
            prompt_len = len(request.prompt_token_ids) if request.prompt_token_ids else 0
            sage_blocks_transferred = bool(
                getattr(request, "sage_blocks_transferred", False)
            )
            scheduled_tokens = int(
                scheduler_output.num_scheduled_tokens[request.req_id]
            )
            num_tokens_to_compute = (
                int(getattr(request, "num_computed_tokens", 0)) + scheduled_tokens
            )
            # For zero-copy parent requests, normalize pre-TTFT accounting so
            # request/computed/load-spec stay consistent at first decode.
            should_trigger_incremental = True
            if self._incremental_blend_strategy is not None:
                should_trigger_incremental = self._should_schedule_incremental_blend(
                    request.req_id,
                    create_if_missing=False,
                )
            if sage_blocks_transferred and prompt_len > 0:
                if num_tokens_to_compute != prompt_len:
                    logger.info(
                        "[SAGE_ZERO_COPY_ALIGN] req=%s align num_tokens_to_compute "
                        "from %d to prompt_len=%d (request_num_computed=%d scheduled=%d)",
                        request.req_id,
                        int(num_tokens_to_compute),
                        int(prompt_len),
                        int(getattr(request, "num_computed_tokens", 0)),
                        int(scheduled_tokens),
                    )
                    num_tokens_to_compute = prompt_len
                # NOTE: Do NOT modify request.num_computed_tokens here!
                # `request` is a NewRequestData in scheduler_output.scheduled_new_reqs.
                # Changing it here overwrites the value the worker receives, causing
                # the model to process a PAD token at position prompt_len instead of
                # the last prompt token at prompt_len-1. (Bug #6)

            if (
                sage_blocks_transferred
                and load_spec is None
                and should_trigger_incremental
            ):
                load_spec = LoadSpec(
                    vllm_cached_tokens=int(num_tokens_to_compute),
                    lmcache_cached_tokens=int(prompt_len),
                    can_load=True,  # Allow loading/blending
                )
                logger.info(
                    "[SAGE_ZERO_COPY] Created load_spec for %s: vllm_cached=%d, "
                    "lmcache_cached=%d",
                    request.req_id,
                    int(load_spec.vllm_cached_tokens),
                    int(load_spec.lmcache_cached_tokens),
                )

            lmcache_cached_tokens = (
                int(load_spec.lmcache_cached_tokens) if load_spec is not None else 0
            )
            request_priority = self._requests_priority.pop(request.req_id, 0)

            skip_save = force_skip_save or (
                self.config.priority_limit is not None
                and request_priority > self.config.priority_limit
            )

            request_tracker = RequestTracker.from_new_request(
                self.config,
                request,
                num_tokens_to_compute,
                lmcache_cached_tokens,
                skip_save,
            )
            self._request_trackers[request.req_id] = request_tracker
            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self._save_decode_cache,
                num_output_tokens=0,
            )
            if req_meta is not None:
                if sage_blocks_transferred:
                    req_meta.sage_blocks_transferred = True
                    # Also copy chunk boundaries and compacted tokens for GPU-direct blending
                    req_meta.sage_chunk_boundaries = getattr(
                        request, "sage_chunk_boundaries", None
                    )
                    req_meta.query_token_count = getattr(
                        request, "query_token_count", 0
                    )
                    logger.info(
                        f"[SAGE_ZERO_COPY] Request {request.req_id} marked for GPU-direct blending, "
                        f"chunk_boundaries={req_meta.sage_chunk_boundaries}"
                    )
                meta.add_request(req_meta)

        cached_reqs = scheduler_output.scheduled_cached_reqs

        # NOTE: For backward compatibility with vllm version < 0.9.2,
        # In the latest vllm version, the type of scheduled_cached_reqs has
        # changed from list to object `CachedRequestData`
        if isinstance(cached_reqs, list):
            for i, req in enumerate(cached_reqs):
                # Sage: skip chunk requests (never tracked)
                if "_chunk_" in req.req_id:
                    continue
                load_spec = self.load_specs.pop(req.req_id, None)
                lmcache_cached_tokens = 0
                vllm_cached_tokens = 0
                if load_spec is not None:
                    lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                    vllm_cached_tokens = load_spec.vllm_cached_tokens
                vllm_request = self._unfinished_requests.get(req.req_id)
                if (
                    load_spec is None
                    and vllm_request is not None
                    and getattr(vllm_request, "sage_blocks_transferred", False)
                    and self._should_schedule_incremental_blend(
                        req.req_id,
                        create_if_missing=False,
                    )
                ):
                    prompt_len = len(vllm_request.prompt_token_ids)
                    load_spec = LoadSpec(
                        vllm_cached_tokens=prompt_len,
                        lmcache_cached_tokens=prompt_len,
                        can_load=True,
                    )
                    lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                    vllm_cached_tokens = load_spec.vllm_cached_tokens
                request_tracker = self._request_trackers[req.req_id]

                # Pass all_token_ids for preempted requests to restore
                # token_ids correctly for chunk key computation
                all_token_ids = None
                if req.resumed_from_preemption:
                    vllm_request = self._unfinished_requests.get(req.req_id)
                    assert vllm_request is not None, (
                        f"Preempted request {req.req_id} not found "
                        "in _unfinished_requests"
                    )
                    all_token_ids = list(vllm_request.all_token_ids)

                request_tracker.update(
                    req.new_token_ids,
                    req.new_block_ids,
                    req.resumed_from_preemption,
                    lmcache_cached_tokens=lmcache_cached_tokens,
                    vllm_cached_tokens=vllm_cached_tokens,
                    all_token_ids=all_token_ids,
                )
                req_meta = ReqMeta.from_request_tracker(
                    request_tracker,
                    self._block_size,
                    self._lmcache_chunk_size,
                    load_spec=load_spec,
                    discard_partial_chunks=self._discard_partial_chunks,
                    save_decode_cache=self._save_decode_cache,
                    num_output_tokens=int(getattr(req, "num_output_tokens", 0)),
                )
                if req_meta is not None:
                    vllm_request = self._unfinished_requests.get(req.req_id)
                    if (
                        vllm_request is not None
                        and getattr(vllm_request, "sage_blocks_transferred", False)
                    ):
                        req_meta.sage_blocks_transferred = True
                        req_meta.sage_chunk_boundaries = getattr(
                            vllm_request, "sage_chunk_boundaries", None
                        )
                    meta.add_request(req_meta)
            return meta

        for i, req_id in enumerate(cached_reqs.req_ids):
            # Sage: skip chunk requests (never tracked)
            if "_chunk_" in req_id:
                continue
            request_tracker = self._request_trackers[req_id]
            num_new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # TODO: this is a dangerous reference to the request object inside vllm
            if request := self._unfinished_requests.get(req_id):
                num_current_tokens = request.num_computed_tokens
                new_token_ids = request.all_token_ids[
                    num_current_tokens : num_current_tokens + num_new_tokens
                ]
            else:
                raise ValueError(
                    f"Request {req_id} is not in _unfinished_requests, "
                    f"but it is scheduled to be cached"
                )
            new_block_ids = cached_reqs.new_block_ids[i]

            load_spec = self.load_specs.pop(req_id, None)
            lmcache_cached_tokens = 0
            vllm_cached_tokens = 0
            if load_spec is not None:
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                vllm_cached_tokens = load_spec.vllm_cached_tokens
            if (
                load_spec is None
                and getattr(request, "sage_blocks_transferred", False)
                and self._should_schedule_incremental_blend(
                    req_id,
                    create_if_missing=False,
                )
            ):
                prompt_len = len(request.prompt_token_ids)
                load_spec = LoadSpec(
                    vllm_cached_tokens=prompt_len,
                    lmcache_cached_tokens=prompt_len,
                    can_load=True,
                )
                lmcache_cached_tokens = load_spec.lmcache_cached_tokens
                vllm_cached_tokens = load_spec.vllm_cached_tokens

            # Handle both old and new versions of CachedRequestData
            if hasattr(cached_reqs, "resumed_req_ids"):
                # New version with resumed_req_ids
                preempted = req_id in cached_reqs.resumed_req_ids
            elif hasattr(cached_reqs, "resumed_from_preemption"):
                # Old version with resumed_from_preemption
                preempted = cached_reqs.resumed_from_preemption[i]
            else:
                # This case should not be reached with supported vLLM versions.
                # Raising an error is safer than assuming not preempted.
                raise AttributeError(
                    f"Unable to determine preemption status for request {req_id}. "
                    f"This might be due to an unsupported vLLM version."
                )
            if preempted:
                assert load_spec is not None, (
                    f"Request {req_id} is preempted but was not given a load spec"
                )
                # num_computed_tokens should be reset to 0 during preemption
                # and then set to the number of already cached tokens (maxxing
                # prefix caching and lmcache)
                # this assumption is crucial for the update() call of RequestTracker
                assert request.num_computed_tokens == max(
                    lmcache_cached_tokens, load_spec.vllm_cached_tokens
                ), (
                    f"Preempted request {req_id} has "
                    f"num_computed_tokens {request.num_computed_tokens} "
                    "but max(lmcache_cached_tokens, vllm_cached_tokens) = "
                    f"{max(lmcache_cached_tokens, vllm_cached_tokens)}"
                )

            # Pass all_token_ids for preempted requests to restore
            # token_ids correctly for chunk key computation
            all_token_ids = list(request.all_token_ids) if preempted else None

            request_tracker.update(
                new_token_ids,
                new_block_ids,
                preempted=preempted,
                lmcache_cached_tokens=lmcache_cached_tokens,
                vllm_cached_tokens=vllm_cached_tokens,
                all_token_ids=all_token_ids,
            )
            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._lmcache_chunk_size,
                load_spec=load_spec,
                discard_partial_chunks=self._discard_partial_chunks,
                save_decode_cache=self._save_decode_cache,
                num_output_tokens=(
                    int(cached_reqs.num_output_tokens[i])
                    if hasattr(cached_reqs, "num_output_tokens")
                    and i < len(cached_reqs.num_output_tokens)
                    else 0
                ),
            )
            if req_meta is not None:
                if getattr(request, "sage_blocks_transferred", False):
                    req_meta.sage_blocks_transferred = True
                    req_meta.sage_chunk_boundaries = getattr(
                        request, "sage_chunk_boundaries", None
                    )
                meta.add_request(req_meta)

        return meta

    @_lmcache_nvtx_annotate
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        # Cleanup if request was aborted
        if request.status == RequestStatus.FINISHED_ABORTED and self.async_loading:
            # Cancel any ongoing async lookup and prefetch tasks on workers
            lookup_id = request.request_id
            assert self.lookup_client is not None
            self.lookup_client.cancel_lookup(  # type: ignore[attr-defined]
                lookup_id
            )

        params = (
            request.kv_transfer_params
            if hasattr(request, "kv_transfer_params")
            else None
        )
        return_params = None

        # NOTE: Used to stream back the first token
        # for disagg prefill
        if params is not None and "ret_first_tok" in params:
            return_params = {
                "first_tok": request._output_token_ids[0],
            }

        return False, return_params

    @_lmcache_nvtx_annotate
    def get_kv_events(self) -> Iterable[CacheStoreEvent]:
        if self.lmcache_engine is not None:
            return self.lmcache_engine.get_kv_events()
        return []
