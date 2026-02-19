# SPDX-License-Identifier: Apache-2.0
# Standard
import os
import time
from typing import Optional, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.models.utils import infer_model_from_vllm
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)


class LMCBlender:
    """
    Cache-blender backend for LMCache.
    This backend uses the Blender implementation for efficient blending computation.
    """

    def __init__(
        self,
        cache_engine,
        gpu_connector,
        vllm_model,
        config: LMCacheEngineConfig,
    ):
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector

        enable_sparse = False
        if config.extra_config is not None:
            enable_sparse = config.extra_config.get("enable_sparse", False)

        self.layerwise_model = infer_model_from_vllm(vllm_model, self, enable_sparse)

        # TODO: remove this hardcode
        # self.num_layers = len(vllm_model.model.layers)
        # Use the extracted language model from layerwise_model instead of the original vllm_model
        # This handles multimodal models like PixtralForConditionalGeneration
        self.num_layers = len(self.layerwise_model.vllm_model.model.layers)

        # TODO(Jiayi): support threshold-based blending
        # TODO(Jiayi): support different ratios for different layers
        # TODO(Jiayi): support "skipping blending if hit too short"
        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=config.blend_check_layers,
            recomp_ratios=config.blend_recompute_ratios,
            thresholds=config.blend_thresholds,
        )

        # This will be set during the blending process
        self.metadata = LMCBlendMetadata(
            imp_indices=None,
            attn_mask=None,
            positions=None,
        )

        # Chunk boundaries for GPU-direct SAGE blending (set externally)
        self.chunk_boundaries: Optional[list[int]] = None
        
        # Flag indicating whether RoPE adjustment is needed during blending
        # When True: chunks were prefilled without position offsets, need to adjust RoPE
        # When False: chunks were prefilled with correct position offsets (optimization)
        # NOTE: RoPE adjustment is now done in gpu_connector.batched_from_gpu_to_buffer
        # using fused_rotary_emb for efficiency (similar to batched_to_gpu for CPU retrieval)
        self.needs_rope_adjustment: bool = False
        
        # Precomputed boundary indices for optimization (set by compute_layer)
        # When set, process_qkv can skip diff_k computation and use these indices directly
        self.precomputed_boundary_indices: Optional[torch.Tensor] = None
        self.enable_timing = os.getenv("SAGE_ENABLE_TIMING", "0").lower() == "1"

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
        attn_output: Optional[torch.Tensor],
        attn_metadata: LMCAttnMetadata,
    ):
        logger.debug(f"Blender is processing KV for layer {layer_id}")
        timing_enabled = self.enable_timing
        total_t0 = 0.0
        rotary_ms = 0.0
        diff_k_ms = 0.0
        if timing_enabled:
            if q.is_cuda:
                torch.cuda.synchronize()
            total_t0 = time.perf_counter()

        old_k, old_v = self.gpu_connector.get_kv(layer_id)

        if attn_output is None:
            attn_output = torch.empty(
                q.shape,
                dtype=q.dtype,
                device=q.device,
            )

        # perform positional encoding
        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64
            )
        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        attn_layer = layer.self_attn

        rotary_t0 = 0.0
        if timing_enabled:
            if q.is_cuda:
                torch.cuda.synchronize()
            rotary_t0 = time.perf_counter()
        q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)
        if timing_enabled:
            if q.is_cuda:
                torch.cuda.synchronize()
            rotary_ms = (time.perf_counter() - rotary_t0) * 1000.0
        if layer_id in self.common_metadata.check_layers:
            diff_t0 = 0.0
            if timing_enabled:
                if q.is_cuda:
                    torch.cuda.synchronize()
                diff_t0 = time.perf_counter()
            diff_k = torch.sum(
                (k.to(torch.float32) - old_k.to(torch.float32)) ** 2, dim=[1]
            )
            total_len = diff_k.shape[0]

            assert self.common_metadata.recomp_ratios is not None

            # TODO(Jiayi): remove `[0]` hardcode
            topk_num = int(total_len * self.common_metadata.recomp_ratios[0])
            topk_num = max(topk_num, 1)

            top_indices = torch.topk(diff_k, k=topk_num).indices
            top_indices, _ = torch.sort(top_indices)

            k, v = k[top_indices], v[top_indices]
            q = q[top_indices]
            residual = residual[top_indices]

            logger.debug(f"Number of indices picked: {len(top_indices)}")

            self.metadata.imp_indices = top_indices
            self.metadata.positions = self.metadata.positions[top_indices]
            attn_output = attn_output[:topk_num]

            attn_metadata.update_from_top_indices(top_indices)
            if timing_enabled:
                if q.is_cuda:
                    torch.cuda.synchronize()
                diff_k_ms = (time.perf_counter() - diff_t0) * 1000.0

        if timing_enabled:
            if q.is_cuda:
                torch.cuda.synchronize()
            total_ms = (time.perf_counter() - total_t0) * 1000.0
            logger.info(
                "[PROCESS_QKV_TIMING] Layer %d: total=%.3fms, rotary_emb=%.3fms, "
                "diff_k_topk=%.3fms",
                layer_id,
                total_ms,
                rotary_ms,
                diff_k_ms,
            )

        if self.metadata.imp_indices is not None:
            old_k[self.metadata.imp_indices] = k
            old_v[self.metadata.imp_indices] = v
            return q, old_k, old_v, residual, attn_output, attn_metadata
        else:
            return q, k, v, residual, attn_output, attn_metadata

    # NOTE(Jiayi): Exposing this `blend_layer` interface as we might
    # want to ochestrate the blending process elsewhere
    def blend_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform layerwiese retrieve + blending.
        """

        # TODO(Jiayi): store is currently not included in this function

        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)
        layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)

        next(layerwise_retriever)
        yield

        for layer_id in range(self.num_layers):
            if self.enable_timing:
                retrieve_start = torch.cuda.Event(enable_timing=True)
                retrieve_end = torch.cuda.Event(enable_timing=True)
                compute_start = torch.cuda.Event(enable_timing=True)
                compute_end = torch.cuda.Event(enable_timing=True)
                retrieve_start.record()
            next(layerwise_retriever)
            if self.enable_timing:
                retrieve_end.record()
                compute_start.record()
            next(layerwise_model_executor)
            if self.enable_timing:
                compute_end.record()
                torch.cuda.synchronize()
                logger.info(
                    "[BLEND_LAYER_TIMING] Layer %d: retrieve=%.3fms, compute=%.3fms",
                    layer_id,
                    retrieve_start.elapsed_time(retrieve_end),
                    compute_start.elapsed_time(compute_end),
                )
            yield

        next(layerwise_retriever)
        self.metadata.clean()
        yield

    def blend(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform blending for the given tokens.
        """
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).cuda()

        blend_start_event = None
        blend_end_event = None
        blend_wall_t0 = None
        if self.enable_timing:
            blend_start_event = torch.cuda.Event(enable_timing=True)
            blend_end_event = torch.cuda.Event(enable_timing=True)
            blend_start_event.record()
            blend_wall_t0 = time.perf_counter()

        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)

        for i in range(self.num_layers + 2):
            next(layerwise_blender)
        if self.enable_timing and blend_start_event is not None and blend_end_event is not None:
            assert blend_wall_t0 is not None
            blend_end_event.record()
            torch.cuda.synchronize()
            wall_time_ms = (time.perf_counter() - blend_wall_t0) * 1000.0
            gpu_time_ms = blend_start_event.elapsed_time(blend_end_event)
            logger.info(
                "[BLEND_TIMING] blend() wall_time=%.3fms, gpu_time=%.3fms for %d tokens",
                wall_time_ms,
                gpu_time_ms,
                int(tokens.shape[0]),
            )

    def blend_layer_from_gpu(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform layerwise blending by reading KV cache directly from GPU paged memory.
        This is optimized for Sage concurrent prefill where the KV cache is already
        computed and stored in GPU paged memory, avoiding CPU round-trip overhead.

        Unlike blend_layer which fetches from CPU storage via retrieve_layer, this
        uses retrieve_layer_from_gpu to read directly from GPU paged memory.

        The flow for each layer:
        1. Read layer i from paged memory into buffer (via retrieve_layer_from_gpu)
        2. Process layer i (model computes new Q,K,V; blending modifies buffer)
        3. Write layer i back to paged memory (happens on next iteration)
        """
        # TODO(Jiayi): store is currently not included in this function

        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)

        # Use cache_engine.retrieve_layer_from_gpu to read directly from GPU paged memory
        # This properly computes starts/ends from the token database
        # Pass chunk_boundaries and needs_rope_adjustment so RoPE adjustment can be done
        # in the gpu_connector (similar to how CPU version does it in batched_to_gpu)
        layerwise_gpu_retriever = self.cache_engine.retrieve_layer_from_gpu(
            tokens, mask, 
            chunk_boundaries=self.chunk_boundaries,
            needs_rope_adjustment=self.needs_rope_adjustment,
            **kwargs
        )

        next(layerwise_gpu_retriever)
        yield

        total_retrieve_ms = 0.0
        total_compute_ms = 0.0
        for layer_id in range(self.num_layers):
            if self.enable_timing:
                retrieve_start = torch.cuda.Event(enable_timing=True)
                retrieve_end = torch.cuda.Event(enable_timing=True)
                compute_start = torch.cuda.Event(enable_timing=True)
                compute_end = torch.cuda.Event(enable_timing=True)
                retrieve_start.record()
            next(layerwise_gpu_retriever)
            if self.enable_timing:
                retrieve_end.record()
                compute_start.record()
            next(layerwise_model_executor)
            if self.enable_timing:
                compute_end.record()
                torch.cuda.synchronize()
                retrieve_ms = retrieve_start.elapsed_time(retrieve_end)
                compute_ms = compute_start.elapsed_time(compute_end)
                total_retrieve_ms += retrieve_ms
                total_compute_ms += compute_ms
                logger.info(
                    "[BLEND_GPU_TIMING] Layer %d: retrieve_from_gpu=%.3fms, compute=%.3fms",
                    layer_id,
                    retrieve_ms,
                    compute_ms,
                )
            yield
        
        # Final cleanup (write last layer back and finalize)
        next(layerwise_gpu_retriever)

        self.metadata.clean()
        self.chunk_boundaries = None  # Clear chunk boundaries after blending
        self.needs_rope_adjustment = False  # Clear RoPE adjustment flag after blending
        if self.enable_timing:
            logger.info(
                "[BLEND_GPU_TIMING] SUMMARY: total_retrieve_from_gpu=%.3fms, total_compute=%.3fms",
                total_retrieve_ms,
                total_compute_ms,
            )
        yield

    def blend_from_gpu(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        chunk_boundaries: Optional[list[int]] = None,
        needs_rope_adjustment: bool = False,
        **kwargs,
    ):
        """
        Perform blending by reading KV cache directly from GPU paged memory.
        This is the main entry point for Sage concurrent prefill optimization.

        Use this when the KV cache has already been computed via concurrent prefill
        and is stored in GPU paged memory, avoiding the CPU memory round-trip.

        :param tokens: The tokens to blend.
        :param mask: Optional mask indicating which tokens need blending.
        :param chunk_boundaries: Optional list of chunk boundary positions (e.g., [0, 2770, 4615, ...]).
            When provided, positions at chunk boundaries are marked for recomputation
            instead of using diff_k which fails for SAGE due to intra-chunk attention.
        :param needs_rope_adjustment: Whether RoPE adjustment is needed for cached KV.
            When True: chunks were prefilled without position offsets, so cached KV has 
            incorrect RoPE that needs to be adjusted before blending.
            When False (default): chunks were prefilled with correct position offsets,
            so cached KV already has correct RoPE (optimization mode).
        :param kwargs: Must include 'kvcaches' and 'slot_mapping'.
        """
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).cuda()

        blend_start_event = None
        blend_end_event = None
        blend_wall_t0 = None
        if self.enable_timing:
            blend_start_event = torch.cuda.Event(enable_timing=True)
            blend_end_event = torch.cuda.Event(enable_timing=True)
            blend_start_event.record()
            blend_wall_t0 = time.perf_counter()

        # Set chunk boundaries for process_qkv to use
        self.chunk_boundaries = chunk_boundaries
        # Set flag for RoPE adjustment
        self.needs_rope_adjustment = needs_rope_adjustment
        if chunk_boundaries is not None:
            logger.debug(
                f"[SAGE_BLEND] Using explicit chunk_boundaries: {chunk_boundaries}, "
                f"needs_rope_adjustment={needs_rope_adjustment}"
            )

        layerwise_blender = self.blend_layer_from_gpu(tokens, mask, **kwargs)

        for i in range(self.num_layers + 2):
            next(layerwise_blender)
        if self.enable_timing and blend_start_event is not None and blend_end_event is not None:
            assert blend_wall_t0 is not None
            blend_end_event.record()
            torch.cuda.synchronize()
            wall_time_ms = (time.perf_counter() - blend_wall_t0) * 1000.0
            gpu_time_ms = blend_start_event.elapsed_time(blend_end_event)
            logger.info(
                "[BLEND_FROM_GPU_TIMING] blend_from_gpu() wall_time=%.3fms, gpu_time=%.3fms "
                "for %d tokens across %d layers",
                wall_time_ms,
                gpu_time_ms,
                int(tokens.shape[0]),
                self.num_layers,
            )
