# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

# Third Party
import torch
import time

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
        q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)

        if layer_id in self.common_metadata.check_layers:
            # For GPU-direct SAGE blending with chunk_boundaries, use explicit indices
            # instead of diff_k which fails due to intra-chunk attention mismatch
            if self.chunk_boundaries is not None and len(self.chunk_boundaries) > 1:
                total_len = k.shape[0]
                assert self.common_metadata.recomp_ratios is not None
                recomp_ratio = self.common_metadata.recomp_ratios[0]

                # Compute tokens to recompute around each chunk boundary
                # Distribute the recompute budget across chunk boundaries
                num_boundaries = (
                    len(self.chunk_boundaries) - 1
                )  # -1 because first boundary is 0
                tokens_per_boundary = max(
                    1, int(total_len * recomp_ratio / num_boundaries)
                )

                # Collect indices to recompute: tokens at and after each chunk boundary
                indices_to_recompute = set()
                for boundary in self.chunk_boundaries[
                    1:
                ]:  # Skip first boundary (position 0)
                    # Add tokens at and after the boundary
                    for offset in range(tokens_per_boundary):
                        pos = boundary + offset
                        if pos < total_len:
                            indices_to_recompute.add(pos)

                top_indices = torch.tensor(
                    sorted(indices_to_recompute), device=k.device, dtype=torch.long
                )
                topk_num = len(top_indices)

                logger.info(
                    f"[BLEND_DEBUG] Layer {layer_id}: Using chunk_boundaries={self.chunk_boundaries}, "
                    f"tokens_per_boundary={tokens_per_boundary}, total_recompute={topk_num}"
                )
            else:
                # Original diff_k based selection
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

                # DEBUG: Log blending decisions
                logger.info(
                    f"[BLEND_DEBUG] Layer {layer_id}: total_len={total_len}, "
                    f"topk_num={topk_num}, recomp_ratio={self.common_metadata.recomp_ratios[0]}"
                )
                logger.info(
                    f"[BLEND_DEBUG] diff_k stats: min={diff_k.min().item():.4f}, "
                    f"max={diff_k.max().item():.4f}, mean={diff_k.mean().item():.4f}"
                )
                logger.info(
                    f"[BLEND_DEBUG] top_indices range: [{top_indices.min().item()}, {top_indices.max().item()}]"
                )
                logger.info(
                    f"[BLEND_DEBUG] First 10 top_indices: {top_indices[:10].tolist()}"
                )
                logger.info(
                    f"[BLEND_DEBUG] old_k norm={old_k.norm().item():.4f}, new_k norm={k.norm().item():.4f}"
                )

            k, v = k[top_indices], v[top_indices]
            q = q[top_indices]
            residual = residual[top_indices]

            logger.debug(f"Number of indices picked: {len(top_indices)}")

            self.metadata.imp_indices = top_indices
            self.metadata.positions = self.metadata.positions[top_indices]
            attn_output = attn_output[:topk_num]

            attn_metadata.update_from_top_indices(top_indices)

        if self.metadata.imp_indices is not None:
            # Log buffer state before modification
            if layer_id == 0:
                logger.info(
                    f"[BLEND_DEBUG] Layer 0 before modification: "
                    f"old_k norm={old_k.norm().item():.4f}, old_v norm={old_v.norm().item():.4f}"
                )
            old_k[self.metadata.imp_indices] = k
            old_v[self.metadata.imp_indices] = v
            # Log buffer state after modification
            if layer_id == 0:
                logger.info(
                    f"[BLEND_DEBUG] Layer 0 after modification: "
                    f"old_k norm={old_k.norm().item():.4f}, old_v norm={old_v.norm().item():.4f}"
                )
            return q, old_k, old_v, residual, attn_output, attn_metadata
        else:
            if layer_id == 0:
                logger.info(
                    f"[BLEND_DEBUG] Layer 0: No imp_indices, returning new k/v. "
                    f"k norm={k.norm().item():.4f}, v norm={v.norm().item():.4f}"
                )
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

        # Timing tracking using CUDA events for accurate GPU timing
        blend_start_event = torch.cuda.Event(enable_timing=True)
        blend_end_event = torch.cuda.Event(enable_timing=True)
        blend_start_event.record()
        blend_layer_start_time = time.perf_counter()
        
        # Lists to store CUDA events for per-layer timing
        per_layer_retrieve_events = []  # (start, end) event pairs
        per_layer_compute_events = []   # (start, end) event pairs

        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)
        layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)

        # Init timing
        init_start = torch.cuda.Event(enable_timing=True)
        init_end = torch.cuda.Event(enable_timing=True)
        init_start.record()
        next(layerwise_retriever)
        init_end.record()
        yield

        for i in range(self.num_layers):
            # Time retrieve_layer step (CPU->GPU transfer, RoPE, etc.)
            retrieve_start = torch.cuda.Event(enable_timing=True)
            retrieve_end = torch.cuda.Event(enable_timing=True)
            retrieve_start.record()
            next(layerwise_retriever)
            retrieve_end.record()
            per_layer_retrieve_events.append((retrieve_start, retrieve_end))

            # Time compute_layer step (forward pass + blending)
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            compute_start.record()
            next(layerwise_model_executor)
            compute_end.record()
            per_layer_compute_events.append((compute_start, compute_end))
            yield

        # Final timing
        final_start = torch.cuda.Event(enable_timing=True)
        final_end = torch.cuda.Event(enable_timing=True)
        final_start.record()
        next(layerwise_retriever)
        final_end.record()

        self.metadata.clean()

        blend_end_event.record()
        
        # Synchronize once at the end to get all timings
        torch.cuda.synchronize()
        blend_layer_wall_time = time.perf_counter() - blend_layer_start_time
        
        # Extract GPU times from events (in milliseconds)
        retrieve_init_time = init_start.elapsed_time(init_end)
        retrieve_final_time = final_start.elapsed_time(final_end)
        blend_layer_gpu_time = blend_start_event.elapsed_time(blend_end_event)
        
        per_layer_retrieve_times = []
        per_layer_compute_times = []
        total_retrieve_time = retrieve_init_time + retrieve_final_time
        total_compute_time = 0.0
        
        for i, ((r_start, r_end), (c_start, c_end)) in enumerate(
            zip(per_layer_retrieve_events, per_layer_compute_events, strict=False)
        ):
            retrieve_time = r_start.elapsed_time(r_end)
            compute_time = c_start.elapsed_time(c_end)
            per_layer_retrieve_times.append(retrieve_time)
            per_layer_compute_times.append(compute_time)
            total_retrieve_time += retrieve_time
            total_compute_time += compute_time
            
            logger.info(
                f"[BLEND_LAYER_TIMING] Layer {i}: "
                f"retrieve={retrieve_time:.3f}ms, "
                f"compute={compute_time:.3f}ms, "
                f"layer_total={retrieve_time + compute_time:.3f}ms"
            )

        logger.info(f"[BLEND_LAYER_TIMING] retrieve_layer init: {retrieve_init_time:.3f}ms")
        logger.info(f"[BLEND_LAYER_TIMING] retrieve_layer final: {retrieve_final_time:.3f}ms")

        # Log summary statistics
        logger.info(
            f"[BLEND_LAYER_TIMING] SUMMARY: "
            f"wall_time={blend_layer_wall_time*1000:.3f}ms, "
            f"gpu_time={blend_layer_gpu_time:.3f}ms, "
            f"total_retrieve={total_retrieve_time:.3f}ms, "
            f"total_compute={total_compute_time:.3f}ms"
        )
        if per_layer_retrieve_times:
            max_retrieve = max(per_layer_retrieve_times)
            max_compute = max(per_layer_compute_times)
            min_retrieve = min(per_layer_retrieve_times)
            min_compute = min(per_layer_compute_times)
            max_retrieve_layer = per_layer_retrieve_times.index(max_retrieve)
            max_compute_layer = per_layer_compute_times.index(max_compute)
            avg_retrieve = sum(per_layer_retrieve_times) / len(per_layer_retrieve_times)
            avg_compute = sum(per_layer_compute_times) / len(per_layer_compute_times)
            
            logger.info(
                f"[BLEND_LAYER_TIMING] RETRIEVE STATS: "
                f"max={max_retrieve:.3f}ms (layer {max_retrieve_layer}), "
                f"min={min_retrieve:.3f}ms, "
                f"avg={avg_retrieve:.3f}ms"
            )
            logger.info(
                f"[BLEND_LAYER_TIMING] COMPUTE STATS: "
                f"max={max_compute:.3f}ms (layer {max_compute_layer}), "
                f"min={min_compute:.3f}ms, "
                f"avg={avg_compute:.3f}ms"
            )
            
            # Identify overall bottleneck
            all_stages = [
                ("retrieve_init", retrieve_init_time),
                ("retrieve_final", retrieve_final_time),
                (f"retrieve_layer_{max_retrieve_layer}", max_retrieve),
                (f"compute_layer_{max_compute_layer}", max_compute),
            ]
            bottleneck_name, bottleneck_time = max(all_stages, key=lambda x: x[1])
            logger.info(
                f"[BLEND_LAYER_TIMING] BOTTLENECK: "
                f"{bottleneck_name}={bottleneck_time:.3f}ms"
            )
            
            # Show which operation dominates per layer
            per_layer_dominance = []
            for i in range(len(per_layer_retrieve_times)):
                if per_layer_retrieve_times[i] > per_layer_compute_times[i]:
                    per_layer_dominance.append("retrieve")
                else:
                    per_layer_dominance.append("compute")
            retrieve_dominant_count = per_layer_dominance.count("retrieve")
            compute_dominant_count = per_layer_dominance.count("compute")
            logger.info(
                f"[BLEND_LAYER_TIMING] PER-LAYER DOMINANCE: "
                f"retrieve_dominant={retrieve_dominant_count} layers, "
                f"compute_dominant={compute_dominant_count} layers"
            )
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
        # Use CUDA events for accurate GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        blend_start_time = time.perf_counter()

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).cuda()

        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)

        for i in range(self.num_layers + 2):
            next(layerwise_blender)

        end_event.record()
        torch.cuda.synchronize()
        blend_wall_time = time.perf_counter() - blend_start_time
        blend_gpu_time = start_event.elapsed_time(end_event)
        logger.info(
            f"[BLEND_TIMING] blend() wall_time={blend_wall_time*1000:.3f}ms, "
            f"gpu_time={blend_gpu_time:.3f}ms "
            f"for {len(tokens)} tokens across {self.num_layers} layers"
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

        # Timing tracking using CUDA events for accurate GPU timing
        blend_start_event = torch.cuda.Event(enable_timing=True)
        blend_end_event = torch.cuda.Event(enable_timing=True)
        blend_start_event.record()
        blend_layer_start_time = time.perf_counter()
        
        # Lists to store CUDA events for per-layer timing
        per_layer_retrieve_events = []  # (start, end) event pairs
        per_layer_compute_events = []   # (start, end) event pairs

        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)

        # Use cache_engine.retrieve_layer_from_gpu to read directly from GPU paged memory
        # This properly computes starts/ends from the token database
        layerwise_gpu_retriever = self.cache_engine.retrieve_layer_from_gpu(
            tokens, mask, **kwargs
        )

        # Init timing
        init_start = torch.cuda.Event(enable_timing=True)
        init_end = torch.cuda.Event(enable_timing=True)
        init_start.record()
        next(layerwise_gpu_retriever)
        init_end.record()
        yield

        for i in range(self.num_layers):
            # Read layer i from paged GPU memory (and write layer i-1 back if i > 0)
            retrieve_start = torch.cuda.Event(enable_timing=True)
            retrieve_end = torch.cuda.Event(enable_timing=True)
            retrieve_start.record()
            next(layerwise_gpu_retriever)
            retrieve_end.record()
            per_layer_retrieve_events.append((retrieve_start, retrieve_end))

            # Process layer i (computes Q,K,V and blends)
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_end = torch.cuda.Event(enable_timing=True)
            compute_start.record()
            next(layerwise_model_executor)
            compute_end.record()
            per_layer_compute_events.append((compute_start, compute_end))
            yield
        
        # Final cleanup (write last layer back and finalize)
        final_start = torch.cuda.Event(enable_timing=True)
        final_end = torch.cuda.Event(enable_timing=True)
        final_start.record()
        next(layerwise_gpu_retriever)
        final_end.record()

        self.metadata.clean()
        self.chunk_boundaries = None  # Clear chunk boundaries after blending

        blend_end_event.record()
        
        # Synchronize once at the end to get all timings
        torch.cuda.synchronize()
        blend_layer_wall_time = time.perf_counter() - blend_layer_start_time
        
        # Extract GPU times from events (in milliseconds)
        retrieve_init_time = init_start.elapsed_time(init_end)
        retrieve_final_time = final_start.elapsed_time(final_end)
        blend_layer_gpu_time = blend_start_event.elapsed_time(blend_end_event)
        
        per_layer_retrieve_times = []
        per_layer_compute_times = []
        total_retrieve_time = retrieve_init_time + retrieve_final_time
        total_compute_time = 0.0
        
        for i, ((r_start, r_end), (c_start, c_end)) in enumerate(
            zip(per_layer_retrieve_events, per_layer_compute_events, strict=False)
        ):
            retrieve_time = r_start.elapsed_time(r_end)
            compute_time = c_start.elapsed_time(c_end)
            per_layer_retrieve_times.append(retrieve_time)
            per_layer_compute_times.append(compute_time)
            total_retrieve_time += retrieve_time
            total_compute_time += compute_time
            
            logger.info(
                f"[BLEND_GPU_TIMING] Layer {i}: "
                f"retrieve_from_gpu={retrieve_time:.3f}ms, "
                f"compute={compute_time:.3f}ms, "
                f"layer_total={retrieve_time + compute_time:.3f}ms"
            )

        logger.info(f"[BLEND_GPU_TIMING] retrieve_layer_from_gpu init: {retrieve_init_time:.3f}ms")
        logger.info(f"[BLEND_GPU_TIMING] retrieve_layer_from_gpu final: {retrieve_final_time:.3f}ms")

        # Log summary statistics
        logger.info(
            f"[BLEND_GPU_TIMING] SUMMARY: "
            f"wall_time={blend_layer_wall_time*1000:.3f}ms, "
            f"gpu_time={blend_layer_gpu_time:.3f}ms, "
            f"total_retrieve_from_gpu={total_retrieve_time:.3f}ms, "
            f"total_compute={total_compute_time:.3f}ms"
        )
        if per_layer_retrieve_times:
            max_retrieve = max(per_layer_retrieve_times)
            max_compute = max(per_layer_compute_times)
            min_retrieve = min(per_layer_retrieve_times)
            min_compute = min(per_layer_compute_times)
            max_retrieve_layer = per_layer_retrieve_times.index(max_retrieve)
            max_compute_layer = per_layer_compute_times.index(max_compute)
            avg_retrieve = sum(per_layer_retrieve_times) / len(per_layer_retrieve_times)
            avg_compute = sum(per_layer_compute_times) / len(per_layer_compute_times)
            
            logger.info(
                f"[BLEND_GPU_TIMING] RETRIEVE STATS: "
                f"max={max_retrieve:.3f}ms (layer {max_retrieve_layer}), "
                f"min={min_retrieve:.3f}ms, "
                f"avg={avg_retrieve:.3f}ms"
            )
            logger.info(
                f"[BLEND_GPU_TIMING] COMPUTE STATS: "
                f"max={max_compute:.3f}ms (layer {max_compute_layer}), "
                f"min={min_compute:.3f}ms, "
                f"avg={avg_compute:.3f}ms"
            )
            
            # Identify overall bottleneck
            all_stages = [
                ("retrieve_init", retrieve_init_time),
                ("retrieve_final", retrieve_final_time),
                (f"retrieve_layer_{max_retrieve_layer}", max_retrieve),
                (f"compute_layer_{max_compute_layer}", max_compute),
            ]
            bottleneck_name, bottleneck_time = max(all_stages, key=lambda x: x[1])
            logger.info(
                f"[BLEND_GPU_TIMING] BOTTLENECK: "
                f"{bottleneck_name}={bottleneck_time:.3f}ms"
            )
            
            # Show which operation dominates per layer
            per_layer_dominance = []
            for i in range(len(per_layer_retrieve_times)):
                if per_layer_retrieve_times[i] > per_layer_compute_times[i]:
                    per_layer_dominance.append("retrieve")
                else:
                    per_layer_dominance.append("compute")
            retrieve_dominant_count = per_layer_dominance.count("retrieve")
            compute_dominant_count = per_layer_dominance.count("compute")
            logger.info(
                f"[BLEND_GPU_TIMING] PER-LAYER DOMINANCE: "
                f"retrieve_dominant={retrieve_dominant_count} layers, "
                f"compute_dominant={compute_dominant_count} layers"
            )
        yield

    def blend_from_gpu(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        chunk_boundaries: Optional[list[int]] = None,
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
        :param kwargs: Must include 'kvcaches' and 'slot_mapping'.
        """
        # Use CUDA events for accurate GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        blend_from_gpu_start_time = time.perf_counter()

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).cuda()

        # Set chunk boundaries for process_qkv to use
        self.chunk_boundaries = chunk_boundaries
        if chunk_boundaries is not None:
            logger.info(
                f"[SAGE_BLEND] Using explicit chunk_boundaries: {chunk_boundaries}"
            )

        layerwise_blender = self.blend_layer_from_gpu(tokens, mask, **kwargs)

        for i in range(self.num_layers + 2):
            next(layerwise_blender)

        end_event.record()
        torch.cuda.synchronize()
        blend_from_gpu_wall_time = time.perf_counter() - blend_from_gpu_start_time
        blend_from_gpu_gpu_time = start_event.elapsed_time(end_event)
        logger.info(
            f"[BLEND_FROM_GPU_TIMING] blend_from_gpu() wall_time={blend_from_gpu_wall_time*1000:.3f}ms, "
            f"gpu_time={blend_from_gpu_gpu_time:.3f}ms "
            f"for {len(tokens)} tokens across {self.num_layers} layers"
        )
