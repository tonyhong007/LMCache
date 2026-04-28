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

        self.enable_timing = os.getenv("SAGE_ENABLE_TIMING", "0").lower() == "1"
        # Selected token indices from the last process_qkv check layer call.
        # Persisted across metadata resets so the adapter can retrieve which
        # token positions were selected after blend_from_gpu completes.
        self._last_imp_indices: Optional[torch.Tensor] = None
        # Full ranking of all N tokens by diff_k score from the scoring step.
        # The incremental strategy slices this into per-decode-step batches
        # (e.g., tokens ranked 1-500 for step 1, 501-1000 for step 2, etc.).
        self._last_full_importance_ranking: Optional[torch.Tensor] = None
        # Layerwise boundary: M-row hidden states saved at end of pre-TTFT
        # pass (blend_from_gpu with save_boundary=True). Allows fused
        # injection to skip already-processed layers during decode.
        self._scoring_boundary_hidden: Optional[torch.Tensor] = None
        self._scoring_boundary_residual: Optional[torch.Tensor] = None
        self._scoring_boundary_layer: int = 0
        self._scoring_boundary_pos_to_row: Optional[torch.Tensor] = None

    def _get_candidate_indices(
        self,
        attn_mask: Optional[torch.Tensor],
        total_len: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if attn_mask is None:
            return None
        assert attn_mask.device == device, (
            "Blend candidate mask must be on the same device as tensors: "
            f"mask={attn_mask.device}, tensors={device}"
        )
        assert attn_mask.dtype == torch.bool, (
            "Blend candidate mask must have bool dtype: "
            f"got {attn_mask.dtype}"
        )
        assert attn_mask.ndim == 1 and attn_mask.numel() == total_len, (
            "Blend candidate mask must be 1D with one value per token: "
            f"shape={tuple(attn_mask.shape)}, expected=({total_len},)"
        )
        candidate_indices = torch.nonzero(attn_mask, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        return candidate_indices

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
        attn_mask = self.metadata.attn_mask
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

        # --- Magnet Stage II: pre-selected indices, slice + write-back per layer ---
        # At layers before check_layer: pass through unchanged (full N tokens).
        # At check_layer: slice q/k/v/residual to the selected K tokens.
        # At all layers: write recomputed K, V back into paged cache at
        # the selected positions.
        if self.metadata.magnet_preselected_indices is not None:
            top_indices = self.metadata.magnet_preselected_indices.to(
                device=q.device, dtype=torch.long,
            )
            if self.metadata.imp_indices is None:
                # Magnet preselection is independent of CacheBlend's
                # diff_k scoring — once indices are set, every layer
                # including layer 0 can be restricted to the selected
                # tokens. Waiting until check_layer wastes a full-context
                # attention pass at every layer below check_layer.
                # Env flag lets you revert to the old behaviour for
                # debugging: SAGE_MAGNET_SLICE_AT_CHECK_LAYER=1.
                _slice_at_check = (
                    os.environ.get(
                        "SAGE_MAGNET_SLICE_AT_CHECK_LAYER", "0"
                    ) == "1"
                )
                if (
                    _slice_at_check
                    and layer_id not in self.common_metadata.check_layers
                ):
                    return q, k, v, residual, attn_output, attn_metadata
                q = q[top_indices]
                k = k[top_indices]
                v = v[top_indices]
                residual = residual[top_indices]
                self.metadata.imp_indices = top_indices
                if self.metadata.positions is not None:
                    self.metadata.positions = self.metadata.positions[top_indices]
                attn_output = attn_output[:top_indices.shape[0]]
                attn_metadata.update_from_top_indices(top_indices)
                self._last_imp_indices = top_indices.detach().clone()
            # Write blended K, V back at every layer for selected tokens.
            old_k[self.metadata.imp_indices] = k
            old_v[self.metadata.imp_indices] = v
            return q, old_k, old_v, residual, attn_output, attn_metadata

        if layer_id in self.common_metadata.check_layers:
            total_len = q.shape[0]
            suffix_len = self.metadata.suffix_len
            ratio = self.metadata.step_recompute_ratio
            topk_num = 0
            if ratio is None:
                assert self.common_metadata.recomp_ratios is not None
                ratio = self.common_metadata.recomp_ratios[0]

            scoring_only = ratio <= 0.0
            # Fast path: 0% ratio, no suffix, not incremental → skip everything
            if scoring_only and suffix_len == 0 and not self.metadata.capture_full_ranking:
                self.metadata.imp_indices = None
                return q, k, v, residual, attn_output, attn_metadata
            # Sync cacheblend at 0% with suffix: skip scoring, just recompute suffix
            if scoring_only and suffix_len > 0 and not self.metadata.capture_full_ranking:
                non_suffix_len = total_len - suffix_len
                top_indices = torch.arange(
                    non_suffix_len, total_len, device=q.device,
                )
                k, v = k[top_indices], v[top_indices]
                q = q[top_indices]
                residual = residual[top_indices]
                self.metadata.imp_indices = top_indices
                self.metadata.positions = self.metadata.positions[top_indices]
                attn_output = attn_output[:top_indices.shape[0]]
                attn_metadata.update_from_top_indices(top_indices)
                self._last_imp_indices = top_indices.detach().clone()
                old_k[top_indices] = k
                old_v[top_indices] = v
                return q, old_k, old_v, residual, attn_output, attn_metadata

            diff_t0 = 0.0
            if timing_enabled:
                if q.is_cuda:
                    torch.cuda.synchronize()
                diff_t0 = time.perf_counter()
            candidate_indices = self._get_candidate_indices(
                attn_mask=attn_mask,
                total_len=total_len,
                device=q.device,
            )
            if candidate_indices is not None and candidate_indices.numel() == 0:
                self.metadata.imp_indices = None
                return q, k, v, residual, attn_output, attn_metadata

            # Compute diff_k only on non-suffix tokens; suffix is always
            # recomputed (matching original CacheBlend's approach).
            non_suffix_len = total_len - suffix_len
            diff_k = torch.sum(
                (k[:non_suffix_len].to(torch.float32)
                 - old_k[:non_suffix_len].to(torch.float32)) ** 2,
                dim=[1],
            )
            # ── TP fix: diff_k reduces over LOCAL KV heads only. At
            # TP > 1 each rank produces a different per-token L2 score
            # because each holds a different head shard. Without an
            # all-reduce, each rank's topk picks different indices →
            # Pass 2 slices Q/K/V at different rows on each rank →
            # incoherent cross-attention → accuracy collapse. SUM
            # across ranks gives every rank the same global score so
            # topk picks identical indices everywhere. Same fix as the
            # magnet path; topk is scale-invariant so no normalize
            # needed.
            try:
                from vllm.distributed import (
                    get_tensor_model_parallel_world_size, get_tp_group,
                )
                _diff_tp_world = get_tensor_model_parallel_world_size()
            except Exception:
                _diff_tp_world = 1
            if _diff_tp_world > 1:
                try:
                    import torch.distributed as _dist
                    _dist.all_reduce(
                        diff_k, op=_dist.ReduceOp.SUM,
                        group=get_tp_group().device_group,
                    )
                except Exception as _e:
                    logger.warning(
                        "[SAGE_TP_FIX] diff_k all_reduce failed (%s); "
                        "per-rank diff will diverge", _e,
                    )
            if self.metadata.capture_full_ranking:
                if candidate_indices is not None:
                    # Filter candidate_indices to non-suffix range
                    cand_mask = candidate_indices < non_suffix_len
                    ns_candidates = candidate_indices[cand_mask]
                    if ns_candidates.numel() > 0:
                        score_view = diff_k[ns_candidates]
                        ranked_local = torch.argsort(score_view, descending=True)
                        full_ranking = ns_candidates[ranked_local]
                    else:
                        full_ranking = torch.argsort(diff_k, descending=True)
                else:
                    full_ranking = torch.argsort(diff_k, descending=True)
                self._last_full_importance_ranking = full_ranking.detach().clone()

            if scoring_only:
                self.metadata.imp_indices = None
                return q, k, v, residual, attn_output, attn_metadata

            # Select topk from non-suffix tokens only
            base_topk_num = int(non_suffix_len * ratio)
            base_topk_num = max(base_topk_num, 1)
            if candidate_indices is not None:
                cand_mask = candidate_indices < non_suffix_len
                ns_candidates = candidate_indices[cand_mask]
                effective_len = int(ns_candidates.numel())
            else:
                ns_candidates = None
                effective_len = non_suffix_len
            topk_num = min(base_topk_num, effective_len)

            if ns_candidates is not None and ns_candidates.numel() > 0:
                top_local_indices = torch.topk(
                    diff_k[ns_candidates], k=topk_num
                ).indices
                top_indices = ns_candidates[top_local_indices]
            else:
                top_indices = torch.topk(diff_k, k=topk_num).indices

            # Append suffix when all layers are processed in this
            # pass (sync cacheblend or ptt=1.0). For incremental
            # (ptt<1.0), suffix is deferred to the final step to
            # avoid conflicts with the fused inject batch.
            if suffix_len > 0 and self.metadata.include_suffix:
                suffix_indices = torch.arange(
                    non_suffix_len, total_len,
                    device=top_indices.device,
                )
                top_indices = torch.cat([top_indices, suffix_indices])
            top_indices, _ = torch.sort(top_indices)

            k, v = k[top_indices], v[top_indices]
            q = q[top_indices]
            residual = residual[top_indices]

            self.metadata.imp_indices = top_indices
            self.metadata.positions = self.metadata.positions[top_indices]
            attn_output = attn_output[:len(top_indices)]
            attn_metadata.update_from_top_indices(top_indices)
            self._last_imp_indices = self.metadata.imp_indices.detach().clone()
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
        self.metadata.attn_mask = mask

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

        max_layers_to_process = int(kwargs.get("max_layers_to_process", self.num_layers))
        if max_layers_to_process <= 0:
            raise ValueError(
                "max_layers_to_process must be >= 1 for blend_layer_from_gpu"
            )
        max_layers_to_process = min(max_layers_to_process, self.num_layers)

        self.metadata.positions = None
        self.metadata.imp_indices = None

        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)

        layerwise_gpu_retriever = self.cache_engine.retrieve_layer_from_gpu(
            tokens, mask,
            **kwargs
        )

        next(layerwise_gpu_retriever)
        yield

        for layer_id in range(max_layers_to_process):
            next(layerwise_gpu_retriever)
            next(layerwise_model_executor)
            yield

        # Final cleanup (write last layer back and finalize)
        next(layerwise_gpu_retriever)

        self.metadata.clean()
        yield

    def blend_from_gpu(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        step_recompute_ratio: Optional[float] = None,
        capture_full_ranking: bool = False,
        save_boundary: bool = False,
        suffix_len: int = 0,
        include_suffix: bool = False,
        **kwargs,
    ):
        """
        Perform blending by reading KV cache directly from GPU paged memory.

        :param save_boundary: If True (layerwise only), save M-row boundary
            hidden states at end of pass for fused injection at the next
            layer group during decode.
        :param suffix_len: Number of suffix tokens (e.g., query) that should
            always be recomputed and excluded from diff_k selection.
        :param include_suffix: If True, append suffix tokens to M selection.
            Set True when all recomputation happens in this pass (no
            incremental steps follow). Set False when incremental steps
            will handle the suffix later.
        """
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).cuda()
        self.metadata.attn_mask = mask
        self.metadata.step_recompute_ratio = step_recompute_ratio
        self.metadata.capture_full_ranking = capture_full_ranking
        self.metadata.suffix_len = suffix_len
        self.metadata.include_suffix = include_suffix
        self.metadata.imp_indices = None
        self._last_imp_indices = None
        self._last_full_importance_ranking = None
        max_layers_to_process = int(
            kwargs.get("max_layers_to_process", self.num_layers)
        )
        if max_layers_to_process <= 0:
            raise ValueError("max_layers_to_process must be >= 1 in blend_from_gpu")
        max_layers_to_process = min(max_layers_to_process, self.num_layers)

        kwargs["max_layers_to_process"] = max_layers_to_process
        kwargs.pop("selected_token_indices", None)

        layerwise_blender = self.blend_layer_from_gpu(
            tokens,
            mask,
            **kwargs,
        )
        for _ in range(max_layers_to_process + 2):
            next(layerwise_blender)

        # Layerwise boundary: save M-row hidden states at end of pass so
        # fused injection can skip already-processed layers during decode.
        if capture_full_ranking and save_boundary:
            _bnd_h = getattr(self.layerwise_model, "_layerwise_out_hidden", None)
            _bnd_r = getattr(self.layerwise_model, "_layerwise_out_residual", None)
            if _bnd_h is not None and _bnd_r is not None:
                _N_total = int(tokens.shape[0])
                _imp = self._last_imp_indices
                _ranking = self._last_full_importance_ranking
                assert _ranking is not None and _ranking.numel() > 0, (
                    "capture_full_ranking=True but no ranking available"
                )
                _indices = (_imp if _imp is not None and _bnd_h.shape[0] == _imp.numel()
                            else _ranking)
                _indices = _indices.to(device=_bnd_h.device, dtype=torch.long)
                M = int(_indices.numel())
                self._scoring_boundary_hidden = _bnd_h.detach().clone()
                self._scoring_boundary_residual = _bnd_r.detach().clone()
                self._scoring_boundary_layer = max_layers_to_process
                _pos_to_row = torch.full(
                    (_N_total,), -1, dtype=torch.long, device=_bnd_h.device,
                )
                _pos_to_row[_indices] = torch.arange(M, device=_bnd_h.device)
                self._scoring_boundary_pos_to_row = _pos_to_row
