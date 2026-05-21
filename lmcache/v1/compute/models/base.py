# SPDX-License-Identifier: Apache-2.0
# Standard
import logging
import os
from abc import ABC, abstractmethod
from typing import Optional

# Third Party
from torch import nn
import torch

logger = logging.getLogger(__name__)

# First Party
from lmcache.v1.compute.attention.utils import infer_attn_backend_from_vllm
from lmcache.v1.compute.positional_encoding import get_fused_rope

# TODO(Jiayi): A few things need to be tested/supported:
# TP, PP, Multimodal


def _magnet_alpha_block(
    q_grouped: "torch.Tensor",
    k_ctx_heads: "torch.Tensor",
    scale: float,
    bf16_scoring: bool,
) -> "torch.Tensor":
    """Per-key α = mean over (heads, groups, queries) of the softmax
    attention probability. Returns values in [0, 1] — each row of the
    softmax sums to 1, so averaging across all three dims yields a
    probability-scale score independent of Q_s.

    Why mean and not sum over Q_s: top-k is invariant under sum/mean
    (monotonic), but threshold comparisons are not. Sum-over-Q_s makes
    the score range [0, Q_s], so a fixed threshold means different
    things when Q_s changes (pre-TTFT vs decode-time rescoring). Mean
    keeps the threshold Q_s-invariant and interpretable (above 1/N =
    above uniform attention). For top-k mode this is a pure refactor."""
    if not bf16_scoring:
        q_grouped = q_grouped.to(torch.float32)
        k_ctx_heads = k_ctx_heads.to(torch.float32)
    logits = torch.einsum(
        "qkgd,nkd->kgqn", q_grouped, k_ctx_heads,
    ) * scale
    probs = torch.softmax(logits.float(), dim=-1)
    return probs.sum(dim=2).mean(dim=(0, 1))

class LMCBaseModel(nn.Module, ABC):
    def __init__(
        self,
        vllm_model,
        blender,
        enable_sparse: bool = False,
    ):
        super().__init__()
        self.vllm_model = vllm_model

        self.num_layers = len(vllm_model.model.layers)

        self.vllm_attn_layers = []
        self.lmc_attn_layers = []
        for i in range(self.num_layers):
            vllm_attn = vllm_model.model.layers[i].self_attn.attn
            self.vllm_attn_layers.append(vllm_attn)

            self.lmc_attn_layers.append(
                infer_attn_backend_from_vllm(vllm_attn, enable_sparse)
            )

        # NOTE(Jiayi): better not to pass the blender in init
        # if we want to make this LMCModel more general.
        self.blender = blender

        # remove hard code
        rotary_emb = vllm_model.model.layers[0].self_attn.rotary_emb
        head_dim = rotary_emb.head_size
        max_position_embeddings = rotary_emb.max_position_embeddings
        rope_scaling = None
        base = rotary_emb.base
        is_neox_style = rotary_emb.is_neox_style
        dtype = rotary_emb.dtype
        self.fused_rotary_emb = get_fused_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=base,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )

    @abstractmethod
    def _process_qkv(self, q, k, v, layer):
        """Process QKV tensors. Model-specific implementation."""
        pass

    @torch.compile
    def compute_layer(
        self,
        input_ids: torch.Tensor,
    ):
        input_ids = input_ids.cuda()
        # Multimodal substitution side-channel: when SAGE blender is
        # rerunning the model on a multimodal prompt, the adapter sets
        # `_lmcache_blender_mm_embeds` and `_lmcache_blender_is_mm` on
        # the vllm_model so that vision_pad tokens get their proper
        # encoder embeddings instead of meaningless text vocab lookups.
        # Without this, recompute corrupts vision-token KV at any ratio
        # > 0 because vocab embeddings for `<|video_pad|>` are noise.
        mm_embeds = getattr(self.vllm_model, "_lmcache_blender_mm_embeds", None)
        is_mm = getattr(self.vllm_model, "_lmcache_blender_is_mm", None)
        outer = getattr(self.vllm_model, "_lmcache_outer_mm_model", None)
        if (
            outer is not None
            and mm_embeds is not None
            and is_mm is not None
        ):
            # Use the outer multimodal wrapper's embed_input_ids — this
            # handles both vision-token substitution and deepstack
            # feature injection (via _set_deepstack_input_embeds), which
            # the inner LM's embed_input_ids cannot do.
            mm_embeds_dev = mm_embeds.to(input_ids.device)
            is_mm_dev = is_mm.to(input_ids.device)
            hidden_states = outer.embed_input_ids(
                input_ids,
                multimodal_embeddings=(mm_embeds_dev,),
                is_multimodal=is_mm_dev,
            )
        else:
            hidden_states = self.vllm_model.embed_input_ids(input_ids)
        residual = None

        attn_output = None

        # TODO(Jiayi): Need to build `attn_metadata` more elegantly.
        attn_metadata = self.lmc_attn_layers[0].init_attn_metadata(
            input_ids=input_ids,
        )
        layers = self.vllm_model.model.layers[
            self.vllm_model.model.start_layer : self.vllm_model.model.end_layer
        ]
        last_layer_idx = len(layers) - 1

        # DEBUG: track hidden_states magnitudes at key layers + suffix
        # tokens to spot where chunked recompute diverges from baseline.
        _dbg = os.environ.get("LMC_DEBUG_HS", "0") == "1"
        if _dbg:
            n = hidden_states.shape[0]
            sfx_len = 16  # match probe's suffix_len
            sfx_start = max(0, n - sfx_len)
            logger.info(
                "[DBG_HS] L0_pre n=%d hs_mean_abs=%.4f sfx_mean_abs=%.4f "
                "outer=%s mm_set=%s",
                n,
                hidden_states.abs().mean().item(),
                hidden_states[sfx_start:].abs().mean().item(),
                "yes" if outer is not None else "no",
                "yes" if (mm_embeds is not None and is_mm is not None) else "no",
            )

        for idx, layer in enumerate(layers):
            # Self Attention
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)

            qkv, _ = layer.self_attn.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [
                    layer.self_attn.q_size,
                    layer.self_attn.kv_size,
                    layer.self_attn.kv_size,
                ],
                dim=-1,
            )

            # Model-specific QKV processing
            q, k, v = self._process_qkv(q, k, v, layer)

            q, k, v, residual, attn_output, attn_metadata = self.blender.process_qkv(
                q, k, v, residual, idx, attn_output, attn_metadata
            )

            num_heads = self.vllm_attn_layers[idx].num_heads
            num_kv_heads = self.vllm_attn_layers[idx].num_kv_heads
            head_size = self.vllm_attn_layers[idx].head_size

            q = q.view(-1, num_heads, head_size)
            k = k.view(-1, num_kv_heads, head_size)
            v = v.view(-1, num_kv_heads, head_size)
            attn_output = attn_output.view(-1, num_heads, head_size)

            attn_output = self.lmc_attn_layers[idx].forward_contiguous(
                q, k, v, attn_output, attn_metadata
            )

            attn_output = attn_output.view(-1, num_heads * head_size)
            k = k.view(-1, num_kv_heads * head_size)
            v = v.view(-1, num_kv_heads * head_size)

            if idx == last_layer_idx:
                yield
                continue

            hidden_states, _ = layer.self_attn.o_proj(attn_output)

            # Fully Connected
            hidden_states, residual = layer.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = layer.mlp(hidden_states)

            if _dbg and idx in (0, last_layer_idx // 2, last_layer_idx):
                n_dbg = hidden_states.shape[0]
                sfx_start_dbg = max(0, n_dbg - 16)
                logger.info(
                    "[DBG_HS] L%d_post hs_mean_abs=%.4f sfx_mean_abs=%.4f",
                    idx,
                    hidden_states.abs().mean().item(),
                    hidden_states[sfx_start_dbg:].abs().mean().item(),
                )

            self._layerwise_out_hidden = hidden_states
            self._layerwise_out_residual = residual

            yield

    def magnet_query_only_score(
        self,
        query_token_ids: "torch.Tensor",
        context_slot_mapping: "torch.Tensor",
        context_len: int,
        query_position_start: int,
        kvcaches: list,
        gpu_connector,
    ) -> "torch.Tensor":
        """ProphetKV Stage I: query-only scoring pass.

        Runs ONLY the query tokens through all L layers with attention
        against the cached context K'. Per-layer cost O(|Q_s| × s) for
        the Q_s @ K' matmul plus O(|Q_s| × d^2) for the forward, avoiding
        the O(s^2) self-attention cost of a full-context pass.

        Returns: fused importance score ᾱ(t) = (1/L) Σ α_l(t) of shape
        [context_len]. Caller applies threshold to select anchors.
        """
        import math
        import os
        import lmcache.c_ops as _lmc_ops
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        device = query_token_ids.device
        Q_s = int(query_token_ids.shape[0])
        layers = self.vllm_model.model.layers
        L = len(layers)

        # Embed query tokens.
        hidden_states = self.vllm_model.embed_input_ids(query_token_ids.cuda())
        residual = None

        # Query positions: context_len .. context_len + Q_s - 1
        q_positions = torch.arange(
            query_position_start,
            query_position_start + Q_s,
            device=device, dtype=torch.int64,
        )

        # Head config (constant across all layers).
        vimpl0 = self.vllm_attn_layers[0]
        num_heads = vimpl0.num_heads
        num_kv_heads = vimpl0.num_kv_heads
        head_dim = vimpl0.head_size
        group_size = num_heads // num_kv_heads
        scale = 1.0 / math.sqrt(head_dim)

        # Single fused scratch buffer: [2, context_len + Q_s, H_kv * D].
        # single_layer_kv_transfer writes the first `context_len` rows of
        # buf[0] (K) and buf[1] (V) from the paged cache. We then write
        # K_self / V_self into the tail rows [context_len:] directly, so
        # no second copy into a separate k_combined/v_combined is needed.
        kv0 = kvcaches[0]
        hidden_kv = num_kv_heads * head_dim
        combined_buf = torch.empty(
            2, context_len + Q_s, hidden_kv,
            dtype=kv0.dtype, device=device,
        )
        k_full = combined_buf[0]  # [N+Q_s, H_kv*D]
        v_full = combined_buf[1]
        cu_seqlens_q = torch.tensor([0, Q_s], device=device, dtype=torch.int32)
        cu_seqlens_k = torch.tensor(
            [0, context_len + Q_s], device=device, dtype=torch.int32,
        )

        # bf16 matmul for α scoring (softmax still in fp32 for stability).
        bf16_scoring = os.environ.get("SAGE_MAGNET_BF16_SCORING", "1") == "1"

        accumulated_alpha = None

        for idx, layer in enumerate(layers):
            # LayerNorm + residual.
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(
                    hidden_states, residual,
                )

            # QKV projection (only for query tokens).
            qkv, _bias = layer.self_attn.qkv_proj(hidden_states)
            q, k_self, v_self = qkv.split(
                [
                    layer.self_attn.q_size,
                    layer.self_attn.kv_size,
                    layer.self_attn.kv_size,
                ],
                dim=-1,
            )
            q, k_self, v_self = self._process_qkv(q, k_self, v_self, layer)

            # RoPE for query Q and its own K.
            q, k_self = layer.self_attn.rotary_emb(q_positions, q, k_self)

            # Read context K'/V' at this layer directly into the head of
            # the combined buffer. Only the first `context_len` rows of
            # buf[0] (K) and buf[1] (V) are written.
            kv = kvcaches[idx]
            _lmc_ops.single_layer_kv_transfer(
                combined_buf, kv, context_slot_mapping[:context_len],
                True, False, gpu_connector.vllm_two_major,
                gpu_connector.use_mla,
            )
            # Write K_self / V_self into the tail rows of the same buffer.
            k_full[context_len:].copy_(k_self.view(Q_s, hidden_kv))
            v_full[context_len:].copy_(v_self.view(Q_s, hidden_kv))

            # NOTE: K in paged cache already has absolute-position RoPE at
            # this point — the rope_prepass_and_remap runs BEFORE magnet
            # scoring. No RoPE correction needed here.

            # Compute α_l(t) via grouped einsum (skips GQA repeat_interleave).
            q_grouped = q.view(Q_s, num_kv_heads, group_size, head_dim)
            k_ctx_heads = k_full[:context_len].view(
                context_len, num_kv_heads, head_dim,
            )
            alpha_l = _magnet_alpha_block(
                q_grouped, k_ctx_heads, scale, bf16_scoring,
            )
            if accumulated_alpha is None:
                accumulated_alpha = alpha_l
            else:
                accumulated_alpha = accumulated_alpha + alpha_l

            # Forward through the rest of this layer via flash_attn_varlen,
            # reading K/V directly from the combined buffer (no extra copy).
            k_view = k_full.view(context_len + Q_s, num_kv_heads, head_dim)
            v_view = v_full.view(context_len + Q_s, num_kv_heads, head_dim)
            q_h = q.view(Q_s, num_heads, head_dim)

            attn_out = flash_attn_varlen_func(
                q=q_h,
                k=k_view,
                v=v_view,
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=Q_s,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_k=context_len + Q_s,
                softmax_scale=scale,
                causal=True,
            )
            attn_out = attn_out.reshape(Q_s, num_heads * head_dim).to(q.dtype)

            # O-proj + norm + MLP (model-specific structure).
            hidden_states, residual = self._stage_i_post_attn(
                layer, attn_out, residual,
            )

        fused = accumulated_alpha / float(L)
        return fused

    def _stage_i_post_attn(
        self,
        layer,
        attn_out: "torch.Tensor",
        residual: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Post-attention norm + MLP for Stage I scoring.

        Override in model-specific subclasses (e.g. Gemma 3's 4-norm).
        Returns (hidden_states, residual).
        """
        hidden_states, _ = layer.self_attn.o_proj(attn_out)
        hidden_states, residual = layer.post_attention_layernorm(
            hidden_states, residual,
        )
        hidden_states = layer.mlp(hidden_states)
        return hidden_states, residual

