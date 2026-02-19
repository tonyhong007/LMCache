# # SPDX-License-Identifier: Apache-2.0
# # Standard
# from abc import ABC, abstractmethod

# # Third Party
# from torch import nn
# import torch

# # First Party
# from lmcache.v1.compute.attention.utils import infer_attn_backend_from_vllm
# from lmcache.v1.compute.positional_encoding import get_fused_rope_from_rotary_emb

# # TODO(Jiayi): A few things need to be tested/supported:
# # TP, PP, Multimodal

# # NOTE(Tony): Chunked prefill implemented in the code below, but only needed on very long sequence
# # lengths or else we OOM on 40GB A100
# class LMCBaseModel(nn.Module, ABC):
#     def __init__(
#         self,
#         vllm_model,
#         blender,
#         enable_sparse: bool = False,
#     ):
#         super().__init__()
#         self.vllm_model = vllm_model

#         self.num_layers = len(vllm_model.model.layers)

#         self.vllm_attn_layers = []
#         self.lmc_attn_layers = []
#         for i in range(self.num_layers):
#             vllm_attn = vllm_model.model.layers[i].self_attn.attn
#             self.vllm_attn_layers.append(vllm_attn)

#             self.lmc_attn_layers.append(
#                 infer_attn_backend_from_vllm(vllm_attn, enable_sparse)
#             )

#         # NOTE(Jiayi): better not to pass the blender in init
#         # if we want to make this LMCModel more general.
#         self.blender = blender

#         # Use the vLLM model's existing rotary_emb directly.
#         # This correctly handles all RoPE variants (including rope_scaling)
#         # since the cos_sin_cache is pre-computed by vLLM with the correct
#         # scaling factors for the full context window.
#         # TODO(Tony): Crucial and unrelated to chunked prefill
#         rotary_emb = vllm_model.model.layers[0].self_attn.rotary_emb
#         self.fused_rotary_emb = get_fused_rope_from_rotary_emb(rotary_emb)

#         # Chunk size for per-token ops to avoid OOM on long sequences.
#         # Uses the same batch size as vLLM's chunked prefill scheduler.
#         try:
#             from vllm.config import get_current_vllm_config
#             self._chunk_size = (
#                 get_current_vllm_config()
#                 .scheduler_config.max_num_batched_tokens
#             )
#         except Exception:
#             self._chunk_size = 32768

#     @abstractmethod
#     def _process_qkv(self, q, k, v, layer):
#         """Process QKV tensors. Model-specific implementation."""
#         pass

#     def compute_layer(
#         self,
#         input_ids: torch.Tensor,
#     ):
#         input_ids = input_ids.cuda()
#         hidden_states = self.vllm_model.embed_input_ids(input_ids)
#         residual = None

#         attn_output = None
#         chunk_size = self._chunk_size

#         # TODO(Jiayi): Need to build `attn_metadata` more elegantly.
#         attn_metadata = self.lmc_attn_layers[0].init_attn_metadata(
#             input_ids=input_ids,
#         )

#         for idx, layer in enumerate(
#             self.vllm_model.model.layers[
#                 self.vllm_model.model.start_layer : self.vllm_model.model.end_layer
#             ]
#         ):
#             num_tokens = hidden_states.shape[0]
#             qkv_split_sizes = [
#                 layer.self_attn.q_size,
#                 layer.self_attn.kv_size,
#                 layer.self_attn.kv_size,
#             ]

#             # --- Phase A: layernorm + QKV projection (per-token, chunked) ---
#             if num_tokens <= chunk_size:
#                 if residual is None:
#                     residual = hidden_states
#                     hidden_states = layer.input_layernorm(hidden_states)
#                 else:
#                     hidden_states, residual = layer.input_layernorm(
#                         hidden_states, residual
#                     )

#                 qkv, _ = layer.self_attn.qkv_proj(hidden_states)
#                 q, k, v = qkv.split(qkv_split_sizes, dim=-1)
#                 q, k, v = self._process_qkv(q, k, v, layer)
#             else:
#                 q_chunks, k_chunks, v_chunks, res_chunks = [], [], [], []
#                 for start in range(0, num_tokens, chunk_size):
#                     end = min(start + chunk_size, num_tokens)
#                     hs_c = hidden_states[start:end]

#                     if residual is None:
#                         res_c = hs_c
#                         hs_c = layer.input_layernorm(hs_c)
#                     else:
#                         hs_c, res_c = layer.input_layernorm(
#                             hs_c, residual[start:end]
#                         )

#                     qkv_c, _ = layer.self_attn.qkv_proj(hs_c)
#                     q_c, k_c, v_c = qkv_c.split(qkv_split_sizes, dim=-1)
#                     q_c, k_c, v_c = self._process_qkv(q_c, k_c, v_c, layer)

#                     q_chunks.append(q_c)
#                     k_chunks.append(k_c)
#                     v_chunks.append(v_c)
#                     res_chunks.append(res_c)

#                 q = torch.cat(q_chunks)
#                 k = torch.cat(k_chunks)
#                 v = torch.cat(v_chunks)
#                 residual = torch.cat(res_chunks)
#                 del q_chunks, k_chunks, v_chunks, res_chunks, hidden_states

#             # --- Blender + Attention (not chunked) ---
#             q, k, v, residual, attn_output, attn_metadata = (
#                 self.blender.process_qkv(
#                     q, k, v, residual, idx, attn_output, attn_metadata
#                 )
#             )

#             num_heads = self.vllm_attn_layers[idx].num_heads
#             num_kv_heads = self.vllm_attn_layers[idx].num_kv_heads
#             head_size = self.vllm_attn_layers[idx].head_size

#             q = q.view(-1, num_heads, head_size)
#             k = k.view(-1, num_kv_heads, head_size)
#             v = v.view(-1, num_kv_heads, head_size)
#             attn_output = attn_output.view(-1, num_heads, head_size)

#             attn_output = self.lmc_attn_layers[idx].forward_contiguous(
#                 q, k, v, attn_output, attn_metadata
#             )

#             attn_output = attn_output.view(-1, num_heads * head_size)
#             k = k.view(-1, num_kv_heads * head_size)
#             v = v.view(-1, num_kv_heads * head_size)

#             # --- Phase B: o_proj + layernorm + MLP (per-token, chunked) ---
#             num_tokens_post = attn_output.shape[0]
#             if num_tokens_post <= chunk_size:
#                 hidden_states, _ = layer.self_attn.o_proj(attn_output)
#                 hidden_states, residual = layer.post_attention_layernorm(
#                     hidden_states, residual
#                 )
#                 hidden_states = layer.mlp(hidden_states)
#             else:
#                 hs_chunks, res_chunks = [], []
#                 for start in range(0, num_tokens_post, chunk_size):
#                     end = min(start + chunk_size, num_tokens_post)

#                     hs_c, _ = layer.self_attn.o_proj(attn_output[start:end])
#                     hs_c, res_c = layer.post_attention_layernorm(
#                         hs_c, residual[start:end]
#                     )
#                     hs_c = layer.mlp(hs_c)

#                     hs_chunks.append(hs_c)
#                     res_chunks.append(res_c)

#                 hidden_states = torch.cat(hs_chunks)
#                 residual = torch.cat(res_chunks)
#                 del hs_chunks, res_chunks

#             yield

# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC, abstractmethod

# Third Party
from torch import nn
import torch

# First Party
from lmcache.v1.compute.attention.utils import infer_attn_backend_from_vllm
from lmcache.v1.compute.positional_encoding import get_fused_rope

# TODO(Jiayi): A few things need to be tested/supported:
# TP, PP, Multimodal


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
        hidden_states = self.vllm_model.embed_input_ids(input_ids)
        residual = None

        attn_output = None

        # TODO(Jiayi): Need to build `attn_metadata` more elegantly.
        attn_metadata = self.lmc_attn_layers[0].init_attn_metadata(
            input_ids=input_ids,
        )

        for idx, layer in enumerate(
            self.vllm_model.model.layers[
                self.vllm_model.model.start_layer : self.vllm_model.model.end_layer
            ]
        ):
            # TODO(Jiayi) The last layer doesn't have to be computed
            # hidden_states, residual = layer(positions, hidden_states, residual)

            # Self Attention
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)
            # hidden_states = self.self_attn(positions=positions,
            #                            hidden_states=hidden_states)

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

            hidden_states, _ = layer.self_attn.o_proj(attn_output)

            # Fully Connected
            hidden_states, residual = layer.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = layer.mlp(hidden_states)

            yield
