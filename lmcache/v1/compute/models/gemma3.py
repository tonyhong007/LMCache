# SPDX-License-Identifier: Apache-2.0
import torch
from lmcache.v1.compute.models.base import LMCBaseModel, logger


class LMCGemma3Model(LMCBaseModel):
    def _stage_i_post_attn(self, layer, attn_out, residual):
        """Gemma 3 post-attention: 4-norm structure."""
        hidden_states, _ = layer.self_attn.o_proj(attn_out)
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states, residual = layer.pre_feedforward_layernorm(
            hidden_states, residual,
        )
        hidden_states = layer.mlp(hidden_states)
        hidden_states = layer.post_feedforward_layernorm(hidden_states)
        return hidden_states, residual

    def _process_qkv(self, q, k, v, layer):
        """Process QKV tensors for Gemma 3 model with q_norm and k_norm."""
        q_by_head = q.view(
            *q.shape[:-1],
            q.shape[-1] // layer.self_attn.head_dim,
            layer.self_attn.head_dim,
        )
        q_by_head = layer.self_attn.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(
            *k.shape[:-1],
            k.shape[-1] // layer.self_attn.head_dim,
            layer.self_attn.head_dim,
        )
        k_by_head = layer.self_attn.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        return q, k, v

    @torch.compile
    def compute_layer(self, input_ids: torch.Tensor):
        """Gemma 3 compute_layer with 4-norm layer structure.

        Gemma 3 differs from Llama/Qwen:
          input_layernorm -> self_attn -> post_attention_layernorm (no residual)
          -> pre_feedforward_layernorm (with residual) -> mlp
          -> post_feedforward_layernorm (no residual)
        """
        input_ids = input_ids.cuda()
        hidden_states = self.vllm_model.embed_input_ids(input_ids)
        residual = None

        attn_output = None

        attn_metadata = self.lmc_attn_layers[0].init_attn_metadata(
            input_ids=input_ids,
        )
        layers = self.vllm_model.model.layers[
            self.vllm_model.model.start_layer : self.vllm_model.model.end_layer
        ]
        last_layer_idx = len(layers) - 1

        for idx, layer in enumerate(layers):
            # Self Attention
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(
                    hidden_states, residual
                )

            qkv, _ = layer.self_attn.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [
                    layer.self_attn.q_size,
                    layer.self_attn.kv_size,
                    layer.self_attn.kv_size,
                ],
                dim=-1,
            )

            # Model-specific QKV processing (q_norm, k_norm)
            q, k, v = self._process_qkv(q, k, v, layer)

            q, k, v, residual, attn_output, attn_metadata = (
                self.blender.process_qkv(
                    q, k, v, residual, idx, attn_output, attn_metadata
                )
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

            # Gemma 3: post_attention_layernorm WITHOUT residual connection
            hidden_states = layer.post_attention_layernorm(hidden_states)

            # Gemma 3: pre_feedforward_layernorm WITH residual connection
            hidden_states, residual = layer.pre_feedforward_layernorm(
                hidden_states, residual
            )

            # MLP
            hidden_states = layer.mlp(hidden_states)

            # Gemma 3: post_feedforward_layernorm WITHOUT residual connection
            hidden_states = layer.post_feedforward_layernorm(hidden_states)

            self._layerwise_out_hidden = hidden_states
            self._layerwise_out_residual = residual

            yield
