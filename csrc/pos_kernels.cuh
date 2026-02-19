// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>

void rotary_embedding_k_fused(const torch::Tensor& old_positions,
                              const torch::Tensor& new_positions,
                              torch::Tensor& key, int64_t head_size,
                              const torch::Tensor& cos_sin_cache, bool is_neox);

// In-place RoPE on paged memory (no intermediate buffer needed!)
void rotary_embedding_paged_inplace(
    const torch::Tensor& old_positions,    // [num_tokens]
    const torch::Tensor& new_positions,    // [num_tokens]
    const torch::Tensor& slot_mapping,     // [num_tokens]
    torch::Tensor& key_cache,              // Paged KV cache
    const torch::Tensor& cos_sin_cache,    // [max_position, rot_dim]
    int64_t head_size,
    bool is_neox,
    bool vllm_two_major);