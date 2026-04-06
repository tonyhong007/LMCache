// SPDX-License-Identifier: Apache-2.0

#include <torch/all.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>

void rotary_embedding_k_fused(const torch::Tensor& old_positions,
                              const torch::Tensor& new_positions,
                              torch::Tensor& key, int64_t head_size,
                              const torch::Tensor& cos_sin_cache, bool is_neox);

// In-place RoPE correction across all layers in a single kernel launch.
void rotary_embedding_paged_fused_multi_layer(
    const torch::Tensor& old_positions,
    const torch::Tensor& new_positions,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& key_cache_ptrs,   // [num_layers] int64 device ptrs
    const torch::Tensor& key_cache_ref,    // shape/dtype reference
    const torch::Tensor& cos_sin_cache,
    int64_t head_size,
    bool is_neox,
    bool vllm_two_major);