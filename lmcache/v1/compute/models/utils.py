# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Dict

# Third Party
from torch import nn

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


def infer_model_from_vllm(vllm_model, blender, enable_sparse: bool = False):
    model_name = type(vllm_model).__name__

    # Handle multimodal models that wrap a language model
    # Check if the model has a get_language_model method (e.g., PixtralForConditionalGeneration)
    if hasattr(vllm_model, "get_language_model"):
        language_model = vllm_model.get_language_model()
        logger.info(
            f"Extracting language model from multimodal model {model_name}: "
            f"{type(language_model).__name__}"
        )
        # Recursively infer the language model
        return infer_model_from_vllm(language_model, blender, enable_sparse)

    if model_name == "LlamaForCausalLM":
        # First Party
        from lmcache.v1.compute.models.llama import LMCLlamaModel

        return LMCLlamaModel(vllm_model, blender, enable_sparse)
    elif model_name == "Qwen2ForCausalLM":
        # First Party
        from lmcache.v1.compute.models.llama import LMCLlamaModel

        return LMCLlamaModel(vllm_model, blender, enable_sparse)
    elif model_name == "Qwen3ForCausalLM":
        # First Party
        from lmcache.v1.compute.models.qwen3 import LMCQwen3Model

        return LMCQwen3Model(vllm_model, blender, enable_sparse)
    elif model_name == "Qwen3MoeForCausalLM":
        # Qwen3 MoE shares the same attention structure (q_norm/k_norm)
        from lmcache.v1.compute.models.qwen3 import LMCQwen3Model

        return LMCQwen3Model(vllm_model, blender, enable_sparse)
    else:
        # TODO(Jiayi): Add support for more models
        raise NotImplementedError(
            f"Model type {model_name} is not supported in LMCache."
        )


class VLLMModelTracker:
    _vllm_models: Dict[str, nn.Module] = {}

    @classmethod
    def register_model(
        cls,
        instance_id: str,
        vllm_model: nn.Module,
    ):
        """
        Register a vllm model by instance_id.
        """
        logger.info(f"Registering vllm model for {instance_id}")
        if instance_id not in cls._vllm_models:
            cls._vllm_models[instance_id] = vllm_model
        else:
            logger.warning(
                f"vllm model for {instance_id} already registered, doing nothing."
            )

    @classmethod
    def get_model(
        cls,
        instance_id: str,
    ) -> nn.Module:
        """
        Get the vllm model by instance_id.
        """
        if instance_id not in cls._vllm_models:
            raise ValueError(f"vllm model for {instance_id} not found.")
        return cls._vllm_models[instance_id]
