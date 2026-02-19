# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import List, Optional, Tuple, Union
import abc
import os

# Third Party
import numpy as np
import torch
import time

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.compute.blend.utils import LMCBlenderBuilder
from lmcache.v1.memory_management import GPUMemoryAllocator  # noqa: E501
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.compute.positional_encoding import PagedRopeInplace

if torch.cuda.is_available():
    # First Party
    import lmcache.c_ops as lmc_ops

logger = init_logger(__name__)


class GPUConnectorInterface(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        # FIXME (Yihua): We shouldn't put start and end here since
        # it's not the responsibility of the GPUConnector to know
        # the token-sequence-related information.
        """Store the data in the memory object into a GPU buffer.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to be copied into GPU.
        :param int start: The starting index of the data in the corresponding
            token sequence.
        :param int end: The ending index of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        # FIXME (Yihua): We shouldn't put start and end here since
        # it's not the responsibility of the GPUConnector to know
        # the token-sequence-related information.
        """Load the data from a GPU buffer into the memory object.
        Sub-classes should define the format of the kwargs.

        :param MemoryObj memory_obj: The memory object to store the data from
            GPU.
        :param int start: The starting index of the data in the corresponding
            token sequence.
        :param int end: The ending index of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        Batched load the data from a GPU memory into the memory objects.
        Sub-classes should define the format of the kwargs.

        :param Union[List[List[MemoryObj]], List[MemoryObj]] memory_obj:
            The memory objects to store the data from GPU.
        :param List[int] starts: The starting indices of the data in the corresponding
            token sequence.
        :param List[int] ends: The ending indices of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def batched_to_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        Batched store the data from the memory objects to GPU kv cache.
        Sub-classes should define the format of the kwargs.

        :param Union[List[List[MemoryObj]], List[MemoryObj]] memory_obj:
            The memory objects to store the data to GPU.
        :param List[int] starts: The starting indices of the data in the corresponding
            token sequence.
        :param List[int] ends: The ending indices of the data in the corresponding
            token sequence.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_shape(self, num_tokens: int) -> torch.Size:
        """Get the shape of the data given the number of tokens."""
        raise NotImplementedError

    def initialize_kvcaches_ptr(self, **kwargs):
        """Initialize the kvcaches pointers if not already initialized."""
        if "kvcaches" in kwargs:
            self.kvcaches = kwargs["kvcaches"]


class VLLMPagedMemGPUConnectorV2(GPUConnectorInterface):
    """
    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_blocks, block_size, num_heads, head_size]

    It will produce / consume memory object with KV_2LTD format
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        If use_gpu is true, it will create a gpu intermediate buffer. In this
        case, it requires the following kwargs:
        - chunk_size: The MAX size of the chunk to be copied to GPU.
        - dtype: The data type of the intermediate buffer.
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.kv_cache_pointers = torch.empty(
            num_layers, dtype=torch.int64, device="cpu"
        )
        # Not sure we need a dict here. Maybe a single GPU connector always
        # works with a single device?
        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.kvcaches: Optional[List[torch.Tensor]] = None

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]
        if use_gpu:
            assert "chunk_size" in kwargs, (
                "chunk_size should be provided to create a GPU buffer."
            )
            assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
            assert "device" in kwargs, (
                "device should be provided to create a GPU buffer."
            )
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(
                shape, dtype=kwargs["dtype"], device=kwargs["device"]
            )

        self.store_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheEngineMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemGPUConnectorV2":
        """Create a connector from LMCacheEngineMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.

        Returns:
            A new instance of VLLMPagedMemGPUConnectorV2.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
        self.device = kv_caches[0].device
        assert self.device.type == "cuda", "The device should be CUDA."
        idx = self.device.index
        if idx in self.kv_cache_pointers_on_gpu:
            return self.kv_cache_pointers_on_gpu[idx]
        self.kv_cache_pointers.numpy()[:] = [t.data_ptr() for t in kv_caches]
        self.kv_cache_pointers_on_gpu[idx] = torch.empty(
            self.num_layers, dtype=torch.int64, device=self.device
        )
        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)
        if self.use_mla:
            # kv_caches[0].shape: [num_pages, page_size, head_size]
            assert kv_caches[0].dim() == 3
            self.page_buffer_size = kv_caches[0].shape[0] * kv_caches[0].shape[1]
        else:
            # kv_caches[0].shape: [2, num_pages, page_size, num_heads, head_size]
            assert kv_caches[0].dim() == 5
            self.page_buffer_size = kv_caches[0].shape[1] * kv_caches[0].shape[2]

        return self.kv_cache_pointers_on_gpu[idx]

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemGPUConnector"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemGPUConnector"
                )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        lmc_ops.multi_layer_kv_transfer(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start:end],
            self.device,
            self.page_buffer_size,
            False,
            self.use_mla,
        )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(self.kvcaches)

        with torch.cuda.stream(self.store_stream):
            if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
                lmc_ops.multi_layer_kv_transfer(
                    memory_obj.tensor,
                    kv_cache_pointers,
                    slot_mapping[start:end],
                    self.kvcaches[0].device,
                    self.page_buffer_size,
                    True,
                    self.use_mla,
                )
            else:
                # kvcaches -> gpu_buffer -> memobj
                assert self.gpu_buffer.device == self.kvcaches[0].device
                tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
                lmc_ops.multi_layer_kv_transfer(
                    tmp_gpu_buffer,
                    kv_cache_pointers,
                    slot_mapping[start:end],
                    self.kvcaches[0].device,
                    self.page_buffer_size,
                    True,
                    self.use_mla,
                )
                memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            self.store_stream.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        with torch.cuda.stream(self.load_stream):
            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
                self.to_gpu(memory_obj, start, end, **kwargs)
        self.load_stream.synchronize()

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])


class VLLMPagedMemGPUConnectorV3(GPUConnectorInterface):
    def __init__(
        self,
        metadata: LMCacheEngineMetadata,
        device: torch.device,
        use_gpu: bool = False,
    ):
        assert device.type == "cuda", "The device should be CUDA."
        self.metadata = metadata
        self.device = device
        self.use_mla = metadata.use_mla
        self.chunk_size = metadata.chunk_size
        self.use_gpu = use_gpu
        self.kvcaches: Optional[List[torch.Tensor]] = None
        self.page_buffer_size = 0

        self.init = False
        self.group_kv_cache_pointers_on_gpu: Optional[list[torch.Tensor]] = None
        self.group_tmp_buffer: Optional[list[torch.Tensor]] = None

        self.store_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheEngineMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemGPUConnectorV3":
        assert device is not None
        return cls(metadata, device, use_gpu)

    def _initialize_kv_cache_pointers(self):
        if self.init:
            return
        assert self.metadata.kv_layer_groups_manager.kv_layer_groups
        if self.use_gpu:
            # init tmp buffer
            tmp_buf_shapes = self.metadata.get_shapes(self.chunk_size)
            tmp_buf_dtypes = self.metadata.get_dtypes()
            assert len(tmp_buf_shapes) == len(tmp_buf_dtypes)
            self.group_tmp_buffer = [
                torch.empty(tmp_buf_shape, dtype=tmp_buf_dtype, device=self.device)
                for tmp_buf_shape, tmp_buf_dtype in zip(
                    tmp_buf_shapes, tmp_buf_dtypes, strict=True
                )
            ]
        self.group_kv_cache_pointers_on_gpu = []
        for group in self.metadata.kv_layer_groups_manager.kv_layer_groups:
            # init kv cache pointers
            num_layers = group.num_layers
            kv_cache_pointers = torch.empty(num_layers, dtype=torch.int64, device="cpu")
            kv_cache_pointers.numpy()[:] = [
                t.data_ptr()
                for i, t in enumerate(self.kvcaches)
                if i in group.layer_indices
            ]
            kv_cache_pointers_on_gpu = torch.empty(
                num_layers, dtype=torch.int64, device=self.device
            )
            kv_cache_pointers_on_gpu.copy_(kv_cache_pointers)
            self.group_kv_cache_pointers_on_gpu.append(kv_cache_pointers_on_gpu)

        if self.use_mla:
            # kvcaches[0].shape: [num_pages, page_size, head_size]
            assert self.kvcaches[0].dim() == 3
            self.page_buffer_size = (
                self.kvcaches[0].shape[0] * self.kvcaches[0].shape[1]
            )
        else:
            # kvcaches[0].shape: [2, num_pages, page_size, num_heads, head_size]
            assert self.kvcaches[0].dim() == 5
            self.page_buffer_size = (
                self.kvcaches[0].shape[1] * self.kvcaches[0].shape[2]
            )
        self.init = True
        logger.info("init kv cache pointers success in VLLMPagedMemGPUConnectorV3")

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.raw_tensor is not None
        assert "slot_mapping" in kwargs
        if self.use_mla:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT
        else:
            assert memory_obj.metadata.fmt == MemoryFormat.KV_2LTD

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None
        assert self.kvcaches[0].device == self.device
        self._initialize_kv_cache_pointers()
        assert self.group_kv_cache_pointers_on_gpu is not None
        for i, kv_cache_pointer in enumerate(self.group_kv_cache_pointers_on_gpu):
            memory_obj_tensor = memory_obj.get_tensor(i)
            assert memory_obj_tensor is not None
            lmc_ops.multi_layer_kv_transfer(
                memory_obj_tensor,
                kv_cache_pointer,
                slot_mapping[start:end],
                self.device,
                self.page_buffer_size,
                False,
                self.use_mla,
            )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        assert memory_obj.raw_tensor is not None
        assert "slot_mapping" in kwargs

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None
        assert self.kvcaches[0].device == self.device
        self._initialize_kv_cache_pointers()
        assert self.group_kv_cache_pointers_on_gpu is not None
        with torch.cuda.stream(self.store_stream):
            if not self.use_gpu or end - start != self.chunk_size:
                for i, kv_cache_pointer in enumerate(
                    self.group_kv_cache_pointers_on_gpu
                ):
                    memory_obj_tensor = memory_obj.get_tensor(i)
                    assert memory_obj_tensor is not None
                    lmc_ops.multi_layer_kv_transfer(
                        memory_obj_tensor,
                        kv_cache_pointer,
                        slot_mapping[start:end],
                        self.device,
                        self.page_buffer_size,
                        True,
                        self.use_mla,
                    )
            else:
                # kvcaches -> gpu_buffer -> memobj
                assert self.group_tmp_buffer is not None
                for i, kv_cache_pointer in enumerate(
                    self.group_kv_cache_pointers_on_gpu
                ):
                    tmp_gpu_buffer = self.group_tmp_buffer[i][:, :, : end - start, :]
                    lmc_ops.multi_layer_kv_transfer(
                        tmp_gpu_buffer,
                        kv_cache_pointer,
                        slot_mapping[start:end],
                        self.device,
                        self.page_buffer_size,
                        True,
                        self.use_mla,
                    )
                    memory_obj_tensor = memory_obj.get_tensor(i)
                    assert memory_obj_tensor is not None
                    memory_obj_tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.raw_tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            self.store_stream.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        with torch.cuda.stream(self.load_stream):
            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
                self.to_gpu(memory_obj, start, end, **kwargs)
        self.load_stream.synchronize()

    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        raise NotImplementedError


class VLLMBufferLayerwiseGPUConnector(GPUConnectorInterface):
    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        use_double_buffer: bool = True,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        self.kvcaches: Optional[List[torch.Tensor]] = None

        # TODO(Jiayi): remove this hardcode
        self.cache_positions = True

        self.fused_rotary_emb = None

        assert use_gpu, "use_gpu must be true in VLLMBufferLayerwiseGPUConnector"
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        assert "device" in kwargs, "device should be provided to create a GPU buffer."

        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]

        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]
        self.buffer_mapping: dict[int, MemoryObj] = {}

        # track gap positions between blended chunks
        self.current_gap_positions = None

        self.use_gpu = use_gpu
        self.gpu_buffer_allocator = None
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

        # Debug/timing switches for Sage paths.
        self.enable_gpu_timing = os.getenv("SAGE_ENABLE_TIMING", "0").lower() == "1"
        
        # Cached bulk buffer for batched_from_gpu_to_buffer (GPU-direct path)
        # Pre-allocated to avoid allocation overhead on each request
        self._cached_bulk_buffer: Optional[torch.Tensor] = None
        self._cached_bulk_buffer_tokens: int = 0  # Current capacity in tokens
        
        # In-place RoPE object for GPU-direct path (no intermediate buffer needed!)
        self._paged_rope_inplace: Optional[PagedRopeInplace] = None

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheEngineMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMBufferLayerwiseGPUConnector":
        """Create a connector from LMCacheEngineMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.

        Returns:
            A new instance of VLLMBufferLayerwiseGPUConnector.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")
            # NOTE (Jiayi): We use the first layer to determine the gpu buffer size.
            # NOTE (Jiayi): Using the exact number of tokens in the first layer
            # is okay since fragmentation shouldn't exist in the `gpu_buffer_allocator`
            # in layerwise mode.

            # flash attention: [num_layers, 2, num_blocks, block_size,
            # num_heads, head_size]
            # flash infer: [num_layers, num_blocks, 2, block_size, num_heads, head_size]
            assert kv_caches[0].shape[0] == 2 or kv_caches[0].shape[1] == 2, (
                "The kv_caches should have shape [num_layers, 2, num_blocks, "
                "block_size, num_heads, head_size] or "
                "[num_layers, num_blocks, 2, block_size, num_heads, head_size]"
            )

            self.vllm_two_major = kv_caches[0].shape[0] == 2

            if self.vllm_two_major:
                k_cache_shape_per_layer = kv_caches[0][0].shape
            else:
                k_cache_shape_per_layer = kv_caches[0][:, 0].shape
            max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]

            logger.info(f"Lazily initializing GPU buffer (max tokens={max_tokens}).")
            num_elements = k_cache_shape_per_layer.numel() * 2
            gpu_buffer_size = num_elements * self.element_size
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )
            
            # Pre-warm PagedRopeInplace to avoid lazy init overhead on first request
            self._prewarm_paged_rope_inplace(self.device)
    
    def _prewarm_paged_rope_inplace(self, device):
        """Pre-warm PagedRopeInplace and transfer cos_sin_cache to GPU.
        
        This avoids ~15-20ms overhead on first request due to:
        1. LMCBlenderBuilder.get() lazy initialization
        2. cos_sin_cache.to(device) CPU→GPU transfer (~10MB)
        """
        if self._paged_rope_inplace is not None:
            return  # Already initialized
            
        try:
            timing_enabled = self.enable_gpu_timing
            if timing_enabled:
                t0 = time.perf_counter()
            lmc_model = LMCBlenderBuilder.get(ENGINE_NAME).layerwise_model
            if lmc_model.fused_rotary_emb is not None:
                self._paged_rope_inplace = PagedRopeInplace(
                    lmc_model.fused_rotary_emb.rope,
                    lmc_model.fused_rotary_emb.is_neox_style,
                )
                # Pre-transfer cos_sin_cache to GPU
                cos_sin_cache = self._paged_rope_inplace.cos_sin_cache
                self._paged_rope_inplace._cos_sin_cache_device = cos_sin_cache.to(device)
                cache_size_mb = (
                    cos_sin_cache.numel() * cos_sin_cache.element_size() / 1024 / 1024
                )
                if timing_enabled:
                    t1 = time.perf_counter()
                    logger.info(
                        f"[GPU_DIRECT] Pre-warmed PagedRopeInplace in {(t1-t0)*1000:.2f}ms, "
                        f"cos_sin_cache={cache_size_mb:.1f}MB transferred to {device}"
                    )
                else:
                    logger.info(
                        f"[GPU_DIRECT] Pre-warmed PagedRopeInplace, "
                        f"cos_sin_cache={cache_size_mb:.1f}MB transferred to {device}"
                    )
        except Exception as e:
            logger.warning(f"[GPU_DIRECT] Failed to pre-warm PagedRopeInplace: {e}")

    def get_kv(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the KV cache for the given layer ID.
        This function is used to get the KV cache from the GPU buffer.
        """
        if layer_id not in self.buffer_mapping:
            raise ValueError(f"Layer {layer_id} is not loaded into GPU buffer.")

        gpu_buffer = self.buffer_mapping[layer_id].tensor
        assert gpu_buffer is not None
        return gpu_buffer[0], gpu_buffer[1]

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to buffer GPU memory. In each iteration i, it (1) loads the KV
        cache of layer i from CPU -> GPU buffer, (2) recovers the positional
        encoding of the layer i-1's KV cache in the GPU buffer, and (3)
        moves the KV cache of layer i-2 from GPU buffer to paged GPU memory.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if self.fused_rotary_emb is None and self.cache_positions:
            # TODO(Jiayi): Make this more elegant
            self.lmc_model = LMCBlenderBuilder.get(ENGINE_NAME).layerwise_model
            self.fused_rotary_emb = self.lmc_model.fused_rotary_emb

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        num_all_tokens = ends[-1] - starts[0]
        slot_mapping_full = slot_mapping[starts[0] : ends[-1]]

        # compute gap positions
        gap_mask = torch.ones(
            num_all_tokens, dtype=torch.bool, device=slot_mapping_full.device
        )
        buf_offset = starts[0]

        for start, end in zip(starts, ends, strict=False):
            gap_mask[start - buf_offset : end - buf_offset] = False

        self.current_gap_positions = torch.where(gap_mask)[0]

        buf_offset = starts[0]
        if self.cache_positions:
            new_positions_full = torch.arange(
                starts[0], ends[-1], dtype=torch.int64, device=self.kvcaches[0].device
            )

        buffer_shape = self.get_shape(num_all_tokens)
        assert self.gpu_buffer_allocator is not None
        compute_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        load_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        assert compute_gpu_buffer_obj is not None, (
            "Failed to allocate GPU buffer in GPUConnector"
        )
        assert load_gpu_buffer_obj is not None, (
            "Failed to allocate GPU buffer in GPUConnector"
        )
        assert compute_gpu_buffer_obj.tensor is not None
        assert load_gpu_buffer_obj.tensor is not None

        # current_stream = torch.cuda.current_stream()

        if self.cache_positions:
            old_positions_full = torch.zeros(
                (num_all_tokens,), dtype=torch.int64, device=self.kvcaches[0].device
            )

        per_layer_copy_events = []
        per_layer_rope_events = []
        per_layer_paged_events = []
        per_layer_gap_zero_events = []
        
        # Track the previous copy end event to measure pipeline overlap
        prev_copy_end_event = None
        
        for layer_id in range(self.num_layers + 2):
            # === DEFAULT STREAM WORK ===
            # Buffer -> paged memory transfer (for layer i-2)
            if layer_id > 1:
                if self.enable_gpu_timing:
                    paged_start_event = torch.cuda.Event(enable_timing=True)
                    paged_start_event.record()
                lmc_ops.single_layer_kv_transfer(
                    self.buffer_mapping[layer_id - 2].tensor,
                    self.kvcaches[layer_id - 2],
                    slot_mapping_full,
                    False,
                    False,  # shape is [2, num_tokens, hidden_dim]
                    self.vllm_two_major,
                    self.use_mla,
                )
                del self.buffer_mapping[layer_id - 2]
                if self.enable_gpu_timing:
                    paged_end_event = torch.cuda.Event(enable_timing=True)
                    paged_end_event.record()
                    per_layer_paged_events.append(
                        (layer_id - 2, paged_start_event, paged_end_event)
                    )

                logger.debug(f"Finished loading layer {layer_id - 2} into paged memory")

            if layer_id > 0 and layer_id <= self.num_layers:
                # Wait only for the corresponding copy stage instead of globally
                # synchronizing the device.
                if prev_copy_end_event is not None:
                    torch.cuda.current_stream().wait_event(prev_copy_end_event)

                # ping-pong the buffers
                compute_gpu_buffer_obj, load_gpu_buffer_obj = (
                    load_gpu_buffer_obj,
                    compute_gpu_buffer_obj,
                )

                if self.cache_positions:
                    assert compute_gpu_buffer_obj.tensor is not None

                    if self.enable_gpu_timing:
                        rope_start_event = torch.cuda.Event(enable_timing=True)
                        rope_start_event.record()
                    compute_gpu_buffer_obj.tensor[0] = self.fused_rotary_emb(
                        old_positions_full,
                        new_positions_full,
                        compute_gpu_buffer_obj.tensor[0],
                    )
                    if self.enable_gpu_timing:
                        rope_end_event = torch.cuda.Event(enable_timing=True)
                        rope_end_event.record()
                        per_layer_rope_events.append(
                            (layer_id - 1, rope_start_event, rope_end_event)
                        )

                # gap zeroing after RoPE
                if self.current_gap_positions.numel():
                    if self.enable_gpu_timing:
                        gap_start_event = torch.cuda.Event(enable_timing=True)
                        gap_start_event.record()
                    compute_gpu_buffer_obj.tensor[:, self.current_gap_positions] = 0.0
                    if self.enable_gpu_timing:
                        gap_end_event = torch.cuda.Event(enable_timing=True)
                        gap_end_event.record()
                        per_layer_gap_zero_events.append(
                            (layer_id - 1, gap_start_event, gap_end_event)
                        )

                self.buffer_mapping[layer_id - 1] = compute_gpu_buffer_obj

                logger.debug(f"Finished loading layer {layer_id - 1} into buffer")

            # === LOAD STREAM WORK (parallel) ===
            if layer_id < self.num_layers:
                memory_objs_layer = yield

                # memobj -> gpu_buffer (CPU -> GPU copy) - on load_stream
                copy_end_event = torch.cuda.Event(enable_timing=False)
                
                with torch.cuda.stream(self.load_stream):
                    if self.enable_gpu_timing:
                        copy_start_event = torch.cuda.Event(enable_timing=True)
                        copy_start_event.record(self.load_stream)
                    for start, end, memory_obj in zip(
                        starts, ends, memory_objs_layer, strict=False
                    ):
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_2TD
                        assert load_gpu_buffer_obj.tensor is not None
                        load_gpu_buffer_obj.tensor[0][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[0], non_blocking=True)

                        load_gpu_buffer_obj.tensor[1][
                            start - buf_offset : end - buf_offset
                        ].copy_(memory_obj.tensor[1], non_blocking=True)

                        if self.cache_positions and layer_id == 0:
                            old_positions_full[
                                start - buf_offset : end - buf_offset
                            ] = memory_obj.metadata.cached_positions
                    copy_end_event.record(self.load_stream)
                if self.enable_gpu_timing:
                    per_layer_copy_events.append(
                        (layer_id, copy_start_event, copy_end_event)
                    )
                prev_copy_end_event = copy_end_event

            elif layer_id == self.num_layers:
                yield

        if self.enable_gpu_timing:
            torch.cuda.synchronize()
            copy_times = {
                layer_id: start_evt.elapsed_time(end_evt)
                for layer_id, start_evt, end_evt in per_layer_copy_events
            }
            rope_times = {
                layer_id: start_evt.elapsed_time(end_evt)
                for layer_id, start_evt, end_evt in per_layer_rope_events
            }
            paged_times = {
                layer_id: start_evt.elapsed_time(end_evt)
                for layer_id, start_evt, end_evt in per_layer_paged_events
            }
            gap_zero_times = {
                layer_id: start_evt.elapsed_time(end_evt)
                for layer_id, start_evt, end_evt in per_layer_gap_zero_events
            }
            total_copy_time = sum(copy_times.values())
            total_rope_time = sum(rope_times.values())
            total_paged_time = sum(paged_times.values())
            total_gap_zero_time = sum(gap_zero_times.values())
            copy_bottleneck_count = 0
            default_stream_bottleneck_count = 0
            for layer_id in range(self.num_layers):
                copy_time = copy_times.get(layer_id, 0)
                rope_time = rope_times.get(layer_id, 0)
                if copy_time > rope_time:
                    copy_bottleneck_count += 1
                else:
                    default_stream_bottleneck_count += 1
            logger.info(
                f"[GPU_CONNECTOR_TIMING] batched_to_gpu SUMMARY: "
                f"total_copy(load_stream)={total_copy_time:.3f}ms, "
                f"total_rope(default_stream)={total_rope_time:.3f}ms, "
                f"total_paged(default_stream)={total_paged_time:.3f}ms, "
                f"total_gap_zero={total_gap_zero_time:.3f}ms"
            )
            logger.info(
                f"[GPU_CONNECTOR_TIMING] PIPELINE ANALYSIS: "
                f"copy_bottleneck={copy_bottleneck_count} layers, "
                f"rope_bottleneck={default_stream_bottleneck_count} layers"
            )

        # free the buffer memory
        load_gpu_buffer_obj.ref_count_down()
        compute_gpu_buffer_obj.ref_count_down()

        assert len(self.buffer_mapping) == 0, (
            "There are still layers in the buffer mapping after "
            "releasing the GPU buffers."
        )

        yield

    # TODO(Jiayi): Reduce repetitive operations in `batched_to_gpu`
    # and `batched_from_gpu`.
    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]], List[MemoryObj]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'kvcaches' is not provided in kwargs.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        buf_start = 0
        slot_mapping_chunks = []
        buf_starts_ends = []
        old_positions_chunks = []
        for start, end in zip(starts, ends, strict=False):
            buf_end = buf_start + end - start
            buf_starts_ends.append((buf_start, buf_end))
            slot_mapping_chunks.append(slot_mapping[start:end])
            buf_start = buf_end
            if self.cache_positions:
                old_positions_chunks.append(
                    torch.arange(
                        start, end, device=self.kvcaches[0].device, dtype=torch.int64
                    )
                )

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)
        buffer_shape = self.get_shape(num_tokens)
        assert self.gpu_buffer_allocator is not None
        tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        assert tmp_gpu_buffer_obj is not None, (
            "Failed to allocate GPU buffer in GPUConnector"
        )
        assert tmp_gpu_buffer_obj.tensor is not None

        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)
                lmc_ops.single_layer_kv_transfer(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[layer_id],
                    slot_mapping_full,
                    True,
                    False,  # shape is [2, num_tokens, hidden_dim]
                    self.vllm_two_major,
                    self.use_mla,
                )
                for (buf_start, buf_end), memory_obj, old_positions in zip(
                    buf_starts_ends,
                    memory_objs_layer,
                    old_positions_chunks,
                    strict=False,
                ):
                    assert memory_obj.tensor is not None
                    memory_obj.tensor[0].copy_(
                        tmp_gpu_buffer_obj.tensor[0][buf_start:buf_end],
                        non_blocking=True,
                    )
                    memory_obj.tensor[1].copy_(
                        tmp_gpu_buffer_obj.tensor[1][buf_start:buf_end],
                        non_blocking=True,
                    )
                    if self.cache_positions:
                        memory_obj.metadata.cached_positions = old_positions

            yield
            self.store_stream.synchronize()
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        tmp_gpu_buffer_obj.ref_count_down()
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, num_tokens, self.hidden_dim_size])

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def batched_from_gpu_to_buffer(
        self,
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        Read KV cache directly from paged GPU memory into the internal buffer,
        apply RoPE adjustment, allow blending to modify it, then write back to
        paged memory.
        This is optimized for Sage concurrent prefill where the KV cache is
        already computed and stored in GPU paged memory, avoiding CPU round-trip.

        Unlike batched_from_gpu which copies to CPU memory objects, this method
        keeps the data in GPU buffer and makes it accessible via get_kv().

        The flow for each layer is:
        1. Read layer i from paged memory into buffer
        2. Apply fused RoPE adjustment (if needs_rope_adjustment is True)
        3. Yield (allows blending via process_qkv which modifies buffer in place)
        4. On next iteration, write layer i back to paged memory before reading i+1

        This is a generator that yields num_layers + 2 times (like batched_to_gpu).
        First yield prepares metadata, subsequent yields read/write each layer.

        :param starts: The starting indices of the KV cache chunks.
        :param ends: The ending indices of the KV cache chunks.
        :param kvcaches: The paged GPU memory KV caches.
        :param slot_mapping: The slot mapping for the tokens.
        :param chunk_boundaries: List of chunk start positions for RoPE adjustment.
        :param needs_rope_adjustment: Whether RoPE adjustment is needed.
        """
        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        # Get RoPE adjustment parameters from kwargs
        chunk_boundaries = kwargs.get("chunk_boundaries", None)
        needs_rope_adjustment = kwargs.get("needs_rope_adjustment", False)

        self._lazy_initialize_buffer(self.kvcaches)

        num_all_tokens = ends[-1] - starts[0]
        slot_mapping_full = slot_mapping[starts[0] : ends[-1]]

        logger.debug(
            "[GPU_DIRECT] batched_from_gpu_to_buffer: tokens=%d, "
            "chunks=%d, needs_rope=%s",
            num_all_tokens,
            len(chunk_boundaries) if chunk_boundaries else 0,
            needs_rope_adjustment,
        )

        # Recompute slot_mapping to account for chunk boundaries.
        # Each chunk stored KV at positions 0..chunk_len-1 within its own blocks.
        # Keep this on-device to avoid GPU->CPU sync and host round-trips.
        pos_in_chunk = None
        if chunk_boundaries is not None and len(chunk_boundaries) > 1:
            kv_cache_config = kwargs.get("kv_cache_config")
            if kv_cache_config is not None:
                block_size = kv_cache_config.block_size
            elif self.use_mla:
                block_size = self.kvcaches[0].shape[1]
            else:
                block_size = self.kvcaches[0].shape[2]
            device = slot_mapping_full.device

            total_blocks = (num_all_tokens + block_size - 1) // block_size
            if total_blocks > 0:
                slot_mapping_i64 = slot_mapping_full.to(dtype=torch.int64)
                block_start_positions = (
                    torch.arange(total_blocks, device=device, dtype=torch.int64)
                    * block_size
                )
                block_ids = torch.div(
                    slot_mapping_i64[block_start_positions],
                    block_size,
                    rounding_mode="floor",
                )

                boundaries = torch.tensor(
                    chunk_boundaries + [num_all_tokens],
                    device=device,
                    dtype=torch.int64,
                )
                chunk_lens = boundaries[1:] - boundaries[:-1]
                blocks_per_chunk = torch.div(
                    chunk_lens + block_size - 1,
                    block_size,
                    rounding_mode="floor",
                )
                cumulative_blocks = torch.zeros(
                    boundaries.shape[0], device=device, dtype=torch.int64
                )
                cumulative_blocks[1:] = torch.cumsum(blocks_per_chunk, dim=0)

                token_positions = torch.arange(
                    num_all_tokens, device=device, dtype=torch.int64
                )
                chunk_id = torch.bucketize(
                    token_positions, boundaries[1:], right=True
                )
                chunk_starts = boundaries[chunk_id]
                pos_in_chunk = token_positions - chunk_starts

                chunk_local_block = torch.div(
                    pos_in_chunk, block_size, rounding_mode="floor"
                )
                parent_block_index = cumulative_blocks[chunk_id] + chunk_local_block
                parent_block_index.clamp_(0, total_blocks - 1)
                actual_block_id = block_ids[parent_block_index]
                pos_in_block = torch.remainder(pos_in_chunk, block_size)
                slot_mapping_full = actual_block_id * block_size + pos_in_block

        # =============================================================
        # IN-PLACE ROPE: Apply RoPE directly on paged memory
        # =============================================================

        # Compute old_positions and new_positions for RoPE adjustment
        old_positions_full = None
        new_positions_full = None
        if needs_rope_adjustment and chunk_boundaries is not None:
            device = self.kvcaches[0].device

            # old_positions: position within each chunk (reuse pos_in_chunk if available)
            if pos_in_chunk is not None:
                old_positions_full = pos_in_chunk  # Already int64 on GPU
            else:
                boundaries = torch.tensor(
                    chunk_boundaries + [num_all_tokens],
                    device=device,
                    dtype=torch.int64,
                )
                token_positions = torch.arange(
                    num_all_tokens, device=device, dtype=torch.int64
                )
                chunk_id = torch.bucketize(
                    token_positions, boundaries[1:], right=True
                )
                old_positions_full = token_positions - boundaries[chunk_id]

            # new_positions: absolute positions
            new_positions_full = torch.arange(
                starts[0], ends[-1], device=device, dtype=torch.int64
            )
        elif needs_rope_adjustment:
            logger.error("[GPU_DIRECT] needs_rope_adjustment=True but chunk_boundaries is None")
            needs_rope_adjustment = False

        # Initialize in-place RoPE if needed
        if needs_rope_adjustment and self._paged_rope_inplace is None:
            logger.warning("[GPU_DIRECT] PagedRopeInplace not pre-warmed, doing lazy init")
            self._prewarm_paged_rope_inplace(self.kvcaches[0].device)
            if self._paged_rope_inplace is None:
                logger.error("[GPU_DIRECT] Failed to initialize PagedRopeInplace!")
                needs_rope_adjustment = False

        # Allocate single-layer buffer
        buffer_shape = self.get_shape(num_all_tokens)
        assert self.gpu_buffer_allocator is not None
        gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
            buffer_shape, self.dtype, MemoryFormat.KV_2TD
        )
        assert gpu_buffer_obj is not None, "Failed to allocate GPU buffer"
        assert gpu_buffer_obj.tensor is not None

        rope_start = None
        rope_end = None

        # Apply RoPE in-place on ALL layers
        if needs_rope_adjustment:
            if self.enable_gpu_timing:
                rope_start = torch.cuda.Event(enable_timing=True)
                rope_start.record()
            for layer_id in range(self.num_layers):
                self._paged_rope_inplace(
                    old_positions_full,
                    new_positions_full,
                    slot_mapping_full,
                    self.kvcaches[layer_id],
                    self.vllm_two_major,
                )
            if self.enable_gpu_timing:
                rope_end = torch.cuda.Event(enable_timing=True)
                rope_end.record()

        # First yield: RoPE complete
        yield

        # Layer-by-layer read -> blend -> write
        # Read from RECOMPUTED slots (where chunk data is),
        # write to ORIGINAL slots (where model expects)
        original_slot_mapping = kwargs["slot_mapping"][starts[0]:ends[-1]]
        per_layer_read_events = []
        per_layer_write_events = []

        for layer_id in range(self.num_layers):
            # Read from paged memory into buffer (from RECOMPUTED slots)
            if self.enable_gpu_timing:
                read_start_evt = torch.cuda.Event(enable_timing=True)
                read_start_evt.record()
            lmc_ops.single_layer_kv_transfer(
                gpu_buffer_obj.tensor,
                self.kvcaches[layer_id],
                slot_mapping_full,
                True,   # from_paged = True
                False,
                self.vllm_two_major,
                self.use_mla,
            )
            if self.enable_gpu_timing:
                read_end_evt = torch.cuda.Event(enable_timing=True)
                read_end_evt.record()
                per_layer_read_events.append((read_start_evt, read_end_evt))
            self.buffer_mapping[layer_id] = gpu_buffer_obj

            yield  # Blender modifies gpu_buffer_obj.tensor

            # Write blended result back (to ORIGINAL slots)
            if self.enable_gpu_timing:
                write_start_evt = torch.cuda.Event(enable_timing=True)
                write_start_evt.record()
            lmc_ops.single_layer_kv_transfer(
                gpu_buffer_obj.tensor,
                self.kvcaches[layer_id],
                original_slot_mapping,
                False,  # from_paged = False (write to paged memory)
                False,
                self.vllm_two_major,
                self.use_mla,
            )
            if self.enable_gpu_timing:
                write_end_evt = torch.cuda.Event(enable_timing=True)
                write_end_evt.record()
                per_layer_write_events.append((write_start_evt, write_end_evt))

            del self.buffer_mapping[layer_id]

        yield  # "write last layer" yield

        if self.enable_gpu_timing:
            torch.cuda.synchronize()
            rope_time = 0.0
            if needs_rope_adjustment and rope_start is not None and rope_end is not None:
                rope_time = rope_start.elapsed_time(rope_end)

            total_read_time = 0.0
            total_write_time = 0.0
            for layer_id, ((rs, re), (ws, we)) in enumerate(
                zip(per_layer_read_events, per_layer_write_events, strict=False)
            ):
                read_time = rs.elapsed_time(re)
                write_time = ws.elapsed_time(we)
                total_read_time += read_time
                total_write_time += write_time
                logger.info(
                    "[GPU_DIRECT_PER_LAYER_TIMING] Layer %d: "
                    "read_from_paged=%.3fms, write_to_paged=%.3fms",
                    layer_id, read_time, write_time,
                )

            logger.info(
                "[GPU_DIRECT_INPLACE_TIMING] SUMMARY: "
                "rope_inplace=%.3fms, total_read=%.3fms, total_write=%.3fms, "
                "num_tokens=%d, num_layers=%d",
                rope_time, total_read_time, total_write_time,
                num_all_tokens, self.num_layers,
            )

        gpu_buffer_obj.ref_count_down()

        # Final cleanup yield
        yield

class VLLMPagedMemLayerwiseGPUConnector(GPUConnectorInterface):
    """ """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.use_gpu = use_gpu

        self.gpu_buffer_allocator = None

        assert "chunk_size" in kwargs, (
            "chunk_size should be provided to create a GPU buffer."
        )
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        assert "device" in kwargs, "device should be provided to create a GPU buffer."

        self.dtype = kwargs["dtype"]
        self.device = kwargs["device"]

        self.kvcaches: Optional[List[torch.Tensor]] = None

        # All sizes are in bytes
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()

        self.load_stream = torch.cuda.Stream()
        self.store_stream = torch.cuda.Stream()

        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheEngineMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemLayerwiseGPUConnector":
        """Create a connector from LMCacheEngineMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.

        Returns:
            A new instance of VLLMPagedMemLayerwiseGPUConnector.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")
            # NOTE (Jiayi): We use the first layer to determine the gpu buffer size.
            # NOTE (Jiayi): Using the exact number of tokens in the first layer
            # is okay since fragmentation shouldn't exist in the `gpu_buffer_allocator`
            # in layerwise mode.

            if self.use_mla:
                # MLA format: [num_blocks, block_size, head_size]
                assert kv_caches[0].dim() == 3, (
                    "For MLA, the kv_caches should have shape [num_blocks, "
                    "block_size, head_size]"
                )
                k_cache_shape_per_layer = kv_caches[0].shape
                max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]
                num_elements = k_cache_shape_per_layer.numel()
                self.vllm_two_major = False  # MLA doesn't need vllm_two_major
            else:
                # flash attention: [num_layers, 2, num_blocks, block_size,
                # num_heads, head_size]
                # flash infer:
                # [num_layers, num_blocks, 2, block_size, num_heads, head_size]
                assert kv_caches[0].shape[0] == 2 or kv_caches[0].shape[1] == 2, (
                    "The kv_caches should have shape [num_layers, 2, num_blocks, "
                    "block_size, num_heads, head_size] or "
                    "[num_layers, num_blocks, 2, block_size, num_heads, head_size]"
                )

                self.vllm_two_major = kv_caches[0].shape[0] == 2

                if self.vllm_two_major:
                    k_cache_shape_per_layer = kv_caches[0][0].shape
                else:
                    k_cache_shape_per_layer = kv_caches[0][:, 0].shape
                max_tokens = k_cache_shape_per_layer[0] * k_cache_shape_per_layer[1]
                num_elements = k_cache_shape_per_layer.numel() * 2

            logger.info(f"Lazily initializing GPU buffer (max tokens={max_tokens}).")
            gpu_buffer_size = num_elements * self.element_size
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """ """

        raise NotImplementedError

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to paged GPU memory. The first iteration will prepare some
        related metadata. In each of the following iterations, it will first
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer ->
        paged GPU memory. The last iteration simply waits for the last layer
        to finish.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        # TODO(Jiayi): Optimize away this `cat`
        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]
        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if sync:
                current_stream.wait_stream(self.load_stream)
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")

            # memobj -> gpu_buffer -> kvcaches
            with torch.cuda.stream(self.load_stream):
                for start, end, memory_obj in zip(
                    starts, ends, memory_objs_layer, strict=False
                ):
                    # Validate memory format
                    if self.use_mla:
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_MLA_FMT, (
                            f"Expected memory format {MemoryFormat.KV_MLA_FMT}, "
                            f"got {memory_obj.metadata.fmt}"
                        )
                    else:
                        assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D, (
                            f"Expected memory format {MemoryFormat.KV_T2D}, "
                            f"got {memory_obj.metadata.fmt}"
                        )
                    if self.use_gpu:
                        tmp_gpu_buffer_obj.tensor[start - offset : end - offset].copy_(
                            memory_obj.tensor, non_blocking=True
                        )
                    else:
                        lmc_ops.single_layer_kv_transfer(
                            memory_obj.tensor,
                            self.kvcaches[layer_id],
                            slot_mapping_full,
                            False,
                            True,
                            self.vllm_two_major,
                            self.use_mla,
                        )

                if self.use_gpu:
                    lmc_ops.single_layer_kv_transfer(
                        tmp_gpu_buffer_obj.tensor,
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        False,
                        True,
                        self.vllm_two_major,
                        self.use_mla,
                    )
        yield

        # synchronize the last layer
        if sync:
            current_stream.wait_stream(self.load_stream)

        # free the buffer memory
        if tmp_gpu_buffer_obj is not None:
            tmp_gpu_buffer_obj.ref_count_down()

        logger.debug(f"Finished loading all {self.num_layers} layers.")
        yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]
        sync: bool = kwargs["sync"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        tmp_gpu_buffer_obj: Optional[MemoryObj] = None
        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)
            assert self.gpu_buffer_allocator is not None
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]
        current_stream = torch.cuda.current_stream()

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            with torch.cuda.stream(self.store_stream):
                self.store_stream.wait_stream(current_stream)
                if self.use_gpu:
                    lmc_ops.single_layer_kv_transfer(
                        tmp_gpu_buffer_obj.tensor,
                        self.kvcaches[layer_id],
                        slot_mapping_full,
                        True,
                        True,
                        self.vllm_two_major,
                        self.use_mla,
                    )
                for start, end, memory_obj in zip(
                    starts, ends, memory_objs_layer, strict=False
                ):
                    assert memory_obj.tensor is not None
                    if self.use_gpu:
                        memory_obj.tensor.copy_(
                            tmp_gpu_buffer_obj.tensor[start - offset : end - offset],
                            non_blocking=True,
                        )
                    else:
                        lmc_ops.single_layer_kv_transfer(
                            memory_obj.tensor,
                            self.kvcaches[layer_id],
                            slot_mapping[start:end],
                            True,
                            True,
                            self.vllm_two_major,
                            self.use_mla,
                        )
                    # Set metadata format
                    if self.use_mla:
                        memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

            yield
            if sync:
                self.store_stream.synchronize()
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        if tmp_gpu_buffer_obj is not None:
            tmp_gpu_buffer_obj.ref_count_down()

        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        if self.use_mla:
            # MLA format: [num_tokens, hidden_dim_size]
            return torch.Size([num_tokens, self.hidden_dim_size])
        else:
            # Standard format: [num_tokens, 2, hidden_dim_size]
            return torch.Size([num_tokens, 2, self.hidden_dim_size])


class SGLangGPUConnector(GPUConnectorInterface):
    """
    The GPU KV cache should be a list of tensors, one for each layer,
    with separate key and value pointers.
    More specifically, we have:
    - kvcaches: Tuple[List[Tensor], List[Tensor]]
      - The first element is a list of key tensors, one per layer.
      - The second element is a list of value tensors, one per layer.
    - Each tensor: [page_buffer_size, head_num, head_size]

    The connector manages the transfer of KV cache data between CPU and GPU
    memory for SGLang using pointer arrays for efficient access.
    It will produce/consume memory objects with KV_2LTD format.
    """

    def __init__(
        self, hidden_dim_size: int, num_layers: int, use_gpu: bool = False, **kwargs
    ):
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

        self.num_kv_cache = num_layers if self.use_mla else num_layers * 2
        self.kv_cache_pointers = torch.empty(
            self.num_kv_cache, dtype=torch.int64, device="cpu"
        )

        if use_gpu:
            assert "chunk_size" in kwargs, (
                "chunk_size should be provided to create a GPU buffer."
            )
            assert "device" in kwargs, (
                "device should be provided to create a GPU buffer."
            )
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(
                shape, dtype=kwargs["dtype"], device=kwargs["device"]
            )
            logger.info(f"GPU buffer: {self.gpu_buffer.shape}")

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:
        assert len(kv_caches) == self.num_kv_cache

        self.kv_cache_pointers.numpy()[:] = [t.data_ptr() for t in kv_caches]
        device = kv_caches[0].device
        assert device.type == "cuda", "The device should be CUDA."
        idx = device.index
        if idx not in self.kv_cache_pointers_on_gpu:
            self.kv_cache_pointers_on_gpu[idx] = torch.empty(
                self.num_kv_cache, dtype=torch.int64, device=device
            )
        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)

        # sglang MLA kv_caches[0].shape: [num_pages * page_size, 1, head_size]
        # sglang MHA kv_caches[0].shape: [num_pages * page_size, num_heads, head_size]
        self.page_buffer_size = kv_caches[0].shape[0]
        return self.kv_cache_pointers_on_gpu[idx]

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "partial slot mapping"
             where its length is the same as the uncached token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    f" order to be processed by {self.__class__.__name__}"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    f" order to be processed by {self.__class__.__name__}"
                )

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        offset = kwargs.get("offset", 0)

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(kvcaches)
        lmc_ops.multi_layer_kv_transfer_unilateral(
            memory_obj.tensor,
            kv_cache_pointers,
            slot_mapping[start - offset : end - offset],
            kvcaches[0][0].device,
            self.page_buffer_size,
            False,
            self.use_mla,
        )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "partial slot mapping"
             where its length is the same as the uncached token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs.")

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        kvcaches: List[torch.Tensor] = kwargs["kvcaches"]
        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        kv_cache_pointers = self._initialize_pointers(kvcaches)

        if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
            lmc_ops.multi_layer_kv_transfer_unilateral(
                memory_obj.tensor,
                kv_cache_pointers,
                slot_mapping[start:end],
                kvcaches[0][0].device,
                self.page_buffer_size,
                True,
                self.use_mla,
            )
        else:
            # kvcaches -> gpu_buffer -> memobj
            assert self.gpu_buffer.device == kvcaches[0][0].device
            tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
            lmc_ops.multi_layer_kv_transfer_unilateral(
                tmp_gpu_buffer,
                kv_cache_pointers,
                slot_mapping[start:end],
                kvcaches[0][0].device,
                self.page_buffer_size,
                True,
                self.use_mla,
            )
            memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            torch.cuda.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim_size])

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)

    # TODO(Yuwei): need to optimize to enable real batching
    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)


class SGLangLayerwiseGPUConnector(GPUConnectorInterface):
    """
    The GPU KV cache should be a list of tensors, one for each layer,
    with separate key and value pointers.
    More specifically, we have:
    - kvcaches: Tuple[List[Tensor], List[Tensor]]
      - The first element is a list of key tensors, one per layer.
      - The second element is a list of value tensors, one per layer.
    - Each tensor: [page_buffer_size, head_num, head_size]

    The connector manages the transfer of KV cache data between CPU and GPU
    memory for SGLang using pointer arrays for efficient access.
    It will produce/consume memory objects with KV_2LTD format.
    """

    def __init__(
        self, hidden_dim_size: int, num_layers: int, use_gpu: bool = False, **kwargs
    ):
        assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
        self.dtype = kwargs["dtype"]
        assert "device" in kwargs, "device should be provided to create a GPU buffer."
        self.device = kwargs["device"]

        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers

        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]

        self.num_kv_cache = num_layers if self.use_mla else num_layers * 2
        self.element_size = torch.tensor([], dtype=self.dtype).element_size()
        self.kv_cache_pointers = torch.empty(
            self.num_kv_cache, dtype=torch.int64, device="cpu"
        )
        self.use_gpu = use_gpu
        self.gpu_buffer_allocator: Optional[GPUMemoryAllocator] = None

    def _lazy_initialize_buffer(self, kv_caches):
        """
        Lazily initialize the GPU buffer allocator if it is not initialized yet.
        Currently, we use the `kv_caches` (kv cache pointer) to determine
        the gpu buffer size in gpu connector.
        Also, the first request might be a bit slower due to buffer creation.
        """
        if self.use_gpu and self.gpu_buffer_allocator is None:
            logger.info("Lazily initializing GPU buffer.")
            k_cache_shape_per_layer = kv_caches[0][0].shape
            max_tokens = k_cache_shape_per_layer[0]
            logger.info(f"Lazily initializing GPU buffer (max tokens={max_tokens}).")
            num_elements = k_cache_shape_per_layer.numel() * 2
            gpu_buffer_size = num_elements * self.element_size
            self.gpu_buffer_allocator = GPUMemoryAllocator(
                gpu_buffer_size, device=self.device
            )

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        raise NotImplementedError

    @_lmcache_nvtx_annotate
    def batched_to_gpu(self, starts: List[int], ends: List[int], **kwargs):
        """
        This function is a generator that moves the KV cache from the memory
        objects to paged GPU memory. The first iteration will prepare some
        related metadata. In each of the following iterations, it will first
        wait until the loading of the previous layer finish, and then load
        one layer of KV cache from the memory objects -> GPU buffer ->
        paged GPU memory. The last iteration simply waits for the last layer
        to finish.
        In total, this the generator will yield num_layers + 2 times.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)

            assert self.gpu_buffer_allocator is not None, (
                "GPU buffer allocator should be initialized"
            )
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        offset = starts[0]

        for layer_id in range(self.num_layers):
            memory_objs_layer = yield
            if layer_id > 0:
                logger.debug(f"Finished loading layer {layer_id - 1}")

            # memobj -> gpu_buffer -> kvcaches
            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.metadata.fmt == MemoryFormat.KV_T2D
                if self.use_gpu:
                    tmp_gpu_buffer_obj.tensor[start - offset : end - offset].copy_(
                        memory_obj.tensor, non_blocking=True
                    )
                else:
                    lmc_ops.single_layer_kv_transfer_sgl(
                        memory_obj.tensor,
                        self.kvcaches[0][layer_id],
                        self.kvcaches[1][layer_id],
                        slot_mapping[start:end],
                        False,
                        True,
                    )

            if self.use_gpu:
                t, h, d = self.kvcaches[0][layer_id].shape

                lmc_ops.single_layer_kv_transfer_sgl(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[0][layer_id].view(t, 1, h, d),
                    self.kvcaches[1][layer_id].view(t, 1, h, d),
                    slot_mapping_full,
                    False,
                    True,
                )

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()

        logger.debug(f"Finished loading layer {layer_id}")
        yield

    @_lmcache_nvtx_annotate
    def batched_from_gpu(
        self,
        memory_objs: Union[List[List[MemoryObj]]],
        starts: List[int],
        ends: List[int],
        **kwargs,
    ):
        """
        This function is a generator that moves the KV cache from the paged GPU
        memory to the memory objects. The first iteration will prepare some
        related metadata and initiate the transfer in the first layer. In each
        of the following iterations, it will first wait until the storing of
        previous layer finishes, and then initiate string the KV cache of the
        current layer one. The storing process of the KV cache is paged GPU
        memory -> GPU buffer -> memory objects. The last iteration simply waits
        for the last layer to finish.
        In total, this the generator will yield num_layers + 1 times.

        :param memory_objs: The memory objects to store the KV cache. The first
            dimension is the number of layers, and the second dimension is the
            number of memory objects (i.e., number of chunks) for each layer.

        :param starts: The starting indices of the KV cache in the corresponding
            token sequence.

        :param ends: The ending indices of the KV cache in the corresponding
            token sequence.

        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        if "sync" not in kwargs:
            raise ValueError("'sync' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        self._lazy_initialize_buffer(self.kvcaches)

        slot_mapping_chunks = []
        for start, end in zip(starts, ends, strict=False):
            slot_mapping_chunks.append(slot_mapping[start:end])

        slot_mapping_full = torch.cat(slot_mapping_chunks, dim=0)

        num_tokens = len(slot_mapping_full)

        if self.use_gpu:
            buffer_shape = self.get_shape(num_tokens)

            assert self.gpu_buffer_allocator is not None, (
                "GPU buffer allocator should be initialized"
            )
            tmp_gpu_buffer_obj = self.gpu_buffer_allocator.allocate(
                buffer_shape, self.dtype, MemoryFormat.KV_T2D
            )
            assert tmp_gpu_buffer_obj is not None, (
                "Failed to allocate GPU buffer in GPUConnector"
            )
            assert tmp_gpu_buffer_obj.tensor is not None

        for layer_id in range(self.num_layers):
            memory_objs_layer = memory_objs[layer_id]
            # kvcaches -> gpu_buffer -> memobj
            if self.use_gpu:
                t, h, d = self.kvcaches[0][layer_id].shape
                lmc_ops.single_layer_kv_transfer_sgl(
                    tmp_gpu_buffer_obj.tensor,
                    self.kvcaches[0][layer_id].view(t, 1, h, d),
                    self.kvcaches[1][layer_id].view(t, 1, h, d),
                    slot_mapping_full,
                    True,
                    True,
                )

            start_idx = 0

            for start, end, memory_obj in zip(
                starts, ends, memory_objs_layer, strict=False
            ):
                assert memory_obj.tensor is not None
                if self.use_gpu:
                    chunk_len = memory_obj.tensor.shape[0]
                    memory_obj.tensor.copy_(
                        tmp_gpu_buffer_obj.tensor[start_idx : start_idx + chunk_len],
                        non_blocking=True,
                    )
                    start_idx += chunk_len
                else:
                    lmc_ops.single_layer_kv_transfer_sgl(
                        memory_obj.tensor,
                        self.kvcaches[0][layer_id],
                        self.kvcaches[1][layer_id],
                        slot_mapping[start:end],
                        True,
                        True,
                    )

            yield
            logger.debug(f"Finished offloading layer {layer_id}")

        # free the buffer memory
        if self.use_gpu:
            tmp_gpu_buffer_obj.ref_count_down()
        yield

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([num_tokens, 2, self.hidden_dim_size])
