import base64
from typing import Any

import numpy as np
import torch
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.online.fp8 import Fp8PerTensorOnlineLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter

from prism_ternary.kernels import GROUP, HADAMARD_BLOCK, LANES, block_layout, ternary_linear

QUANT_METHOD = "prism_ternary"
TERNARY_SUFFIXES = frozenset(
    {"qkv_proj", "o_proj", "gate_up_proj", "down_proj", "in_proj_qkvz", "out_proj"}
)


def pack_signs(signs: np.ndarray) -> str:
    return base64.b64encode(np.packbits(signs < 0).tobytes()).decode()


def unpack_signs(encoded: str, width: int) -> torch.Tensor:
    bits = np.unpackbits(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8))[:width]
    return torch.from_numpy(np.where(bits == 1, -1.0, 1.0).astype(np.float32))


@register_quantization_config(QUANT_METHOD)
class PrismTernaryConfig(QuantizationConfig):
    def __init__(self, signs: dict[int, torch.Tensor]) -> None:
        super().__init__()
        self.signs = signs

    @classmethod
    def get_name(cls) -> str:
        return QUANT_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        # SM75 (Turing) is supported by the FP16 Tensor Core fallback in
        # kernels.cu. SM80+ keeps the original BF16 Tensor Core path.
        return 75

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PrismTernaryConfig":
        if config["group_size"] != GROUP or config["hadamard_block"] != HADAMARD_BLOCK:
            raise ValueError(f"unsupported prism_ternary layout: {config}")
        return cls(
            {int(width): unpack_signs(encoded, int(width)) for width, encoded in config["signs"].items()}
        )

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> QuantizeMethodBase | None:
        if prefix.startswith("mtp."):
            return Fp8PerTensorOnlineLinearMethod() if isinstance(layer, LinearBase) else None
        if isinstance(layer, ParallelLMHead):
            return PrismTernaryLinearMethod(self)
        if isinstance(layer, LinearBase):
            return (
                PrismTernaryLinearMethod(self)
                if prefix.rsplit(".", 1)[-1] in TERNARY_SUFFIXES
                else UnquantizedLinearMethod()
            )
        return UnquantizedEmbeddingMethod() if isinstance(layer, VocabParallelEmbedding) else None


class PrismTernaryLinearMethod(LinearMethodBase):
    def __init__(self, quant_config: PrismTernaryConfig) -> None:
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        weight_loader = extra_weight_attrs["weight_loader"]
        rows = sum(output_partition_sizes)
        if input_size_per_partition % HADAMARD_BLOCK:
            raise ValueError(f"input width {input_size_per_partition} is not a multiple of {HADAMARD_BLOCK}")
        layer.register_parameter(
            "weight",
            PackedvLLMParameter(
                data=torch.empty(rows, input_size_per_partition // LANES, dtype=torch.int32),
                input_dim=1,
                output_dim=0,
                packed_dim=1,
                packed_factor=LANES,
                weight_loader=weight_loader,
            ),
        )
        layer.register_parameter(
            "scales",
            GroupQuantScaleParameter(
                data=torch.empty(rows, input_size_per_partition // GROUP, dtype=torch.float16),
                input_dim=1,
                output_dim=0,
                weight_loader=weight_loader,
            ),
        )
        signs = torch.empty(input_size_per_partition, dtype=torch.float32)
        signs.copy_(self.quant_config.signs[input_size_per_partition])
        layer.register_buffer("hadamard_signs", signs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight, scales = block_layout(layer.weight.data, layer.scales.data)
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.scales = torch.nn.Parameter(scales, requires_grad=False)

    def apply(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        y = ternary_linear(x, layer.weight, layer.scales, layer.hadamard_signs)
        return y if bias is None else y + bias


def register() -> None:
    return None
