# SPDX-License-Identifier: Apache-2.0

from typing import Callable, List, Optional

import torch
from compressed_tensors.quantization import QuantizationStrategy
from torch.nn import Parameter

from vllm.distributed import get_sp_group
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    Fp8LinearOp, maybe_create_device_identity, normalize_e4m3fn_to_e4m3fnuz,
    requantize_with_max_scale)
from vllm.model_executor.parameter import (ChannelQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)
from vllm.platforms import current_platform

__all__ = ["CompressedTensorsW8A8Fp8"]


class CompressedTensorsW8A8Fp8(CompressedTensorsScheme):

    def __init__(self, strategy: str, is_static_input_scheme: bool):
        self.strategy = strategy
        self.out_dtype = torch.get_default_dtype()
        self.is_static_input_scheme = is_static_input_scheme
        self.fp8_linear = Fp8LinearOp(use_per_token_if_dynamic=True)

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def process_weights_after_loading(self, layer) -> None:
        # If per tensor, when we have a fused module (e.g. QKV) with per
        # tensor scales (thus N scales being passed to the kernel),
        # requantize so we can always run per tensor
        if self.strategy == QuantizationStrategy.TENSOR:
            max_w_scale, weight = requantize_with_max_scale(
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                logical_widths=layer.logical_widths,
            )

            if current_platform.is_fp8_fnuz():
                input_scale = getattr(layer, 'input_scale', None)

                weight, max_w_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    weight=weight,
                    weight_scale=max_w_scale,
                    input_scale=input_scale)
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale,
                                                  requires_grad=False)

            layer.weight = Parameter(weight.t(), requires_grad=False)
            layer.weight_scale = Parameter(max_w_scale, requires_grad=False)

        # If channelwise, scales are already lined up, so just transpose.
        elif self.strategy == QuantizationStrategy.CHANNEL:
            weight = layer.weight

            if current_platform.is_fp8_fnuz():
                input_scale = getattr(layer, 'input_scale', None)

                weight, weight_scale, input_scale = \
                    normalize_e4m3fn_to_e4m3fnuz(
                        weight=weight,
                        weight_scale=layer.weight_scale,
                        input_scale=input_scale)
                if input_scale is not None:
                    layer.input_scale = Parameter(input_scale,
                                                  requires_grad=False)
            else:
                weight_scale = layer.weight_scale.data

            layer.weight = Parameter(weight.t(), requires_grad=False)
            # required by torch.compile to be torch.nn.Parameter
            layer.weight_scale = Parameter(weight_scale, requires_grad=False)

        else:
            raise ValueError(f"Unknown quantization strategy {self.strategy}")

        # INPUT SCALE
        if self.is_static_input_scheme and hasattr(layer, 'input_scale'):
            layer.input_scale = Parameter(layer.input_scale.max(),
                                          requires_grad=False)
        else:
            layer.input_scale = None

        # TODO: skip below if shapeshifter threshold is 0
        sp_size = get_sp_group().world_size
        sp_rank = get_sp_group().rank_in_group
        output_partition_sizes = layer.logical_widths
        if output_partition_sizes == [layer.weight.shape[1]]:
            assert layer.weight.shape[0] % sp_size == 0
            chunk_size = layer.weight.shape[0] // sp_size
            self.sp_tp_weight = layer.weight.split(
                chunk_size, dim=0)[sp_rank].t().contiguous().t()
        else:
            assert layer.weight.shape[1] % sp_size == 0
            chunk_sizes = []
            for size in output_partition_sizes:
                chunk_size = size // sp_size
                chunk_sizes.extend([chunk_size] * sp_size)
            split = layer.weight.split(chunk_sizes, dim=1)
            self.sp_tp_weight = torch.cat(
                [split[i] for i in range(sp_rank, len(split), sp_size)],
                dim=1).t().contiguous().t()

        if get_sp_group().rank == 0:
            print(f"layer {layer}")
            if output_partition_sizes == [layer.weight.shape[1]]:
                print(f"row parallel SP: {sp_size}.")
            else:
                print(f"column parallel {sp_size}.")
            print(f"loaded weight shape: {layer.weight.shape} "
                  f"stride {layer.weight.stride()} "
                  f"contiguous {layer.weight.is_contiguous()} "
                  f"tcontiguous {layer.weight.t().is_contiguous()} "
                  f" {layer.weight.dtype}")
            print(f"     logical widths: {layer.logical_widths}")
            print(f"      SP_TP weights: {self.sp_tp_weight.shape} "
                  f"stride {self.sp_tp_weight.stride()} "
                  f"contiguous {self.sp_tp_weight.is_contiguous()} "
                  f"tcontiguous {self.sp_tp_weight.t().is_contiguous()} "
                  f" {self.sp_tp_weight.dtype}")

    def create_weights(self, layer: torch.nn.Module,
                       output_partition_sizes: List[int],
                       input_size_per_partition: int,
                       params_dtype: torch.dtype, weight_loader: Callable,
                       **kwargs):
        maybe_create_device_identity()

        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes

        # WEIGHT
        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=torch.float8_e4m3fn),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        # TODO: update create_xxx_parameter functions to return
        # the newly added parameters
        if self.strategy == QuantizationStrategy.CHANNEL:
            weight_scale = ChannelQuantScaleParameter(
                data=torch.empty((sum(output_partition_sizes), 1),
                                 dtype=torch.float32),
                output_dim=0,
                weight_loader=weight_loader)
        else:
            assert self.strategy == QuantizationStrategy.TENSOR
            weight_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32),
                                                   weight_loader=weight_loader)

        # min requirement for fp8 kernels
        weight_scale[:] = torch.finfo(torch.float32).min
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(data=torch.empty(
                len(output_partition_sizes), dtype=torch.float32),
                                                  weight_loader=weight_loader)
            input_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", input_scale)

        if get_sp_group().rank == 0:
            print(f"create weights {layer.weight.shape} {layer.weight.dtype}")

    def apply_weights(self,
                      layer: torch.nn.Module,
                      x: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:

        from vllm.v1.worker.gpu_model_runner import SP_TP_MODE
        return self.fp8_linear.apply(
            input=x,
            weight=self.sp_tp_weight if SP_TP_MODE else layer.weight,
            weight_scale=layer.weight_scale,
            out_dtype=self.out_dtype,
            input_scale=layer.input_scale,
            bias=bias)