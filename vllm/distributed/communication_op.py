from typing import Any, Dict, Optional, Union

import torch
import torch.distributed

from .parallel_state import get_tp_group


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(input_: torch.Tensor,
                                     dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(input_: torch.Tensor,
                                 dst: int = 0,
                                 dim: int = -1) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(tensor_dict: Optional[Dict[Any, Union[torch.Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    if not torch.distributed.is_initialized():
        return tensor_dict

    torch.cuda.synchronize()
    torch.distributed.barrier()
    if torch.distributed.get_rank() == src:
        print(f"before broadcast_tensor_dict {tensor_dict}") 
    torch.cuda.synchronize()
    torch.distributed.barrier()
    output = get_tp_group().broadcast_tensor_dict(tensor_dict, src)
    torch.cuda.synchronize()
    torch.distributed.barrier()
    if torch.distributed.get_rank() == src:
        print(f"after broadcast_tensor_dict {tensor_dict}") 
    torch.cuda.synchronize()
    torch.distributed.barrier()
    return output
