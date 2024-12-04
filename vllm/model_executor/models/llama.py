# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only LLaMA model compatible with HuggingFace weights."""
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
from torch import nn
from transformers import LlamaConfig

from vllm.attention import Attention, AttentionMetadata
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_world_group, get_tp_group, get_sp_group, get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    get_compressed_tensors_cache_scale)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader, maybe_remap_kv_scale_name)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.utils import is_hip

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (AutoWeightsLoader, PPMissingLayer, is_pp_missing_parameter,
                    make_empty_intermediate_tensors_factory, make_layers)

class LlamaMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation: {hidden_act}. "
                             "Only silu is supported for now.")
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        if torch.distributed.get_rank() == 0:
            print(f"llama MLP x: {x.shape} gate_up: {gate_up.shape}")
        x = self.act_fn(gate_up)
        if torch.distributed.get_rank() == 0:
            print(f"llama MLP x: {x.shape} activation")
        x, _ = self.down_proj(x)
        if torch.distributed.get_rank() == 0:
            print(f"llama MLP x: {x.shape} down_proj")
        return x


class LlamaAttention(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config: Optional[QuantizationConfig] = None,
        bias: bool = False,
        cache_config: Optional[CacheConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tp_group().world_size
        # sp_size = get_sp_group().world_size
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        is_neox_style = True
        if quant_config is not None and quant_config.get_name() == "gguf":
            is_neox_style = False

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )
        self.attn = Attention(
            self.num_heads//get_sp_group().world_size,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads//get_sp_group().world_size,
            cache_config=cache_config,
            quant_config=quant_config,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        N_ranks: List[int],
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        
        # variables for Ulysses attention
        SP = get_sp_group().world_size
        TP = get_tp_group().world_size
        N = sum(N_ranks) 
        N_ulysses = N_ranks[get_sp_group().rank_in_group]
        d = self.total_num_heads * self.head_dim
        d_kv = self.total_num_kv_heads * self.head_dim
        assert N_ulysses == hidden_states.shape[0]
        assert d == hidden_states.shape[1]
        assert d//TP == self.q_size
        assert d_kv//TP == self.kv_size

        test = torch.ones((5, 3), device=get_world_group().device, dtype=torch.float16)
        for i in range(torch.distributed.get_world_size()):
            if torch.distributed.get_rank() == i:
                print(f"test type {test.dtype} shape {test.shape} {test}", flush=True)
            torch.cuda.synchronize()
            torch.distributed.barrier()

        qkv_ = torch.ones((N, (self.q_size+2*self.kv_size)//SP), dtype=torch.float16, device=get_world_group().device)
        for i in range(torch.distributed.get_world_size()):
            if torch.distributed.get_rank() == i:
                print(f"before all-to-all qkv_ type {qkv_.dtype} shape {qkv_.shape} {qkv_}", flush=True)
            torch.cuda.synchronize()
            torch.distributed.barrier()     

        # qkv projection
        qkv, _ = self.qkv_proj(hidden_states)

        qkv = torch.transpose(qkv, 0, 1).contiguous()
        torch.cuda.synchronize()
        torch.distributed.barrier()
        for i in range(torch.distributed.get_world_size()):
            if torch.distributed.get_rank() == i:
                print(f"qkv type {qkv.dtype} shape {qkv.shape} {qkv}", flush=True)
            torch.cuda.synchronize()
            torch.distributed.barrier()

        # pack send buffer
        # q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        #qkv = torch.cat((q.view((N_ulysses, SP, self.q_size//SP)),
        #                 k.view((N_ulysses, SP, self.kv_size//SP)),
        #                 v.view((N_ulysses, SP, self.kv_size//SP))), dim=-1).transpose(0, 1).contiguous()
        

        # communication
        torch.distributed.all_to_all_single(qkv_, qkv, output_split_sizes=N_ranks, group=get_sp_group().device_group)

        for i in range(torch.distributed.get_world_size()):
            if torch.distributed.get_rank() == i:
                print(f"after all-to-all qkv_ type {qkv_.dtype} shape {qkv_.shape} {qkv_}", flush=True)
            torch.cuda.synchronize()
            torch.distributed.barrier()

        # unpack receive buffer
        q_, k_, v_ = qkv_.split([self.q_size//SP, self.kv_size//SP, self.kv_size//SP], dim=-1)

        for i in range(torch.distributed.get_world_size()):
            if torch.distributed.get_rank() == i:
                print(f"q_ type {q_.dtype} shape {q_.shape} {q_}", flush=True)
                print(f"k_ type {k_.dtype} shape {k_.shape} {k_}", flush=True)
                print(f"v_ type {v_.dtype} shape {v_.shape} {v_}", flush=True)
            torch.cuda.synchronize()
            torch.distributed.barrier()

        # if torch.distributed.get_rank() == 0:
        #         print(f"llama attention q {q.shape}, k {k.shape}, v {v.shape}")
        #         print(f"llama attention qkv {qkv.shape} is_contiguous {qkv.is_contiguous()}")
        #         print(f"llama attention qkv_ {qkv_.shape} is_contiguous {qkv_.is_contiguous()}")
        #         print(f"llama attention q_ {q_.shape}, k_ {k_.shape}, v_ {v_.shape}")

        # positional embeddings
        q_, k_ = self.rotary_emb(positions, q_, k_)

        # attention 
        attn_output = self.attn(q_, k_, v_, kv_cache, attn_metadata)

        # communication
        c = torch.empty((SP, N_ulysses, self.q_size//SP), dtype=hidden_states.dtype, device=hidden_states.device)
        torch.distributed.all_to_all_single(c, attn_output, input_split_sizes=N_ranks, group=get_sp_group().device_group)
        c = torch.transpose(c, 0, 1).reshape(N_ulysses, self.q_size)

        # output projection
        output, _ = self.o_proj(c)


        torch.cuda.synchronize()
        torch.distributed.barrier()
        for i in range(torch.distributed.get_world_size()):
            if torch.distributed.get_rank() == i:
                print(f"hidden_states type {hidden_states.dtype} shape {hidden_states.shape} {hidden_states}", flush=True)
            torch.cuda.synchronize()
            torch.distributed.barrier()


        return output


class LlamaDecoderLayer(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
                config, "original_max_position_embeddings", None):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings)
        max_position_embeddings = getattr(config, "max_position_embeddings",
                                          8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False)
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads",
                                 config.num_attention_heads),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        self.inference = 0

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        N_ranks: List[int],
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print(f"llama decoder layer positions {positions.shape}, hidden_states {hidden_states.shape}, N_ranks {N_ranks}, kv_cache {kv_cache.shape} residual {residual.shape if residual is not None else None}")
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print(f"llama decoder layer input_layernorm hidden_states {hidden_states.shape}, residual {residual.shape}")
        hidden_states = self.self_attn(positions=positions,
                                       hidden_states=hidden_states,
                                       N_ranks=N_ranks,
                                       kv_cache=kv_cache,
                                       attn_metadata=attn_metadata)
        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print("test", flush=True)
        return hidden_states, residual


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
        "intermediate_tensors": 0,
    })
class LlamaModel(nn.Module):

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size
        if get_pp_group().is_first_rank or (config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: LlamaDecoderLayer(config=config,
                                             cache_config=cache_config,
                                             quant_config=quant_config,
                                             prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size))
        
        self.hidden_size = config.hidden_size
        self.numforward = 0

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors],
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:


        torch.cuda.synchronize()
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print(f"start inference {self.numforward} *********************", flush=True)
        torch.cuda.synchronize()
        torch.distributed.barrier()

        assert inputs_embeds is None
        N = len(input_ids)
        SP = get_sp_group().world_size
        N_ranks = [N//SP]*SP
        for i in range(N % SP):
            N_ranks[i] += 1

        torch.cuda.synchronize()
        torch.distributed.barrier()
        for i in range(torch.distributed.get_world_size()):
            if torch.distributed.get_rank() == i:
                print(f"myid {torch.distributed.get_rank()} input_ids {input_ids.shape}, positions {positions.shape}, N {N}, N_ranks {N_ranks}")
            torch.cuda.synchronize()
            torch.distributed.barrier()

        SP_rank = get_sp_group().rank_in_group
        # input_ids = torch.narrow(input_ids, 0, sum(N_ranks[:SP_rank]), N_ranks[SP_rank])

        torch.cuda.synchronize()
        torch.distributed.barrier()
        for i in range(torch.distributed.get_world_size()):
            if torch.distributed.get_rank() == i:
                print(f"myid {torch.distributed.get_rank()} narrowed input_ids {input_ids.shape} {input_ids}")
            torch.cuda.synchronize()
            torch.distributed.barrier()

        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # torch.empty gives CUDA exception
        # hidden_states_ = torch.ones((sum(N_ranks), self.hidden_size), device=get_world_group().device, dtype=torch.float16)

        hidden_states_ = torch.narrow(hidden_states, 0, sum(N_ranks[:SP_rank]), N_ranks[SP_rank]).clone()

        torch.cuda.synchronize()
        torch.distributed.barrier()
        for i in range(torch.distributed.get_world_size()):
            if torch.distributed.get_rank() == i:
                print(f"myid {torch.distributed.get_rank()} hidden_states_ {hidden_states_.shape} {hidden_states_}")
            torch.cuda.synchronize()
            torch.distributed.barrier()

        P = get_world_group().world_size
        TP = get_tp_group().world_size
        PP = get_pp_group().world_size
        # torch.set_printoptions(profile="full")
        if torch.distributed.get_rank() == 0:
            print(f"P {P} TP {TP}, SP {SP}, PP {PP}")
            print(f"start_layer {self.start_layer}, end_layer {self.end_layer}")
        # torch.set_printoptions(profile="default")

        torch.cuda.synchronize()
        torch.distributed.barrier()
        for i in range(torch.distributed.get_world_size()):
            if torch.distributed.get_rank() == i:
                print(f"myid {torch.distributed.get_rank()} hidden_states {hidden_states.shape} {hidden_states} residual {residual.shape if residual is not None else None} {residual}")
            torch.cuda.synchronize()
            torch.distributed.barrier()
        
        # for i in range(self.start_layer, self.end_layer):
        for i in range(self.start_layer, 3):
            layer = self.layers[i]
            if torch.distributed.get_rank() == 0:
                print(f"layer {i}")
            hidden_states_, residual = layer(positions, hidden_states_, N_ranks,
                                            kv_caches[i - self.start_layer],
                                            attn_metadata, residual)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print("test 2", flush=True)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        for i in range(torch.distributed.get_world_size()):
            if torch.distributed.get_rank() == i:
                print(f"myid {torch.distributed.get_rank()} Llama Model: hidden_states type: {type(hidden_states)} shape: {hidden_states.shape} device: {hidden_states.device} shape[1]: {hidden_states.shape[1]} {hidden_states}")
            torch.cuda.synchronize()
            torch.distributed.barrier()

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })
        
        hidden_states_, _ = self.norm(hidden_states_, residual)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print(f"test 3 forward {self.numforward}", flush=True)




        # all-gather sequences
        # hidden_states_ = torch.split(torch.empty((sum(N_ranks), hidden_states.shape[1]), device=hidden_states.device, dtype=hidden_states.dtype), N_ranks)
        # hidden_states_ = torch.empty((sum(N_ranks), self.hidden_size), device=get_world_group().device, dtype=torch.float16)
        # hidden_states_ = torch.empty((5, 10), device=get_sp_group().get_device(), dtype=hidden_states.dtype)


        torch.cuda.synchronize()
        torch.distributed.barrier()
        for i in range(torch.distributed.get_world_size()):
            if torch.distributed.get_rank() == i:
                print(f"myid {torch.distributed.get_rank()} Llama Model: hidden_states_ type: {type(hidden_states_)} shape: {hidden_states_.shape} {hidden_states_}")
            torch.cuda.synchronize()
            torch.distributed.barrier()


        hidden_states_list = [torch.narrow(hidden_states, 0, sum(N_ranks[:i]), N_ranks[i]) for i in range(SP)]

        # print(f"myid {torch.distributed.get_rank()} sp_group ranks {get_sp_group().ranks}", flush=True)
        torch.distributed.all_gather(hidden_states_list, hidden_states_, group=get_sp_group().device_group)


        # hidden_states_list = [torch.empty((N_ranks[i], hidden_states.shape[1]), dtype=hidden_states.dtype, device=hidden_states.device) for i in range(SP)]
        # print(f"myid {torch.distributed.get_rank()} {[hidden_states_list[i].shape in range(SP)]}\n", flush=True)
        # hidden_states = torch.empty((sum(N_ranks), hidden_states.shape[1]), dtype=hidden_states.dtype, device=hidden_states.device)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        print(f"myid {torch.distributed.get_rank()} numforward {self.numforward} hidden_states_ type {[hidden_states_list[i].type for i in range(SP)]} shape {[hidden_states_list[i].shape for i in range(SP)]}", flush=True)
        # if torch.distributed.get_rank() == 0:
            #for i in range(SP):
        #     print(f"myid {torch.distributed.get_rank()} hidden_states_list type {[hidden_states_list[i].type for i in range(SP)]} shape {[hidden_states_list[i].shape for i in range(SP)]}\n", flush=True)
            # print(f"hidden_states_list {hidden_states_list}", flush=True)
        # hidden_states = torch.cat(hidden_states_)

        # hidden_states = torch.cat(hidden_states_list)
        # hidden_states = hidden_states_ # torch.empty((sum(N_ranks), hidden_states.shape[1]), dtype=hidden_states.dtype, device=hidden_states.device)
        # torch.cat(hidden_states_list, out=hidden_states)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        # print(f"myid {torch.distributed.get_rank()} after allgather hidden_states {type(hidden_states)} shape {hidden_states.shape}\n", flush=True)
        # print(f"hidden_states_list {hidden_states_list}", flush=True)
        # print(f"hidden_states_ {hidden_states_}", flush=True)
        # print(f"myid {torch.distributed.get_rank()} after allgather hidden_states_ {type(hidden_states_)} shape {hidden_states_.shape}", flush=True)
        torch.cuda.synchronize()
        torch.distributed.barrier()

        # return torch.ones((sum(N_ranks), self.config.hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print(f"end of inference {self.numforward} *************************** hidden_states {hidden_states.shape}", flush=True)


        # if self.numforward == 2:
        #     exit()
        self.numforward += 1

        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if ("rotary_emb.cos_cached" in name
                    or "rotary_emb.sin_cached" in name):
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            if scale_name := get_compressed_tensors_cache_scale(name):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)

    # If this function is called, it should always initialize KV cache scale
    # factors (or else raise an exception). Thus, handled exceptions should
    # make sure to leave KV cache scale factors in a known good (dummy) state
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        for layer_idx, scaling_factor in kv_cache_scales_loader(
                quantization_param_path, tp_rank, tp_size,
                self.config.num_hidden_layers,
                self.config.__class__.model_type):
            if not isinstance(self.layers[layer_idx], nn.Identity):
                layer_self_attn = self.layers[layer_idx].self_attn

            if is_hip():
                # The scaling factor convention we are assuming is
                # quantized_value * scaling_factor ~= true_value
                # which is consistent with the practice of setting
                # scaling_factor = tensor_amax / FPtype_max
                scaling_factor *= 2
            if hasattr(layer_self_attn, "kv_scale"):
                layer_self_attn.attn._kv_scale = scaling_factor
            else:
                raise RuntimeError("Self attention has no KV cache scaling "
                                   "factor attribute!")


class LlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"]
    }

    # LoRA specific attributes
    supported_lora_modules = [
        "qkv_proj", "o_proj", "gate_up_proj", "down_proj", "embed_tokens",
        "lm_head"
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings"
    }
    embedding_padding_modules = ["lm_head"]

    # BitandBytes specific attributes
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    # in TP, these weights are partitioned along the column dimension (dim=-1)
    column_parallel_weights_modules = [".down_proj.", ".o_proj."]
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    # Mistral/Llama models can also be loaded with --load-format mistral
    # from consolidated.safetensors checkpoints
    mistral_mapping = {
        "layers": "model.layers",
        "attention": "self_attn",
        "wq": "q_proj",
        "wk": "k_proj",
        "wv": "v_proj",
        "wo": "o_proj",
        "attention_norm": "input_layernorm",
        "feed_forward": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "ffn_norm": "post_attention_layernorm",
        "tok_embeddings": "model.embed_tokens",
        "output": "lm_head",
        "norm": "model.norm"
    }

    def __init__(
        self,
        config: LlamaConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.lora_config = lora_config

        self.model = LlamaModel(config,
                                cache_config,
                                quant_config,
                                lora_config=lora_config,
                                prefix="model")
        if get_pp_group().is_last_rank:
            self.unpadded_vocab_size = config.vocab_size
            if lora_config:
                self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not lora_config else
                    lora_config.lora_vocab_padding_size),
                quant_config=quant_config,
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(
                    self.model.embed_tokens)

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                    config.vocab_size,
                                                    logit_scale)
            self.sampler = Sampler()
        else:
            self.lm_head = PPMissingLayer()
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors)
        
        self.numforward = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[torch.Tensor, IntermediateTensors]:
        model_output = self.model(input_ids, positions, kv_caches,
                                  attn_metadata, intermediate_tensors)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print(f"forward {self.numforward} llama model_output {model_output.shape}", flush=True)
        # if self.numforward == 2:
        #     exit()
        self.numforward += 1
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        for i in range(torch.distributed.get_world_size()):
            if i == torch.distributed.get_rank():
                print(f"myid {torch.distributed.get_rank()} compute_logits hidden_states {hidden_states.shape} sampling_metadata {sampling_metadata.selected_token_indices} ", flush=True)
            torch.cuda.synchronize()
            torch.distributed.barrier()
        logits = self.logits_processor(self.lm_head, hidden_states,
                                       sampling_metadata)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print(f"compute_logits logits type {type(logits)}", flush=True)
            print(f"compute_logits logits {logits.shape}", flush=True)
        if self.numforward == 3:
            exit()
        return logits

    def sample(self, logits: torch.Tensor,
               sampling_metadata: SamplingMetadata) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."]
                           if self.config.tie_word_embeddings else None),
        )
        loader.load_weights(
            self.maybe_remap_mistral(name, loaded_weight)
            for name, loaded_weight in weights)

    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    # This function is used to remap the mistral format as
    # used by Mistral and Llama <=2
    def maybe_remap_mistral(
        self,
        name: str,
        loaded_weight: torch.Tensor,
    ) -> Tuple[str, torch.Tensor]:

        def permute(w: torch.Tensor, n_heads: int):
            attn_in = self.config.head_dim * n_heads
            attn_out = self.config.hidden_size

            return w.view(n_heads, attn_in // n_heads // 2, 2,
                          attn_out).transpose(1, 2).reshape(attn_in, attn_out)

        mapping = self.mistral_mapping
        modules = name.split(".")

        # rotary embeds should be sliced
        if "wk" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_key_value_heads)
        elif "wq" in modules:
            loaded_weight = permute(loaded_weight,
                                    self.config.num_attention_heads)

        for item in modules:
            if item in mapping and mapping[item] not in name:
                name = name.replace(item, mapping[item])

        return name, loaded_weight
