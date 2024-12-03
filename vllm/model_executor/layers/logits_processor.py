"""A layer that compute logits from hidden_stats."""
import inspect
from typing import Optional

import torch
import torch.nn as nn

from vllm.distributed import (tensor_model_parallel_all_gather,
                              tensor_model_parallel_gather)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.platforms import current_platform
from vllm.distributed import get_sp_tp_group


class LogitsProcessor(nn.Module):
    """Process logits and apply logits processors from sampling metadata.

    This layer does the following:
    1. Gather logits from model hidden_states.
    2. Scale logits if needed.
    3. Apply logits processors (if any).
    """

    def __init__(self,
                 vocab_size: int,
                 org_vocab_size: Optional[int] = None,
                 scale: float = 1.0,
                 logits_as_input: bool = False,
                 soft_cap: Optional[float] = None) -> None:
        """
        Args:
            scale: A scaling factor to apply to the logits.
        """
        super().__init__()
        self.scale = scale
        self.vocab_size = vocab_size
        # Whether the input is logits (default is hidden states).
        self.logits_as_input = logits_as_input
        # original vocabulary size (without LoRA).
        self.org_vocab_size = org_vocab_size or vocab_size
        # Soft cap the logits. Used in Gemma 2.
        self.soft_cap = soft_cap
        # Whether to use gather or all-gather to gather the logits.
        self.use_gather = not current_platform.is_tpu()

        self.numforward = 0

    def forward(
        self,
        lm_head: VocabParallelEmbedding,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        torch.cuda.synchronize()
        torch.distributed.barrier()

        for i in range(torch.distributed.get_world_size()):
            if i == torch.distributed.get_rank():
                print(f"myid {torch.distributed.get_rank()} LogitsProcessor {self.numforward} hidden_states shape {hidden_states.shape} logits_as_input {self.logits_as_input}", flush=True)
            torch.cuda.synchronize()
            torch.distributed.barrier()
        # if torch.distributed.get_rank() == 0:
        # print(f"myid {torch.distributed.get_rank()} LogitsProcessor {self.numforward} hidden_states shape {hidden_states.shape} logits_as_input {self.logits_as_input}\n", flush=True)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        for i in range(torch.distributed.get_world_size()):
            if i == torch.distributed.get_rank():
                print(f"myid {torch.distributed.get_rank()} hidden_states before pruning shape {hidden_states.shape} sampling indices {sampling_metadata.selected_token_indices} hidden_states {hidden_states}", flush=True)
            torch.cuda.synchronize()
            torch.distributed.barrier()
        if self.numforward == 2:
            exit()


        if self.logits_as_input:
            logits = hidden_states
        else:
            torch.cuda.synchronize()
            torch.distributed.barrier()
            # if torch.distributed.get_rank() == 0:
            for i in range(torch.distributed.get_world_size()):
                if i == torch.distributed.get_rank():
                    print(f"myid {torch.distributed.get_rank()} hidden_states before pruning shape {hidden_states.shape} sampling indices {sampling_metadata.selected_token_indices} hidden_states {hidden_states}\n", flush=True)
                torch.cuda.synchronize()
                torch.distributed.barrier()
            # print(f"myid {torch.distributed.get_rank()} hidden_states before pruning shape {hidden_states.shape} sampling indices {sampling_metadata.selected_token_indices} hidden_states {hidden_states}\n", flush=True)
            # hidden_states = _prune_hidden_states(hidden_states,
            #                                      sampling_metadata)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            if self.numforward == 2:
                exit()
            hidden_states = torch.index_select(hidden_states, 0, sampling_metadata.selected_token_indices)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            # if torch.distributed.get_rank() == 0:
            print(f"myid {torch.distributed.get_rank()} hidden_states after pruning shape {hidden_states.shape} embedding_bias type {type(embedding_bias)}", flush=True)
            # Get the logits for the next tokens.
            logits = self._get_logits(hidden_states, lm_head, embedding_bias)
        torch.cuda.synchronize()
        torch.distributed.barrier()
        print(f"myid {torch.distributed.get_rank()} LogitsProcessor logits type  after _get_logits{type(logits)}\n", flush=True)
        self.numforward += 1

        
        if not get_sp_tp_group().is_first_rank:
            logits = None

        if logits is not None:
            if self.soft_cap is not None:
                logits = logits / self.soft_cap
                logits = torch.tanh(logits)
                logits = logits * self.soft_cap

            if self.scale != 1.0:
                logits *= self.scale

            # Apply logits processors (if any).
            logits = _apply_logits_processors(logits, sampling_metadata)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print(f"LogitsProcessor logits type {type(logits)}", flush=True)
            print(f"LogitsProcessor logits shape {logits.shape}", flush=True)

        return logits

    def _get_logits(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print(f"logits_processor _get_logits hidden_states shape {hidden_states.shape}", flush=True)
        # Get the logits for the next tokens.
        logits = lm_head.linear_method.apply(lm_head,
                                             hidden_states,
                                             bias=embedding_bias)
        if self.use_gather:
            # None may be returned for rank > 0
            logits = tensor_model_parallel_gather(logits)
        else:
            # Gather is not supported for some devices such as TPUs.
            # Use all-gather instead.
            # NOTE(woosuk): Here, the outputs of every device should not be None
            # because XLA requires strict SPMD among all devices. Every device
            # should execute the same operations after gathering the logits.
            logits = tensor_model_parallel_all_gather(logits)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        if torch.distributed.get_rank() == 0:
            print(f"logits_processor _get_logits logits shape {logits.shape}", flush=True)
        # Remove paddings in vocab (if any).
        if logits is not None:
            logits = logits[..., :self.org_vocab_size]
        return logits

    def extra_repr(self) -> str:
        s = f"vocab_size={self.vocab_size}"
        s += f", forg_vocab_size={self.org_vocab_size}"
        s += f", scale={self.scale}, logits_as_input={self.logits_as_input}"
        return s


def _prune_hidden_states(
    hidden_states: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    return hidden_states.index_select(0, sampling_metadata.selected_token_indices)


def _apply_logits_processors(
    logits: torch.Tensor,
    sampling_metadata: SamplingMetadata,
) -> torch.Tensor:
    found_logits_processors = False
    logits_processed = 0
    for seq_group in sampling_metadata.seq_groups:
        seq_ids = seq_group.seq_ids
        sampling_params = seq_group.sampling_params
        logits_processors = sampling_params.logits_processors
        if logits_processors:
            found_logits_processors = True

            for seq_id, logits_row_idx in zip(seq_ids,
                                              seq_group.sample_indices):
                logits_row = logits[logits_row_idx]
                past_tokens_ids = seq_group.seq_data[seq_id].output_token_ids
                prompt_tokens_ids = seq_group.seq_data[seq_id].prompt_token_ids

                for logits_processor in logits_processors:
                    parameters = inspect.signature(logits_processor).parameters
                    if len(parameters) == 3:
                        logits_row = logits_processor(prompt_tokens_ids,
                                                      past_tokens_ids,
                                                      logits_row)
                    else:
                        logits_row = logits_processor(past_tokens_ids,
                                                      logits_row)

                logits[logits_row_idx] = logits_row

        logits_processed += len(seq_group.sample_indices) + len(
            seq_group.prompt_logprob_indices)

    if found_logits_processors:
        # verifies that no rows in logits were missed unexpectedly
        assert logits_processed == logits.shape[0]
    return logits
