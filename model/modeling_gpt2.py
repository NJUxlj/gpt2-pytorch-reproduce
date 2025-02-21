# coding=utf-8
# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch OpenAI GPT-2 model."""

import math
import os
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, SequenceSummary
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_gpt2 import GPT2Config


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "openai-community/gpt2"
_CONFIG_FOR_DOC = "GPT2Config"




def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """Load tf checkpoints in a pytorch model
    
    ##Notice:
    we do not need it now.
    
    """
    
    
def eager_attention_forward(module, query, key, value, attention_mask, head_mask=None, **kwargs):
    """Eager mode implementation of the attention block
    
    
    ## Args:
    query: query.shape = (batch_size, num_heads, seq_len, head_dim)
    key: key.shape = (batch_size, num_heads, seq_len, head_dim)
    value: value.shape = (batch_size, num_heads, seq_len, head_dim)
    attention_mask: attention_mask.shape = (batch_size, 1, seq_len, seq_len)
    
    """
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    
    
    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            attn_weights.shape, 
            value.size(-1)**0.5, 
            dtype=attn_weights.dtype, 
            device=attn_weights.device
        )
    
    
    # Layer-wise attention scaling
    if module.scale_attn_by_inverse_layer_idx:  
        attn_weights = attn_weights/(module.layer_idx+1)
    
    
    '''
    主要功能是在非交叉注意力（not module.is_cross_attention）的情况下，
    为注意力权重（attn_weights）应用因果掩码（causal mask）。
    
    因果掩码用于确保在生成序列时，每个位置只能关注到它之前的位置，
    从而保证模型的自回归特性。
    '''
    
    if not module.is_cross_attention:
        # if only "normal" attention layer implements causal mask
        query_length, key_length = query.size(-2), key.size(-2)

        # module.bias.shape = (1, 1, seq_len, seq_len)
        # 从 module.bias 中提取因果掩码。module.bias 是一个预先计算好的下三角矩阵，用于表示因果关系。
        # 通过切片操作，提取出与当前查询和键序列长度匹配的部分。
        # 掩码对齐：key_length - query_length 处理的是当key序列比query长时的偏移对齐（例如在prefix decoding场景）
        causal_mask = module.bias[:, :, key_length - query_length : key_length, :key_length]
        mask_value = torch.finfo(attn_weights.dtype).min
        # 获取注意力权重数据类型的最小值。这个最小值将用于掩码未被允许关注的位置。假设最小数据类型是int，那么对应的最小值就是 -32768。

        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        # [] 表示创建一个零维张量（即标量）
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        
        '''
        casual mask:
        [[ T,  F,  F,  F ],  
        [ T,  T,  F,  F ],  
        [ T,  T,  T,  F ],  
        [ T,  T,  T,  T ]] 
        
        
        attn_weights:
        
        [[2.0, -∞,  -∞,  -∞ ],  
        [1.0, 2.0, -∞,  -∞ ],  
        [3.0, 0.5, 2.0, -∞ ],  
        [0.5, 1.0, 1.5, 2.0]]  
        '''

    
    
    if attention_mask is not None: # padding mask
        attn_weights = attn_weights + attention_mask
    
    attn_weights =  nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)
    
    if head_mask is not None:
        attn_weights = attn_weights * head_mask  # (batch_size, num_heads, query_length, key_length) * (1, num_heads, 1, 1), where head_mask[0,:,0,0] = [1, 0.1, 0], 这样做可以快速地把一个头上的注意力矩阵给抹掉
        
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1,2)
    
    return attn_output, attn_weights


class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True

        self.pruned_heads = set()





class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states