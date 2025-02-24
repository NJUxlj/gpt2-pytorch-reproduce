# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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

import inspect
import os
import warnings
from contextlib import contextmanager

import torch
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory
from huggingface_hub import hf_hub_download
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput, TokenClassifierOutput
from transformers.utils import PushToHubMixin

from .tuners import (
    AdaLoraModel,
    AdaptionPromptModel,
    LoraModel,
    PrefixEncoder,
    PromptEmbedding,
    PromptEncoder,
)
from .utils import (
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    WEIGHTS_NAME,
    PeftConfig,
    PeftType,
    PromptLearningConfig,
    TaskType,
    _set_adapter,
    _set_trainable,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    shift_tokens_right,
)