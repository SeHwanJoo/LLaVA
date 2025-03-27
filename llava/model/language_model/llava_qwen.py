#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch.nn as nn
from transformers import (AutoConfig, AutoModelForCausalLM, Qwen2Config,
                          Qwen2ForCausalLM, Qwen2Model)

from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel, setup_model_functions


class LlavaQwen2Config(Qwen2Config):
    model_type = "llava_qwen2"


class LlavaLlamaModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwen2Config

    def __init__(self, config: Qwen2Config):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaQwen2ForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        setup_model_functions(self)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


AutoConfig.register("llava_qwen2", LlavaQwen2Config)
AutoModelForCausalLM.register(LlavaQwen2Config, LlavaQwen2ForCausalLM)
