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
from transformers import (AutoConfig, AutoModelForCausalLM, Phi3Config,
                          Phi3ForCausalLM, Phi3Model)

from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel, setup_model_functions


class LlavaPhiConfig(Phi3Config):
    model_type = "llava_phi3"


class LlavaLlamaModel(LlavaMetaModel, Phi3Model):
    config_class = LlavaPhiConfig

    def __init__(self, config: Phi3Config):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaPhiForCausalLM(Phi3ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaPhiConfig

    def __init__(self, config):
        super(Phi3ForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        setup_model_functions(self)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model


AutoConfig.register("llava_phi3", LlavaPhiConfig)
AutoModelForCausalLM.register(LlavaPhiConfig, LlavaPhiForCausalLM)
