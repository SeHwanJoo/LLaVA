# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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

import os
import hydra
import pathlib

import torch


from llava.train.llava_trainer import LLaVATrainer


from transformers.trainer_utils import set_seed
from llava.train.utils import config2argument, prepare_models_args, save_model
from llava.dataset import make_supervised_data_module


CONFIG_DIR = os.environ.get("OP_CONFIG_DIR") or "../../configs"


@hydra.main(
    config_path=CONFIG_DIR, config_name="experiments/example", version_base="1.3"
)
def train(config):
    global local_rank
    config = config.experiments
    set_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config.training_args.local_rank = int(os.getenv("LOCAL_RANK", "0"))
    training_args, model_args, data_args = config2argument(config)

    local_rank = training_args.local_rank

    model, tokenizer, training_args, data_args = prepare_models_args(
        training_args, model_args, data_args
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = LLaVATrainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    save_model(model, trainer, training_args)


if __name__ == "__main__":
    train()
