import os
import copy
import hydra
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from omegaconf import OmegaConf

import torch

import transformers
import tokenizers

from llava.utils.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer
from llava.utils.config import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
    ImageEncoderArguments,
)

from llava.utils import conversation as conversation_lib
from llava.model import *
from llava.utils.mm_utils import tokenizer_image_token

from PIL import Image
from omegaconf import DictConfig
from transformers.trainer_utils import set_seed
from .dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )
