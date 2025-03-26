from llava.dataset.image_dataset import ImageDataset
from llava.dataset.video_dataset import VideoDataset
import transformers
from typing import Dict
from llava.utils.constants import IGNORE_INDEX
from typing import Dict, Sequence
from dataclasses import dataclass
import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import Sampler


def build_dataset(tokenizer: transformers.PreTrainedTokenizer,
                  data_args_list) -> Dict:
    train_dataset = []
    for data_args in data_args_list:
        if data_args.data_type == "video":
            dataset = VideoDataset(
                tokenizer=tokenizer,
                data_path=data_args.data_path,
                data_args=data_args)
        elif data_args.data_type == "image":
            dataset = ImageDataset(
                tokenizer=tokenizer,
                data_path=data_args.data_path,
                data_args=data_args)
        else:
            raise NotImplementedError
        train_dataset.append(dataset)
    train_dataset = ConcatDataset(train_dataset)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        if 'video' in instances[0]:
            video = [instance['video'] for instance in instances]
            if all(x is not None and x.shape == video[0].shape for x in video):
                batch['video'] = torch.stack(video)
            else:
                batch['video'] = video

        return batch
