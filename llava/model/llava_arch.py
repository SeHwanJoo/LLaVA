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


from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

import torch
import torch.nn as nn

from llava.utils.constants import (DEFAULT_IM_END_TOKEN,
                                   DEFAULT_IM_START_TOKEN,
                                   DEFAULT_VI_END_TOKEN,
                                   DEFAULT_VI_START_TOKEN,
                                   DEFAULT_IMAGE_PATCH_TOKEN,
                                   IGNORE_INDEX,
                                   IMAGE_TOKEN_INDEX,
                                   VIDEO_TOKEN_INDEX)
from llava.utils.mm_utils import get_anyres_image_grid_shape

from .image_encoder.builder import build_image_tower
from .video_encoder.builder import build_video_tower
from .multimodal_projector.builder import build_vision_projector
from omegaconf import DictConfig


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "image_encoder"):
            self.image_tower = build_image_tower(config.image_encoder, delay_load=True)
            self.image_mm_projector = build_vision_projector(config.image_encoder)

            if "unpad" in getattr(config.image_encoder, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(
                    torch.empty(config.image_encoder.hidden_size, dtype=self.dtype)
                )
        if hasattr(config, "video_encoder"):
            self.video_tower = build_video_tower(config.video_encoder, delay_load=True)
            self.video_mm_projector = build_vision_projector(config.video_encoder)

            if "unpad" in getattr(config.video_encoder, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(
                    torch.empty(config.video_encoder.hidden_size, dtype=self.dtype)
                )

    def get_image_tower(self):
        image_tower = getattr(self, "image_tower", None)
        if type(image_tower) is list:
            image_tower = image_tower[0]
        return image_tower
    
    def get_video_tower(self):
        video_tower = getattr(self, "video_tower", None)
        if type(video_tower) is list:
            video_tower = video_tower[0]
        return video_tower

    def initialize_vision_modules(self, model_args, key="image_encoder", fsdp=None):
        vision_tower = getattr(model_args, key).vision_tower
        mm_vision_select_layer = getattr(model_args, key).mm_vision_select_layer
        mm_vision_select_feature = getattr(model_args, key).mm_vision_select_feature
        pretrain_mm_mlp_adapter = getattr(model_args, key).pretrain_mm_mlp_adapter
        mm_patch_merge_type = getattr(model_args, key).mm_patch_merge_type
        setattr(self.config, key, DictConfig({}))
        getattr(self.config, key).mm_vision_tower = vision_tower
        if key == "image_encoder":
            if self.get_image_tower() is None:
                vision_tower = build_image_tower(getattr(model_args, key))

                if fsdp is not None and len(fsdp) > 0:
                    self.image_tower = [vision_tower]
                else:
                    self.image_tower = vision_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    vision_tower = self.image_tower[0]
                else:
                    vision_tower = self.image_tower
                vision_tower.load_model()
        elif key == "video_encoder":
            if self.get_video_tower() is None:
                vision_tower = build_video_tower(getattr(model_args, key))

                if fsdp is not None and len(fsdp) > 0:
                    self.video_tower = [vision_tower]
                else:
                    self.video_tower = vision_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    vision_tower = self.video_tower[0]
                else:
                    vision_tower = self.video_tower
                vision_tower.load_model()
        else:
            raise NotImplementedError

        key_config = getattr(self.config, key)
        key_config.use_mm_proj = True
        key_config.mm_projector_type = getattr(getattr(model_args, key), 'mm_projector_type', 'linear')
        key_config.mm_hidden_size = vision_tower.hidden_size
        key_config.mm_vision_select_layer = mm_vision_select_layer
        key_config.mm_vision_select_feature = mm_vision_select_feature
        key_config.mm_patch_merge_type = mm_patch_merge_type
        key_config.hidden_size = self.config.hidden_size
        if key == "image_encoder":
            mm_projector = getattr(self, 'image_mm_projector', None)
            if mm_projector is None:
                self.image_mm_projector = build_vision_projector(getattr(self.config, key))

                if 'unpad' in mm_patch_merge_type:
                    embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                    self.image_newline = nn.Parameter(
                        torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                    )
            else:
                # In case it is frozen by LoRA
                for p in self.image_mm_projector.parameters():
                    p.requires_grad = True
        elif key == "video_encoder":
            mm_projector = getattr(self, 'video_mm_projector', None)
            if mm_projector is None:
                self.video_mm_projector = build_vision_projector(getattr(self.config, key))

                if 'unpad' in mm_patch_merge_type:
                    embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                    self.video_newline = nn.Parameter(
                        torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                    )
            else:
                # In case it is frozen by LoRA
                for p in self.image_mm_projector.parameters():
                    p.requires_grad = True
        else:
            raise NotImplementedError
        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            if key == "image_encoder":
                self.image_mm_projector.load_state_dict(get_w(mm_projector_weights, 'image_mm_projector'))
            elif key == "video_encoder":
                self.video_mm_projector.load_state_dict(get_w(mm_projector_weights, 'video_mm_projector'))
            else:
                raise NotImplementedError


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self, images, video):
        if images is not None:
            return self.get_image_tower()
        elif video is not None:
            return self.get_video_tower()

    def get_image_tower(self):
        return self.get_model().get_image_tower()

    def get_video_tower(self):
        return self.get_model().get_video_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_image_tower()(images)
        image_features = self.get_model().image_mm_projector(image_features)
        return image_features
    
    def encode_video(self, video):
        video_features = self.get_model().get_video_tower()(video)
        video_features = self.get_model().video_mm_projector(video_features)
        return video_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        video,
        sizes=None,
    ):
        vision_tower = self.get_vision_tower(images, video)

        if (vision_tower is None and images is None) or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )
        inputs = video if images is None else images
        if type(inputs) is list or inputs.ndim == 5:
            if type(inputs) is list:
                inputs = [x.unsqueeze(0) if x.ndim == 3 else x for x in inputs]
            concat_inputs = torch.cat([image for image in inputs], dim=0)
            if images is not None:
                features = self.encode_images(concat_inputs)
            elif video is not None:
                features = self.encode_video(concat_inputs)
            split_sizes = [input.shape[0] for input in inputs]
            features = torch.split(features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            input_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            if mm_patch_merge_type == "flat":
                features = [x.flatten(0, 1) for x in features]
            elif mm_patch_merge_type.startswith("spatial"):
                new_features = []
                for idx, feature in enumerate(features):
                    if feature.shape[0] > 1:
                        base_feature = feature[0]
                        feature = feature[1:]
                        height = width = self.get_vision_tower(images, video).num_patches_per_side
                        assert height * width == base_feature.shape[0]
                        if input_aspect_ratio == "anyres":
                            num_patch_width, num_patch_height = (
                                get_anyres_image_grid_shape(
                                    sizes[idx],
                                    self.config.image_grid_pinpoints,
                                    self.get_vision_tower(images, video).config.image_size,
                                )
                            )
                            feature = feature.view(
                                num_patch_height, num_patch_width, height, width, -1
                            )
                        else:
                            raise NotImplementedError
                        if "unpad" in mm_patch_merge_type:
                            feature = feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            feature = feature.flatten(1, 2).flatten(2, 3)
                            feature = unpad_image(
                                feature, sizes[idx]
                            )
                            feature = torch.cat(
                                (
                                    feature,
                                    self.model.image_newline[:, None, None]
                                    .expand(*feature.shape[:-1], 1)
                                    .to(feature.device),
                                ),
                                dim=-1,
                            )
                            feature = feature.flatten(1, 2).transpose(0, 1)
                        else:
                            feature = feature.permute(
                                0, 2, 1, 3, 4
                            ).contiguous()
                            feature = feature.flatten(0, 3)
                        feature = torch.cat(
                            (base_feature, feature), dim=0
                        )
                    else:
                        feature = feature[0]
                        if "unpad" in mm_patch_merge_type:
                            feature = torch.cat(
                                (
                                    feature,
                                    self.model.image_newline[None].to(
                                        feature.device
                                    ),
                                ),
                                dim=0,
                            )
                    new_features.append(feature)
                features = new_features
            else:
                raise ValueError(
                    f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}"
                )
        else:
            if images is not None:
                features = self.encode_images(images)
            elif video is not None:
                features = self.encode_video(video)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
            self.config, "mm_use_start_end", False
        ):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        cur_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if images is not None:
                num_inputs = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            elif video is not None:
                num_inputs = (cur_input_ids == VIDEO_TOKEN_INDEX).sum()
            if num_inputs == 0:
                cur_features = features[cur_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_idx += 1
                continue
            if images is not None:
                token_indices = (
                    [-1]
                    + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                    + [cur_input_ids.shape[0]]
                )
            elif video is not None:
                token_indices = (
                    [-1]
                    + torch.where(cur_input_ids == VIDEO_TOKEN_INDEX)[0].tolist()
                    + [cur_input_ids.shape[0]]
                )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[
                        token_indices[i] + 1 : token_indices[i + 1]
                    ]
                )
                cur_labels_noim.append(
                    cur_labels[token_indices[i] + 1 : token_indices[i + 1]]
                )
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(
                torch.cat(cur_input_ids_noim)
            )
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_inputs + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_inputs:
                    cur_features = features[cur_idx]
                    cur_idx += 1
                    cur_new_input_embeds.append(cur_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_start_end:
            # TODO
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

    def forward_(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        video: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                video,
                image_sizes,
            )

        return self.get_model().forward_(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate_(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = (
                self.prepare_inputs_labels_for_multimodal(
                    inputs,
                    position_ids,
                    attention_mask,
                    None,
                    None,
                    images,
                    video,
                    image_sizes=image_sizes,
                )
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return self.get_model().generate_(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation_(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        video = kwargs.pop("video", None)
        inputs = self.get_model().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        if video is not None:
            inputs["video"] = video
        return inputs
    
def setup_model_functions(instance):
    instance.model.forward_ = instance.forward
    instance.model.generate_ = instance.generate
    instance.forward = super(instance.__class__, instance).forward_
    instance.generate = super(instance.__class__, instance).generate_
    instance.prepare_inputs_for_generation = super(instance.__class__, instance).prepare_inputs_for_generation_
