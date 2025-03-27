# 📝 Change Log

## 🔄 Changes
- **Refactored Experiment Management**  
  - **Key Points**: `Collaboration`, `Configuration`
  - **Before**: Experiment management was based on script (bash), making it difficult to manage multiple experiments.
  ```bash
  scripts/v1_5
    ├── finetune_lora.sh
    ├── finetune.sh
    ├── finetune_task_lora.sh
    ├── finetune_task.sh
    └── pretrain.sh
  ```
  ```bash
    deepspeed llava/train/train_mem.py \
        --deepspeed ./scripts/zero2.json \
        --model_name_or_path lmsys/vicuna-13b-v1.5 \
        --version plain \
        --data_path ./playground/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
        --image_folder ./playground/data/LLaVA-Pretrain/images \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --tune_mm_mlp_adapter True \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --bf16 True \
        --output_dir ./checkpoints/llava-v1.5-13b-pretrain \
        --num_train_epochs 1 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy "no" \
        --save_strategy "steps" \
        --save_steps 24000 \
        --save_total_limit 1 \
        --learning_rate 1e-3 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to wandb
  ```
  - **After**: Replaced with **Hydra-based** configuration for easier multi-experiment setup.
  ```
    configs/
    ├── data_args
    │   ├── llava_image.yaml
    │   └── ssv2_video.yaml
    ├── experiments
    │   ├── finetune_lora.yaml
    │   ├── finetune_task_lora.yaml
    │   ├── finetune_task.yaml
    │   ├── finetune.yaml
    │   ├── pretrain.yaml
    │   └── train_video.yaml
    ├── model_args
    │   ├── image_encoder
    │   │   ├── base.yaml
    │   │   └── from_adapter.yaml
    │   ├── liuhaotian_vicuna-13b.yaml
    │   └── lmsys_vicuna-13b.yaml
    └── training_args
        ├── finetune_lora.yaml
        ├── finetune.yaml
        └── pretrain.yaml
  ```
  ```yaml
  defaults:
    - /data_args/ssv2_video.yaml
    - /model_args/lmsys_vicuna-13b
    - /training_args/pretrain
    - _self_

    seed: 1997
    training_args:
    output_dir: ./checkpoints/llava-v1.5-13b-pretrain-video
    per_device_train_batch_size: 16
  ```
  - **Expected effect**: 
    - **Enhanced Collaboration**: Modular and well-structured configurations eliminate script-baed complexity, making it easier for teams to collabotate, modify, and scale experiments efficiently.
    - **Improved Experiment Management**: Hydra's hierarachical and reusable configs enable seamless multi-experiment setup, reducing errors and improving ocnsistency across training workflows.

- **Code Cleanup & Structure Refinement** 
  - **Key Points**: `Collaboration` 
  - **Before**: Common functions were scattered across different files, making reuse difficult.
  ```
  llava
    ├── __init__.py
    ├── 📂 eval
    ├── 📂 model
    ├── 📂 serve
    ├── 📂 train
    │   ├── llama_flash_attn_monkey_patch.py
    │   ├── llama_xformers_attn_monkey_patch.py
    │   ├── llava_trainer.py
    │   ├── train_mem.py
    │   ├── train.py
    │   └── train_xformers.py
    ├── constants.py
    ├── conversation.py
    ├── mm_utils.py
    └── utils.py
  ```
  - **After**: Organized common utility functions and slightly modified file structure for better maintainability.
  ```
  llava
    ├── __init__.py
    ├── 📂 dataset
    │   ├── __init__.py
    │   ├── 📂 image_dataset
    │   ├── 📂 video_dataset
    │   ├── utils.py
    │   └── builder.py
    ├── 📂 eval
    ├── 📂 model
    ├── 📂 serve
    ├── 📂 train
    │   ├── llama_flash_attn_monkey_patch.py
    │   ├── llava_trainer.py
    │   ├── train.py
    │   └── utils.py
    └── 📂 utils
        ├── __init__.py
        ├── config.py
        ├── constants.py
        ├── conversation.py
        ├── mm_utils.py
        └── utils.py
  ```
  - **Expected effect**: Enhanced code maintainability and reusability, improving collaboration efficiency.

- **Video Dataset and extendable dataset**
  - **Key Points**: `Data scale-up`, `Collaboration`
  - **Before:** No support for video datasets and multi-datasets configurations.
  - **After**: Added support for **video datasets** and modificate dataset for **scalability**.

  **note: multi-type dataset is not supported**
  
  eg) video dataset and image dataset

  Managed by configs
  ```yaml
    - /data_args:
      - ssv2_video.yaml
      - ssv2_video_2.yaml
  ```
  ```sh
  configs/data_args/
  ├── llava_image.yaml
  └── ssv2_video.yaml
  ```
  Folder structure
  ```sh
  llava/dataset/
  ├── __init__.py
  ├── builder.py
  ├── utils.py
  ├── image_dataset
  │   └── dataset.py
  └── video_dataset
      ├── dataset.py
      ├── functional.py
      ├── masking_generator.py
      ├── rand_augment.py
      ├── random_erasing.py
      └── video_transforms.py
  ```
  - **Expected effect**: 
    - **scalable dataset**: Previously, only one dataset could be used, but now multiple datasets can be trained simultaneously through easy configuration management.
    - **video dataset**: Added support for previously unsupported video datasets.

- **Video Encoder and Extendable Encoder**
  - **Key Points**: `Model Scale-up`, `Collaboration`
  - **Before**: No support for video encoder and multi-encoder.
  - **After**: Added support for **video encoder** and modificate encoder for **scalability**.

    Managed by configuration
    ```yaml
    defaults:
      - image_encoder: base
      - video_encoder: base
      - _self_
    ```
    ```sh
    configs/model_args/
    ├── image_encoder
    │   ├── base.yaml
    │   └── from_adapter.yaml
    └── video_encoder
        ├── base.yaml
        └── from_adapter.yaml
    ```
    Folder structure
    ```sh
    llava/model/
    ├── image_encoder
    │   ├── builder.py
    │   └── clip_encoder.py
    └── video_encoder
        ├── builder.py
        └── clip_encoder.py
    ```
  - **Expected effect**: 
    - **scalable encoder**: Video data includes various features like images, frames, and audio. To effectively process these, multiple encoders are required, and the strucutre has been designed to scale accordingly.
    - **video encoder**: Added support for previously unsupported video encoder.

- **Various LLM**  
  - **Key Points**: `Model Scale-up`
  - **Before**: Only the Llama model was supported, and adding new models required extensive code modifications, as shown below:
  ```python
  class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
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
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
      ...skip...

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
      ...skip...

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
      ...skip...
  ```
  - **After**: Added easy support for adding new LLM models with configuration through a streamlined approach:
  code
  ```python
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
  ```
  config
  ```yaml
  class_name: LlavaQwen2ForCausalLM
  model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
  ```
  - **Expected effect**: With the continuous emergence of new LLMs, this approach allows for effortless intergration and experimentation with newly released models.
---

# Future works
- **Efficient Adapter**: AS-IS, only LoRA is supported, but various adapter methods such as DoRA and MoRA are actively being researched. The framework should be extended to suport multiple adapter types while maintain flexible configuration.
- **Feature Project**: AS-IS, extracted image and video features are processed independently. However, to enable joint learning across modalities - including image, video, audio, etc - the framework should be modified to allow seamless intergration of these features.
- **Time Embedding**: Video preprocessing extracts frames based on fixed segments sizes. But, since video durations vary, segmenting by time intervals could be more effective. Incoporating time embeddings could provide benefits similar to positional embeddins enabling better temporal understanding. This approach would also allow handling long videos by segmenting and embedding them.
- **Code Refinement & CleanUp**: The project lacks a standardized code convention, requiring significant refactoring. Unnecessary files should be removed.
- **evaluation & serving**: While modifications have been made to the train pipeline, the evaluation and serving components have not yet been updated accoordingly.
