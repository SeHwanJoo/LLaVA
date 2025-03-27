# 🚀 Video Vision-Language Model Training Repository

## 📌 Overview  
This repository provides an implementation of a Vision-Language Model (VLM) for video-based tasks. It supports dataset preprocessing, model training.

## ✨ Features  
- Efficient training with distributed computing
- LLaVA based architecture

## 📂 Directory Structure  
```
├── 📂 checkpoints                        # checkpoint directory for trained models
│   ├── 📂llava-v1.5-13b-pretrain
│   └── 📂llava-v1.5-13b-pretrain-video
├── 📂 configs                            # configuration files (.yaml)
│   ├── 📂 data_args                      # config for data
│   ├── 📂 experiments                    # config for experiments config
│   ├── 📂 model_args                     # config for models
│   └── 📂 training_args                  # config for train
├── 📂 llava
│   ├── 📂 dataset                        # Data implementations
│   │   ├── 📂 image_dataset
│   │   └── 📂 video_dataset
│   ├── 📂 eval
│   ├── 📂 model                          # Model implementations for LLaVA
│   │   ├── 📂 language_model
│   │   ├── 📂 image_encoder
│   │   ├── 📂 video_encoder
│   │   └── 📂 multimodal_projector
│   ├── 📂 serve
│   ├── 📂 train                          # Train implementations
│   └── 📂 utils
├── 📂 playground                         # Train data for video and LLaVA
│   └── 📂 data
│       ├── 📂 LLaVA-Pretrain
│       └── 📂 video
└── 📂 scripts
```

## 🛠️ Installation  

1. Clone this repository and navigate to LLaVA folder
```sh
git clone https://github.com/SeHwanJoo/LLaVA.git
cd LLaVA
```

2. Install Pacakge
```sh
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip
pip install .
```

3. Install additional packages for training cases
```sh
pip install -e ".[train]"
pip install flash-attn==2.5.5 --no-build-isolation
```

## 📊 Dataset Preparation  
Download and preprocess datasets before training

If you prefer to `skip downloading` the dataset and only proceed with training, you can skip Dataset Preparation. I've already included a few sample images and videos there as a dummy dataset. 

### 🌄 Image Dataset (LLaVA)
1. Download `image.zip` from [here](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/images.zip)
2. unzip image.zip following the folder structure below.
```sh
playground/data/
└── LLaVA-Pretrain
    ├── images
    └── images.zip
```
3. Download `meta_data` from [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/blip_laion_cc_sbu_558k.json)
4. Final folder structure should look like this.
```sh
playground/data/
└── LLaVA-Pretrain
    ├── blip_laion_cc_sbu_558k.json
    └── images
```
Reference: https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md#pretraining-dataset
<details>
<summary>detailed explanation</summary>

The pretraining dataset used in this release is a subset of CC-3M dataset, filtered with a more balanced concept coverage distribution.  Please see [here](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K) for a detailed description of the dataset structure and how to download the images.

If you already have CC-3M dataset on your disk, the image names follow this format: `GCC_train_000000000.jpg`.  You may edit the `image` field correspondingly if necessary.

| Data | Chat File | Meta Data | Size |
| --- |  --- |  --- | ---: |
| CC-3M Concept-balanced 595K | [chat.json](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/chat.json) | [metadata.json](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/metadata.json) | 211 MB
| LAION/CC/SBU BLIP-Caption Concept-balanced 558K | [blip_laion_cc_sbu_558k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/blip_laion_cc_sbu_558k.json) | [metadata.json](#) | 181 MB

**Important notice**: Upon the request from the community, as ~15% images of the original CC-3M dataset are no longer accessible, we upload [`images.zip`](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K/blob/main/images.zip) for better reproducing our work in research community. It must not be used for any other purposes. The use of these images must comply with the CC-3M license. This may be taken down at any time when requested by the original CC-3M dataset owner or owners of the referenced images.
</details>

### 🎥 Video Dataset
1. Download the dataset from [official website](https://www.qualcomm.com/developer/artificial-intelligence/datasets).
2. Preprocess the dataset by changing the video extensoin from `.webm` to `.mp4`. You can use this [script](https://github.com/SeHwanJoo/LLaVA/tree/main/scripts/dataset/webm2mp4.py).
3. Download `train.csv`, `val.csv`, `test.csv` from [here](https://drive.google.com/drive/folders/1cfA-SrPhDB9B8ZckPvnh8D5ysCjD-S_I).
4. Download `something-something-v2-id2label.json` from [here](https://huggingface.co/datasets/huggingface/label-files/blob/main/something-something-v2-id2label.json).
```sh
playground/data/
└── video
    ├── 20bn-something-something-v2-mp4
    ├── something-something-v2-id2label.json
    ├── test.csv
    ├── train.csv
    └── val.csv
```
5. Preprocess the dataset for instruction format by this [script](https://github.com/SeHwanJoo/LLaVA/tree/main/scripts/dataset/convert2instruct.py). Then final folder structure should look like this.
```sh
playground/data/
└── video
    ├── 20bn-something-something-v2-mp4
    ├── something-something-v2-id2label.json
    ├── something-something-v2_test.json
    ├── something-something-v2_train.json
    ├── something-something-v2_val.json
    ├── test.csv
    ├── train.csv
    └── val.csv
```


Reference: https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md#pretraining-dataset

## 🏋️‍♂️ Training  
To train the model, use the following command:  
```sh
deepspeed --no_local_rank llava/train/train.py --config-name=experiments/pretrain
```
There are various ways to train the model. The following are methods that utilize the predefined configs in the `configs/experiments` directory:
<details> 
<summary>detailed explanation</summary>

1. Training with images
```sh
deepspeed --no_local_rank llava/train/train.py --config-name=experiments/pretrain
```
2. Training with videos
```sh
deepspeed --no_local_rank llava/train/train.py --config-name=experiments/pretrain_video
```
```
3. Training the Phi base model with video
```sh
deepspeed --no_local_rank llava/train/train.py --config-name=experiments/train_phi
```
4. Training the Qwen base model with video
```sh
deepspeed --no_local_rank llava/train/train.py --config-name=experiments/train_qwen
```
</details>

## 🧪 Evaluation  
TBD

## 🚀 Model Inference  
TBD

## 🧑🏻‍🔧 Model Serve
TBD
