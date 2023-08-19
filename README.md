# Dataset Quantization

Official implementation of "[Dataset Quantization]()". 

<p align="center"><img src="./figs/effects.png" align="center" height="400" ></p>

**Dataset Quantization**
[Daquan Zhou*](https://scholar.google.com/citations?user=DdCAbWwAAAAJ&hl=en), [Kai Wang*](https://kaiwang960112.github.io/), [Jianyang Gu*](https://vimar-gu.github.io/), [Xiangyu Peng](https://scholar.google.com/citations?user=KRUTk7sAAAAJ&hl=en), [Dongze Lian](https://dongzelian.com/), [Yifan Zhang](https://sites.google.com/view/yifan-zhang/%E9%A6%96%E9%A1%B5), [Yang You](https://ai.comp.nus.edu.sg/), [Jiashi Feng](https://sites.google.com/site/jshfeng/) (*Equal Contribution)

## Highlight :sparkles:​

- DQ is accepted by **ICCV2023**!
- DQ is able to generate condensed small datasets for training unseen network architectures with state-of-the-art compression ratios for lossless model training. 
- We support both vision and language dataset compression:
- Vision tasks: with 60% data from ImageNet, the models can be trained with no performance drop including classification, semantic segmentation, and object detection.
- Language tasks: with 20% data from Alpaca’s instruction tuning data, the models can be trained with negligible performance on BBH, DROP, MMLU, and Human-Eval. 

## Getting Started

Download the repo:

```bash
git clone https://github.com/vimar-gu/DQ.git
cd DQ
```

Set up the environment:

```bash
conda create -n dq python=3.9
conda activate dq
pip install -r requirements.txt
```

prepare the pretrained MAE model for the image reconstruction. 

```bash
wget https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth
mv mae_visualize_vit_large_ganloss.pth ./pretrained
```

## DQ for Image Classification

### Overview

![pipeline-classification](./figs/pipeline-classification.png)

Dataset Quantization is conducted in the following steps:

1. Dataset bin generation. Firstly we iteratively select non-overlapping dataset bins according to the submodular function. 
2. Bin sampling. Then we uniformly sample a certain portion (the required data keep ratio) from each bin and form the final compact set. 
3. Pixel quantization and reconstruction. We employ a GradCAM module to select informative image patches. By only storing the informative patches, the required storage can be further reduced. An MAE model is adopted for image reconstruction. For simplicity, here we directly conduct the reconstruction for evaluating our full method. 

### Data Preparation

Put [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) data to `~/data_cifar`.
Put unzipped [ImageNet](https://www.image-net.org/) data to `~/data_imagenet`.

### Quantization

**CIFAR-10**

```bash
# Dataset bin generation (By default we use a bin number of 10)
CUDA_VISIBLE_DEVICES=0 python -u quantize_sample.py \
    --fraction 0.1 --dataset CIFAR10 --data_path ~/data_cifar \
    --num_exp 10 --workers 10 -se 0 --selection Submodular --model ResNet18 \
    -sp ../results/bin_cifar_010 \
    --batch 128 --submodular GraphCut --submodular_greedy NaiveGreedy --pretrained

# Bin sampling (Change the fraction parameter to obtain different data keep ratio)
CUDA_VISIBLE_DEVICES=0 python -u quantize_bin.py \
    --fraction 0.125 --dataset CIFAR10 --data_path ~/data_cifar \
    --workers 10 --selection_path ../results/bin_cifar_010/ \
    -sp ../results/sample_quantized_cifar_010

# Image reconstruction (Change the mask_ratio parameter to obtain different patch drop ratio)
CUDA_VISIBLE_DEVICES=0 python -u quantize_pixel.py \
    --data CIFAR10 --data_path ~/data_cifar \
    --output_dir ../results/pixel_quantized_cifar --model mae_vit_large_patch16 \
    --resume ./pretrained/mae_visualize_vit_large_ganloss.pth --batch_size 128 \
    --mask_ratio 0.2 --cam_mask
```

**ImageNet**

```bash
# Dataset bin generation (By default we use a bin number of 10)
CUDA_VISIBLE_DEVICES=0 python -u quantize_sample.py \
    --fraction 0.1 --dataset ImageNet --data_path ~/data_imagenet \
    --num_exp 10 --workers 10 -se 0 --selection Submodular --model ViT_Base_16 \
    -sp ../results/bin_imagenet_010 \
    --batch 128 --submodular GraphCut --submodular_greedy NaiveGreedy --pretrained

# Bin sampling (Change the fraction parameter to obtain different data keep ratio)
CUDA_VISIBLE_DEVICES=0 python -u quantize_bin.py \
    --fraction 0.125 --dataset ImageNet --data_path ~/data_imagenet \
    --workers 10 --selection_path ../results/bin_imagenet_010/ \
    -sp ../results/sample_quantized_imagenet_010

# Image reconstruction (Change the mask_ratio parameter to obtain different patch drop ratio)
CUDA_VISIBLE_DEVICES=0 python -u quantize_pixel.py \
    --data ImageNet --data_path ~/data_imagenet/train \
    --output_dir ../results/pixel_quantized_imagenet --model mae_vit_large_patch16 \
    --resume ./pretrained/mae_visualize_vit_large_ganloss.pth --batch_size 128 \
    --mask_ratio 0.2 --cam_mask
```

### Training

For data keep ratio higher than 10%, we use a batch size of 128. Otherwise, we use a batch size of 16 for sufficient training. 

Note that the final data keep ratio is the multiplication of the fraction in bin sampling and the patch keep ratio in image reconstruction. 

**CIFAR-10**

```bash
CUDA_VISIBLE_DEVICES=0 python validate_cifar.py \
    --data_dir ../results/pixel_quantized_cifar/ \
    --select_indices ../results/sample_quantized_cifar_010/select_indices_CIFAR10_0.125.npy \
    --batch_size 16
```

**ImageNet**

We use `timm` for evaluating the quantized ImageNet data.
For more instructions you can refer to the [README](./pytorch_image_models/README.md) inside the pytorch_image_models directory or the official [timm repo](https://github.com/huggingface/pytorch-image-models). 

```bash
cd pytorch_image_models
sh distributed_train.sh 8 \
    ../../results/pixel_quantized_imagenet ~/data_imagenet/val \
    --select-indices ../../results/sample_quantized_imagenet_010/select_indices_ImageNet_0.125.npy \
    --output ../../results/training_logs \
    --model resnet50 --sched cosine --epochs 260 --lr 0.6 --reprob 0.6 --remode pixel \
    --batch-size 128 --amp --aug-splits 3 -aa rand-m9-mstd0.5-inc1 --resplit --split-bn --jsd --dist-bn reduce
```

## DQ for Instruction Fine-tuning

Here we provide the code of DQ to compress the instruction fine-tunning datasets [alpaca](https://github.com/tatsu-lab/stanford_alpaca/tree/main), which consists of 52K instructions. To compress the dataset, we first extract the embeddings for each instruction with response by OpenAI Embedding API, and then use DQ to sample a fraction of dataset.

### Embedding Extraction

The extracted embeddings can be downloaded from this [link](https://drive.google.com/file/d/1cd9b_EyFpvtzBo2rLDtsSdzBfEZszMAN/view?usp=sharing). The generation fees are smaller than $1.

Optionally, you can generate by yourself with the following steps, the index and num parameter can be used for parallelization.

```bash
python embed.py --index 0 --nums 10000
python embed.py --index 1 --nums 10000
python embed.py --index 2 --nums 10000
python embed.py --index 3 --nums 10000
python embed.py --index 4 --nums 10000
python embed.py --index 5 --nums 10000
```

Then, you can merge the embeddings with the following command.

```bash
python alpaca_embed.py --merge
```

### DQ Sampling

To generate the sampled dataset, you can run the following command.

```bash
python alpaca_sample.py --ratio 0.1 --k 10
```

We provided some sampled results in the data folder for your reference.

### Training

We use [stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca/tree/main) to finetune the 7B llama model. Please follow the instructions in the repo to run the finetuning. For 1k sampled dataset, we use the following command to finetune the model. The hyper-parameter comes from the [LIMA](https://arxiv.org/abs/2305.11206) paper.

```bash
torchrun --nproc_per_node=8 --master_port=<your_random_port> train.py \
    --model_name_or_path <your_path_to_hf_converted_llama_ckpt_and_tokenizer> \
    --output_dir ./data/alpaca_data_k5_1k.json \
    --data_path <your_output_dir> \
    --bf16 True \
    --num_train_epochs 15 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --adam_beta2 0.95 \
    --weight_decay 0.1 \
    --lr_scheduler_type "linear" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True
```

### Evaluation

We use [instruct-eval](https://github.com/declare-lab/instruct-eval) repo to evaluate the finetuned model. Please follow the instructions in the repo to run the evaluation.

## Acknowledgement

This project is mainly developed based on the following repos:
- [DeepCore](https://github.com/PatrickZH/DeepCore)
- [pytorch-image-models](https://github.com/huggingface/pytorch-image-models)
- [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
