# PyTorch Image Models

Employ `timm` for ImageNet evaluation.

## Training

Different from standard `timm` scripts, we separate the root directory of train and eval data, as the images are reconstructed in the quantization process. 
Besides, you can change the `select_indices` parameter to specify the sample-level quantized sample indices. Multiple indices can be specified here. 
We use the `ResNet50` model as the template here. For the other models, you can refer to the [timm documentation](https://rwightman.github.io/pytorch-image-models/) and conduct the above modifications. 

```bash
sh distributed_train.sh 9 [TRAIN_ROOT] [EVAL_ROOT] --select-indices [INDICES1] [INDICES2] --output [OUTPUT_DIR] --model resnet50 --sched cosine --epochs 260 --lr 0.6 --reprob 0.6 --remode pixel --batch-size 128 --amp --aug-splits 3 -aa rand-m9-mstd0.5-inc1 --resplit --split-bn --jsd --dist-bn reduce
```

## Getting Started (Documentation)

Current [documentation](https://rwightman.github.io/pytorch-image-models/) for `timm` covers the basics.

[Getting Started with PyTorch Image Models (timm): A Practitioner’s Guide](https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055) by [Chris Hughes](https://github.com/Chris-hughes10) is an extensive blog post covering many aspects of `timm` in detail.

[timmdocs](http://timm.fast.ai/) is quickly becoming a much more comprehensive set of documentation for `timm`. A big thanks to [Aman Arora](https://github.com/amaarora) for his efforts creating timmdocs.

[paperswithcode](https://paperswithcode.com/lib/timm) is a good resource for browsing the models within `timm`.
