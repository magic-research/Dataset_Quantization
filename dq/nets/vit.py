import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled, flatten, Tensor
from .nets_utils import EmbeddingRecorder
from timm.models import vision_transformer


class Transformer(vision_transformer.VisionTransformer):
    def __init__(self, patch_size, embed_dim, depth, num_heads,
                 record_embedding, no_grad, **kwargs):
        super().__init__(patch_size=patch_size, embed_dim=embed_dim,
                         depth=depth, num_heads=num_heads, **kwargs)
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.head

    def forward(self, x: Tensor) -> Tensor:
        with set_grad_enabled(not self.no_grad):
            x = self.forward_features(x)
            self.embedding_recorder(x[:, 0])
            x = self.forward_head(x)

        return x


def TransformerBuilder(arch: str, num_classes: int, record_embedding: bool = False,
                       no_grad: bool = False, pretrained: bool = False):
    arch = arch.lower()
    if arch == 'vit_base_patch16_224':
        net = Transformer(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                          record_embedding=record_embedding, no_grad=no_grad)
    elif arch == 'vit_large_patch16_224':
        net = Transformer(patch_size=16, embed_dim=1024, depth=24, num_heads=16,
                          record_embedding=record_embedding, no_grad=no_grad)

    if pretrained:
        from timm.models.helpers import resolve_pretrained_cfg, load_custom_pretrained
        pretrained_cfg = resolve_pretrained_cfg(arch)
        load_custom_pretrained(net, pretrained_cfg=pretrained_cfg)

    if num_classes != 1000:
        net.head = nn.Linear(net.embed_dim, num_classes)

    return net


def MAEBuilder(arch: str, num_classes: int, record_embedding: bool = False,
               no_grad: bool = False, pretrained: bool = False):
    arch = arch.lower()
    if arch == 'vit_base_patch16_224':
        url = 'https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_base.pth'
        net = Transformer(patch_size=16, embed_dim=768, depth=12, num_heads=12,
                          record_embedding=record_embedding, no_grad=no_grad)
    elif arch == 'vit_large_patch16_224':
        url = 'https://dl.fbaipublicfiles.com/mae/finetune/mae_finetuned_vit_large.pth'
        net = Transformer(patch_size=16, embed_dim=1024, depth=24, num_heads=16,
                          record_embedding=record_embedding, no_grad=no_grad)

    if pretrained:
        from timm.models.helpers import resolve_pretrained_cfg, load_pretrained
        pretrained_cfg = resolve_pretrained_cfg(arch)
        pretrained_cfg['url'] = url
        load_pretrained(net, pretrained_cfg=pretrained_cfg,
                        filter_fn=vision_transformer.checkpoint_filter_fn,
                        strict=False)

    if num_classes != 1000:
        net.head = nn.Linear(net.embed_dim, num_classes)

    return net


def ViT_Base_16(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
                pretrained: bool = False, **kwargs):
    return TransformerBuilder('vit_base_patch16_224', num_classes=num_classes,
                              record_embedding=record_embedding, no_grad=no_grad,
                              pretrained=pretrained, **kwargs)


def MAE_Base_16(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
                pretrained: bool = False, **kwargs):
    return MAEBuilder('vit_base_patch16_224', num_classes=num_classes,
                      record_embedding=record_embedding, no_grad=no_grad,
                      pretrained=pretrained, **kwargs)


def ViT_Large_16(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = False, **kwargs):
    return TransformerBuilder('vit_large_patch16_224', num_classes=num_classes,
                              record_embedding=record_embedding, no_grad=no_grad,
                              pretrained=pretrained, **kwargs)


def MAE_Large_16(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = False, **kwargs):
    return MAEBuilder('vit_large_patch16_224', num_classes=num_classes,
                      record_embedding=record_embedding, no_grad=no_grad,
                      pretrained=pretrained, **kwargs)

