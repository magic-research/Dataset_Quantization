import os
import argparse
import numpy as np
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageFolder

import mae_models


def get_args_parser():
    parser = argparse.ArgumentParser('MAE Reconstruction', add_help=False)
    parser.add_argument('--batch_size', default=1024, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--mask_ratio', default=0.2, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--cam_mask', action='store_true', default=False,
                        help='whether to use gradcam to select dropping patches')

    # Dataset parameters
    parser.add_argument('--data', default='cifar10', type=str, help='dataset name')
    parser.add_argument('--data_path', default='../data_cifar', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='../output/recons_cifar10_base',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='./mae_visualize_vit_large_ganloss.pth', help='resume from checkpoint')
    parser.add_argument('--num_workers', default=10, type=int)

    return parser


def prepare_model(chkpt_dir, arch='mae_vit_base_patch16', cam_mask=False):
    # build model
    model = getattr(mae_models, arch)(cam_mask=cam_mask)
    # load model
    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def run_images(x, model, mask_ratio=0.75):
    # run MAE
    x = x.cuda()
    loss, y, mask = model(x, mask_ratio=mask_ratio)
    y = model.unpatchify(y)

    if mask_ratio == 0.0:
        return y.cpu(), loss

    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0] ** 2 * 3)
    mask = model.unpatchify(mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    return im_paste.cpu(), loss


# return the original path together with the image
class ImageFolderWithPaths(ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        image = original_tuple[0]
        label = original_tuple[1]
        return image, label, path


if __name__ == '__main__':
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    args = get_args_parser()
    args = args.parse_args()
    
    transform_test = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])

    chkpt_dir = args.resume
    model_mae = prepare_model(chkpt_dir, args.model, args.cam_mask)
    model_mae.cuda()
    print('Model loaded.')
    if args.data == 'CIFAR10':
        dataset_train = CIFAR10(root=args.data_path, train=True, download=True, transform=transform_test)
    elif args.data == 'ImageNet':
        dataset_train = ImageFolderWithPaths(root=args.data_path, transform=transform_test)
    dataloader = DataLoader(
        dataset_train,
        batch_size = args.batch_size
    )
    total_loss = 0.0
    
    # reconstruct the datasets with MAE
    model_mae.eval()
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
    for idx, data in pbar:
        pbar.set_description('Loss: {:.3f}'.format(total_loss / (idx +1)))

        # maintain the image names
        if args.data == 'CIFAR10':
            image, labels = data
            labels = [label.item() for label in labels]
            image_names = np.arange(idx * args.batch_size, (idx + 1) * args.batch_size)
            image_names = [str(name)+'.png' for name in image_names]
        elif args.data == 'ImageNet':
            image, _, paths = data
            labels = []
            image_names = []
            for path in paths:
                label, image_name = path.split('/')[-2:]
                labels.append(label)
                image_names.append(image_name)

        torch.manual_seed(args.seed)
        recovery_img, loss = run_images(image, model_mae, args.mask_ratio)
        total_loss += loss.item()

        # save the reconstructed images
        for j in range(recovery_img.shape[0]):
            reconstruction_path = os.path.join(args.output_dir, str(labels[j]))
            if not os.path.exists(reconstruction_path):
                os.makedirs(reconstruction_path)
            fpath = os.path.join(reconstruction_path, image_names[j])

            # de-normalize
            recovery_img_j = torch.einsum('chw->hwc', recovery_img[j])
            recovery_img_j = recovery_img_j * imagenet_std + imagenet_mean
            recovery_img_j = torch.einsum('hwc->chw', recovery_img_j)

            # resize the image if belonging to CIFAR-10
            if args.data == 'CIFAR10':
                recovery_img_j = torchvision.transforms.functional.resize(recovery_img_j, [32, 32])

            save_image(recovery_img_j, fpath)
