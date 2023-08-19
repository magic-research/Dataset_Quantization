'''Train CIFAR10 with PyTorch.'''
import os
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from dq.nets import ResNet18


# Training
def train(epoch, net, trainloader, criterion, optimizer, device):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    accuracy = 0.
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader), total=len(trainloader))
    for batch_idx, (inputs, targets) in pbar:
        pbar.set_description('Loss: {:.3f} Acc: {:.2%}'.format(
            train_loss / (batch_idx + 1), accuracy))

        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        accuracy = correct / total


def test(args, best_acc, epoch, net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    accuracy = 0.
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(testloader), total=len(testloader))
        for batch_idx, (inputs, targets) in pbar:
            pbar.set_description('Loss: {:.3f} Acc: {:.2%}'.format(
            test_loss / (batch_idx + 1), accuracy))
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            accuracy = correct / total

    # Save checkpoint.
    if args.result_path != '' and accuracy > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': accuracy,
            'epoch': epoch,
        }
        if not os.path.isdir(args.result_path):
            os.mkdir(args.result_path)
        torch.save(state, os.path.join(args.result_path, 'ckpt.pth'))
        best_acc = accuracy

    return best_acc


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='resume from checkpoint')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--data_dir', default='', type=str)
    parser.add_argument('--select_indices', default=[], type=str, nargs='+',
                        help='pre-defined subset indices')
    parser.add_argument('--result_path', default='', type=str,
                        help='dynamic save path, leave empty if not saving')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if args.data_dir == '':
        trainset = torchvision.datasets.CIFAR10(
            root='/data/personal/nus-gjy/data', train=True, download=True, transform=transform_train)
    else:
        trainset = ImageFolder(root=args.data_dir, transform=transform_train)

    if len(args.select_indices) > 0:
        select_indices = np.array([]).astype(int)
        for indices in args.select_indices:
            select_indices = np.append(select_indices, np.load(indices))
        trainset = torch.utils.data.Subset(trainset, select_indices)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='/data/personal/nus-gjy/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2)

    # Model
    print('==> Building model..')
    net = ResNet18(channel=3, num_classes=10, im_size=(32, 32))
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(200):
        train(epoch, net, trainloader, criterion, optimizer, device)
        best_acc = test(args, best_acc, epoch, net, testloader, criterion, device)
        scheduler.step()
        if len(args.select_indices) > 0:
            index_names = '-'.join([index.split('/')[-1][:-4] for index in args.select_indices])
            with open(osp.join(args.result_path, index_names+'.txt'), 'a') as fp:
                fp.write(str(epoch) + ' ' + str(best_acc) + '\n')
    print(best_acc)


if __name__ == '__main__':
    main()
