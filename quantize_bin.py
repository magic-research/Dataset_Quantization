import os
import time
import torch
import argparse
import numpy as np

import dq.methods as methods
import dq.datasets as datasets
from util.utils import str_to_bool


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')

    # Basic arguments
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--selection', type=str, default="Uniform", help="selection method")
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--print_freq', '-p', default=50, type=int, help='print frequency (default: 20)')
    parser.add_argument('--fraction', default=0.1, type=float, help='fraction of data to be selected (default: 0.1)')
    parser.add_argument('--seed', default=int(time.time() * 1000) % 100000, type=int, help="random seed")
    parser.add_argument('--balance', default=True, type=str_to_bool,
                        help="whether balance selection is performed per class")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # Algorithm
    parser.add_argument('--submodular', default="GraphCut", help="specifiy submodular function to use")
    parser.add_argument('--submodular_greedy', default="NaiveGreedy", help="specifiy greedy algorithm for submodular optimization")
    parser.add_argument('--uncertainty', default="Entropy", help="specifiy uncertanty score to use")
    parser.add_argument('--replace', action='store_true', default=False, help='whether the samples can be selected repeatedly')

    # Checkpoint and resumption
    parser.add_argument('--selection_path', type=str, default='', help='path to previous selection results')
    parser.add_argument('--save_path', "-sp", type=str, default='', help='path to save results (default: do not save)')

    args = parser.parse_args()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.save_path != "" and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    select_indices_files = sorted(os.listdir(args.selection_path))
    select_indices_list = [np.load(os.path.join(args.selection_path, fp)) for fp in select_indices_files]

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset] \
        (args.data_path)
    args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names

    torch.random.manual_seed(args.seed)
    print(args)

    all_mapped_indices = np.array([], dtype=np.int64)

    # conduct non-overlapping uniform sampling from all the bins
    for exp, select_indices in enumerate(select_indices_list):
        print('\n================== Exp %d ==================\n' % exp)
        sub_dst_train = torch.utils.data.Subset(dst_train, select_indices)
        print('Exp: {}, Dst Size: {}, Fraction: {}'.format(exp, len(sub_dst_train), args.fraction))
        method = methods.__dict__[args.selection](sub_dst_train, args, args.fraction, args.seed, balance=args.balance)
        subset = method.select()
        mapped_indices = select_indices[subset["indices"]]
        print('Exp: {}, Available indices: {}, Select Subset: {}'.format(exp, len(select_indices), len(mapped_indices)))
        all_mapped_indices = np.append(all_mapped_indices, mapped_indices)

    # save the selection results
    select_save_path = os.path.join(
        args.save_path, 'select_indices_{}_{}.npy'.format(args.dataset, args.fraction)
    )
    np.save(select_save_path, all_mapped_indices)


if __name__ == '__main__':
    main()
