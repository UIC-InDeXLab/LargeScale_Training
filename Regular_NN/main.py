
import sys
from argparse import ArgumentParser

import torch

from data import get_norb
from util_mnist import TestGroup


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
            '--n_epoch', type=int, default=50
            , help='number of training epochs')
    parser.add_argument(
        '--d_hidden', type=int, default=1000, help='dimension of hidden layers')
    parser.add_argument(
        '--n_layer',
        type=int,
        default=1,
        help='number of layers, including the output layer')
    parser.add_argument(
        '--d_minibatch', type=int, default=1, help='size of minibatches')
    parser.add_argument(
        '--random_seed', type=int, default=42, help='random seed')
    parser.add_argument(
        '--dev', type=str, default="cpu", help='specify "cuda" or "cpu"')
    parser.set_defaults()
    args = parser.parse_args()
    if args.dev == 'cuda':
        if torch.cuda.is_available():
            print('using cuda')
            args.device = torch.device('cuda')
        else:
            print('requested cuda device but cuda unavailable. using cpu instead')
            args.device = torch.device('cpu')
    else:
        print('using cpu')
        args.device = torch.device('cpu')

    return args


def main():
    args = get_args()
    trn, dev, tst = get_norb()

    # change the sys.stdout to a file object to write the results to the file
    group = TestGroup(
        args,
        trn,
        args.d_minibatch,
        args.d_hidden,
        args.n_layer,
        dev,
        tst,
        cudatensor=False,
        file=sys.stdout)

    # results may be different at each run
    # with torch.autograd.profiler.profile(use_cpu=True) as prof:
    group.run()
    # print(prof.key_averages())

if __name__ == '__main__':
    main()
