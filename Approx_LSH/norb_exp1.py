
import sys
from argparse import ArgumentParser
import tracemalloc
import torch

from data import get_norb
#from data import get_mnist
from util_norb import TestGroup


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
        default=3,
        help='number of layers, including the output layer')
    parser.add_argument(
        '--n_inputs',
        type=int,
        default=9216,
        help='number of inputs')
    parser.add_argument(
        '--n_outputs',
        type=int,
        default=5,
        help='number of outputs')
    
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
    trn, dev, tst = get_norb()   #mnist()

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

    tracemalloc.start()
    # show how much RAM the above code allocated and the peak usage
    group.run()
    # print(prof.key_averages())
    current, peak =  tracemalloc.get_traced_memory()
    print("Run function RAM")
    print(f"{current:0.2f}, {peak:0.2f}")
    tracemalloc.stop()
if __name__ == '__main__':
    tracemalloc.start()
    main()
    current, peak =  tracemalloc.get_traced_memory()
    print("Total RAM")
    print(f"{current:0.2f}, {peak:0.2f}")
    tracemalloc.stop()
