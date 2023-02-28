'''
Codes for loading the MNIST data
'''

import numpy
import torch
from torchvision import datasets, transforms
import sys
sys.path.append('.')
from smallnorb.dataset import SmallNORBDataset
class PartDataset(torch.utils.data.Dataset):
    '''
    Partial Dataset:
        Extract the examples from the given dataset,
        starting from the offset.
        Stop if reach the length.
    '''

    def __init__(self, dataset, offset, length):
        self.dataset = dataset
        self.offset = offset
        self.length = length
        super(PartDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.dataset[i + self.offset]
import tensorflow_datasets as tfds
import torch
import pandas
from torch.utils.data import DataLoader


#plt.ion()
def get_fashionmnist(datapath='~/datasets/fashion_mnist', download=True):
    '''
    The MNIST dataset in PyTorch does not have a development set, and has its own format.
    We use the first 5000 examples from the training dataset as the development dataset. (the same with TensorFlow)
    Assuming 'datapath/processed/training.pt' and 'datapath/processed/test.pt' exist, if download is set to False.
    '''
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
    trn = datasets.FashionMNIST(
        datapath,
        train=True,
        download=download,
        transform=transform)
    dev = PartDataset(trn, 0, 5000)
    trnn = PartDataset(trn, 5000, 55000)
    tst = datasets.FashionMNIST(datapath, train=False, transform=transform)

    return trnn, dev, tst

def get_norb():


    transform = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean = (0.2502,),
                         std= (0.1755,))])
    dataset = SmallNORBDataset(dataset_root='/home/sana/small_norb/smallnorb', length= 24300)
    #train_data = dataset.dataset['train']
    #data = transform(dataset)
    dev = PartDataset(dataset, 0, 2000)
    trnn = PartDataset(dataset, 2000, 22300)
    test = dataset2 = SmallNORBDataset(dataset_root='/home/sana/small_norb/smallnorb', length= 24300, training='False')
    #train_data = dataset.dataset['train']
    #image_data_loader = DataLoader(trnn,
    # batch size is whole dataset
    #batch_size=len(trnn),
    #shuffle=False,
    #num_workers=0)
    #def mean_std(loader):
    #    images, lebels = next(iter(loader))
        # shape of images = [b,c,w,h]
    #    print(images.shape)
    #    mean, std = images.mean([0,1,2]), images.std([0, 1,2])
    #    return mean, std
    #mean, std = mean_std(image_data_loader)
    #print(mean, std)
    return trnn, dev, test

def get_mnist(datapath='~/datasets/mnist', download=True):
    '''
    The MNIST dataset in PyTorch does not have a development set, and has its own format.
    We use the first 5000 examples from the training dataset as the development dataset. (the same with TensorFlow)
    Assuming 'datapath/processed/training.pt' and 'datapath/processed/test.pt' exist, if download is set to False.
    '''
    trn = datasets.MNIST(
        datapath,
        train=True,
        download=download,
        transform=transforms.ToTensor())
    dev = PartDataset(trn, 0, 5000)
    trnn = PartDataset(trn, 5000, 55000)
    tst = datasets.MNIST(
        datapath, train=False, transform=transforms.ToTensor())
    return trnn, dev, tst
