'''
Helper class to facilitate experiments with different k
'''
import sys
import time
from statistics import mean
import psutil
import torch
import torch.cuda
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

from model import MLP


class TestGroup(object):
    '''
    A network and k in meporp form a test group.
    Test groups differ in minibatch size, hidden features, layer number and dropout rate.
    '''

    def __init__(self,
                 args,
                 trnset,
                 mb,
                 hidden,
                 layer,
                 unified,
                 devset=None,
                 tstset=None,
                 cudatensor=False,
                 file=sys.stdout):
        self.args = args
        self.mb = mb
        self.hidden = hidden
        self.layer = layer
        self.file = file
        self.trnset = trnset
        self.unified = unified

        if cudatensor:  # dataset is on GPU
            self.trainloader = torch.utils.data.DataLoader(
                trnset, batch_size=mb, num_workers=0)
            if tstset:
                self.testloader = torch.utils.data.DataLoader(
                    tstset, batch_size=mb, num_workers=0)
            else:
                self.testloader = None
        else:  # dataset is on CPU, using prefetch and pinned memory to shorten the data transfer time
            self.trainloader = torch.utils.data.DataLoader(
                trnset,
                batch_size=mb,
                shuffle=True,
                num_workers=1,
                pin_memory=False)
            if devset:
                self.devloader = torch.utils.data.DataLoader(
                    devset,
                    batch_size=mb,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=False)
            else:
                self.devloader = None
            if tstset:
                self.testloader = torch.utils.data.DataLoader(
                    tstset,
                    batch_size=mb,
                    shuffle=False,
                    num_workers=1,
                    pin_memory=False)
            else:
                self.testloader = None
        self.basettime = None
        self.basebtime = None

    def reset(self):
        '''
        Reinit the trainloader at the start of each run,
        so that the traning examples is in the same random order
        '''
        torch.manual_seed(self.args.random_seed)
        self.trainloader = torch.utils.data.DataLoader(
            self.trnset,
            batch_size=self.mb,
            shuffle=True,
            num_workers=1,
            pin_memory=True)

    def _train(self, model, opt):
        '''
        Train the given model using the given optimizer
        Record the time and loss
        '''
        model.train()
        ftime = 0
        btime = 0
        utime = 0
        tloss = 0
        for bid, (data, target) in enumerate(self.trainloader):
            data, target = data.to(self.args.device), target.view(-1).to(self.args.device)
            opt.zero_grad()
            s = time.time()
            output = model(data)
            loss = F.nll_loss(output, target)
            ftime += time.time()-s
            s = time.time()
            loss.backward()
            btime = time.time()-s
            opt.step()
            tloss += loss.item()

        tloss /= len(self.trainloader)

        return tloss, ftime, btime, utime

    def _evaluate(self, model, loader, name='test'):
        '''
        Use the given model to classify the examples in the given data loader
        Record the loss and accuracy.
        '''
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in loader:
            data, target = Variable(
                #data, requires_grad=False).cuda(), Variable(target).cuda()
                data, requires_grad=False).to(self.args.device), Variable(target).to(self.args.device)
            output = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()




        # test_loss = test_loss
        test_loss /= len(loader)  # loss function already averages over batch size
        print(
            '{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                name, test_loss, correct,
                len(loader.dataset), 100. * float(correct) / len(loader.dataset)),
            file=self.file,
            flush=True)
        return 100. * float(correct) / len(loader.dataset)

    def run(self, k=None, epoch=None):
        '''
        Run a training loop.
        '''
        if epoch is None:
            epoch = self.args.n_epoch
        print(
            'mbsize: {}, hidden size: {}, layer: {} '.
            format(self.mb, self.hidden, self.layer),
            file=self.file)
        # Init the model, the optimizer and some structures for logging
        self.reset()

        model = MLP(self.hidden,self.layer)
        model.reset_parameters()

        # model.cpu()

        opt = optim.Adam(model.parameters())

        acc = 0  # best dev. acc.
        accc = 0  # test acc. at the time of best dev. acc.
        e = -1  # best dev iteration/epoch

        losses = []
        ftime =[]
        btime=[]
        utime = []
        print('Initial evaluation on dev set:')
        self._evaluate(model, self.devloader, 'dev')

        start = time.time()
        # training loop
        RAM_bytes_bf = int(psutil.virtual_memory().total - psutil.virtual_memory().available)

        for t in range(epoch):
            print('{}：'.format(t), end='', file=self.file, flush=True)
            # train
            loss, ft, bt, ut = self._train(model, opt)
            print("(wall time: {:.1f} sec) ".format(time.time() - start), end='')
            # times.append(ttime)
            print("(feedforward time: {:.1f} sec)(backprop time: {:.1f} sec) ".format(ft, bt), end='')
            losses.append(loss)
            ftime.append(ft)
            btime.append(bt)
            utime.append(ut)
            # predict
            curacc = self._evaluate(model, self.devloader, 'dev')

        etime = [sum(t) for t in zip(ftime, btime, utime)]
        print('test acc: {:.2f}'.format(self._evaluate(model, self.testloader, '    test')))
        print(
            'best on val set - ${:.2f}|{:.2f} at {}'.format(acc, accc, e),
            file=self.file,
            flush=True)
        print('', file=self.file)
        RAM_bytes = int(psutil.virtual_memory().total - psutil.virtual_memory().available)
        print('RAM usage after training is {} MB'.format(int(RAM_bytes / 1024 / 1024) - RAM_bytes_bf))

    def _stat(self, name, t, agg=mean):
        return '{:<5}:\t{:8.3f}; {}'.format(
            name, agg(t), ', '.join(['{:8.2f}'.format(x) for x in t]))
