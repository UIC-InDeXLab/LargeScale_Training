'''
Helper class to facilitate experiments with different k
'''
import torch.nn.functional as F
import sys
import time
from statistics import mean
import psutil
import torch
import torch.cuda
import matplotlib.pyplot  as plt
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from sklearn.metrics import classification_report, roc_auc_score, multilabel_confusion_matrix
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

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
                 devset=None,
                 tstset=None,
                 cudatensor=False,
                 file=sys.stdout):
        self.args = args
        self.mb = mb
        self.hidden = hidden
        self.layer = layer
        #self.inputs = inputs
        #self.outputs = outputs
        self.file = file
        self.trnset = trnset
        # self.unified = unified

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
                num_workers=0,
                pin_memory=False)
            if devset:
                self.devloader = torch.utils.data.DataLoader(
                    devset,
                    batch_size=mb,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False)
            else:
                self.devloader = None

            if tstset:
                self.testloader = torch.utils.data.DataLoader(
                    tstset,
                    batch_size=mb,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False)
            else:
                self.testloader = None
        self.basettime = None
        self.basebtime = None
        self.name = 'norb'
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
            num_workers=0,
            pin_memory=True)

    def _train(self, model, opt):
        '''
        Train the given model using the given optimizer
        Record the time and loss
        '''
        model.train()
        model.time = 0
        ftime = 0
        btime = 0
        utime = 0
        tloss = 0

        for bid, (data, target) in enumerate(self.trainloader):
            
            target = target.type(torch.LongTensor)
            data, target = data.to(self.args.device), target.view(-1).to(self.args.device)
            t = 50 if bid <= 1000 else 100
            if bid % t == 0 and bid != 0:
                model.update_tables()
            opt.zero_grad()

            output = model(data)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            # correct += pred.eq(target.data).cpu().sum()

            loss = F.nll_loss(output, target)
            loss.backward()
            opt.step()
            tloss += loss.item()

        tloss /= len(self.trainloader)
        htime = model.time
        model.time = 0
        return tloss, htime, ftime, btime, utime

    def _evaluate(self, model, loader, file, name='test'):
        '''
        Use the given model to classify the examples in the given data loader
        Record the loss and accuracy.
        '''
        model.eval()
        test_loss = 0
        correct = 0
        y_predict = []
        y_target = []
        #print(loader)
        for data, target in loader:
            data, target = Variable(
                data, requires_grad=False).to(self.args.device), Variable(target.type(torch.LongTensor)).to(self.args.device)
            output = model(data)
            test_loss += F.nll_loss(output, target).item()
            pred = output.data.argmax(dim=1) # get the index of the max log-probability
            correct += pred.eq(target.data).cpu().sum()
            y_predict.append(pred.detach().numpy().flatten())
            y_target.append(target.detach().numpy().flatten())


        labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        if name == 'test':
            file.write(classification_report(y_target, y_predict, target_names=labels))
            file.write('\n'+'-----'*10 +'\n')
        y_predict = np.array(y_predict).flatten()
        y_target = np.array(y_target).flatten()

        # test_loss = test_loss
        test_loss /= len(loader)  # loss function already averages over batch size
        print(
            '{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
                name, test_loss, correct,
                len(loader.dataset), 100. * float(correct) / len(loader.dataset)),
            file=self.file,
            flush=True)
        model.random_nodes()
        return 100. * float(correct) / len(loader.dataset), y_predict, y_target

    def roc_auc(self, y_target, y_pred):

        y_p = (y_pred == np.unique(y_pred,)[ :, None]).astype(int)
        y_t = (y_target == np.unique(y_target,)[:, None]).astype(int)
        macro_roc_auc_ovr = roc_auc_score(np.transpose(y_t), np.transpose(y_p), multi_class="ovr", average="macro")
        weighted_roc_auc_ovr = roc_auc_score(np.transpose(y_t), np.transpose(y_p), multi_class="ovr", average="weighted")

        return macro_roc_auc_ovr, weighted_roc_auc_ovr

    def plot_confusion_matrix(self, cm, num_layer, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        import itertools
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        plt.figure(1)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('confiusion_matrix'+self.name+str(num_layer)+'.png')


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

        model = MLP(self.hidden, self.layer, self.args.n_inputs, self.args.n_outputs)
        model.reset_parameters()

        # model.cpu()

        opt = optim.Adam(model.parameters(), eps=1e-4)

        acc = 0  # best dev. acc.
        accc = 0  # test acc. at the time of best dev. acc.
        e = -1  # best dev iteration/epoch

        losses = []
        ftime =[]
        btime=[]
        utime = []
        weighted_auc = []
        macro_auc = []
        file_object = open('classification_report'+self.name + str(model.layer) + '.txt', 'a')
        print('Initial evaluation on dev set:')
        self._evaluate(model, self.devloader,file_object, 'dev')

        start = time.time()
        # training loop

        for t in range(epoch):
            print('{}：'.format(t), end='', file=self.file, flush=True)
            # train
            loss, ptime, ft, bt, ut = self._train(model, opt)
            model.update_tables()
            print("(wall time: {:.1f} sec) ".format(time.time() - start), end='')
            # times.append(ttime)
            model.timing()
            print("(feedfoward time: {:.3f} sec) ".format(model.forward_time), end='')
            losses.append(loss)
            ftime.append(ft)
            btime.append(bt)
            utime.append(ut)
            # pre_proc.append(ptime)
            print("(preprocess time: {:.1f} sec) ".format(ptime), end='\n')
            # predict
            curacc, y_pred, y_target = self._evaluate(model, self.devloader, file_object, 'dev')
            

        epoch = [i for i in range(epoch)]
        
        etime = [sum(t) for t in zip(ftime, btime, utime)]
        curacc, y_pred, y_target =self._evaluate(model, self.testloader, file_object, 'test')
        cnf_matrix = confusion_matrix(y_target, y_pred)
        self.plot_confusion_matrix(cnf_matrix, model.layer, classes = [0, 1, 2, 3, 4],normalize=False, title='Confusion matrix, without normalization')

        print('test acc: {:.2f}'.format(curacc))
        print(
            'best on val set - ${:.2f}|{:.2f} at {}'.format(acc, accc, e),
            file=self.file,
            flush=True)
        print('', file=self.file)
        file_object.close()
        torch.save(model, "trained_model_norb_"+model.layer)
    def _stat(self, name, t, agg=mean):
        return '{:<5}:\t{:8.3f}; {}'.format(
            name, agg(t), ', '.join(['{:8.2f}'.format(x) for x in t]))
