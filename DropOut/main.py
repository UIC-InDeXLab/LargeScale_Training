from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot  as plt
from torch.autograd import Variable
from layers import Standout
from utils import saveLog
from sklearn.metrics import classification_report, roc_auc_score, multilabel_confusion_matrix
import numpy as np
from torch.autograd import Variable

from sklearn.metrics import confusion_matrix
from data import get_mnist
import time
# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.99)')
parser.add_argument(
        '--dev', type=str, default="cpu", help='specify "cuda" or "cpu"')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--standout', action='store_true', default=False,
                    help='Activates standout training!')

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
# args.device = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 0, 'pin_memory': False} if args.dev == 'cuda' else {}

train, dev, test = get_mnist()
train_loader = torch.utils.data.DataLoader(
    train,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test,
    batch_size=args.test_batch_size, shuffle=True, **kwargs)
dev_loader = torch.utils.data.DataLoader(
    dev,
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self, standout, layer, hidden, input, output):
        super(Net, self).__init__()
        #### SELF ARGS ####
        self.standout = standout
        self.input = input
        self.output = output
        self.num_layers = layer
        self.num_units = hidden
        self.forward_time = 0
        #### MODEL PARAMS ####
        self.fc1 = nn.Linear(self.input, self.num_units)
        self.fc1_drop = Standout(self.fc1, 0.5, 1) if standout else nn.Dropout(0.95)
        self.fc = []
        self.fc_drop = []
        for i in range(layer):

            self.fc.append(nn.Linear(self.num_units, self.num_units))
            self.fc_drop.append(Standout(self.fc[i], 0.5, 1) if standout else nn.Dropout(0.95))


        self.fc_final = nn.Linear(self.num_units, self.output)

    def forward(self, x):
        # Flatten input
        start = time.time()
        x = x.view(-1, self.input)
        # Keep it for standout

        #FIRST FC
        previous = x
        x_relu = F.relu(self.fc1(x))
        # Select between dropouts styles
        x = self.fc1_drop(previous, x_relu) if self.standout else self.fc1_drop(x_relu)

        #SECOND FC
        for i in range (self.num_layers):
            previous = x
            x_relu = F.relu(self.fc[i](x))
        # Select between dropouts styles
            fc_drop = self.fc_drop[i]
            x = fc_drop(previous, x_relu) if self.standout else fc_drop(x_relu)

        x = self.fc_final(x)
        self.forward_time += (time.time()-start)
        return F.log_softmax(x, dim=1)
def test(model, loader, standout, epoch, file_obj, name = 'dev'):
    optimizer = optim.Adam(model.parameters(), eps = 1e-4)
    model.eval()
    test_loss = 0
    correct = 0
    y_predict = []
    y_target = []
    for data, target in loader:
        data, target = Variable(
            data, requires_grad=False).to(args.device), Variable(target.type(torch.LongTensor)).to(args.device)
        #if torch.cuda.is_available():
        #    data, target = data.cuda(), target.cuda()
        #data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()# sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        y_predict.append(pred.detach().numpy().flatten())
        y_target.append(target.detach().numpy().flatten())




    labels = ['0', '1', '2', '3', '4','5','6','7','8','9']
    if name == 'test':
        file_obj.write(classification_report(y_target, y_predict, target_names=labels))
        file_obj.write('\n' + '-----' * 10 + '\n')
    y_predict = np.array(y_predict).flatten()
    y_target = np.array(y_target).flatten()
    test_loss /= len(loader.dataset)
    test_acc = 100. * correct / len(loader.dataset)
    print(
        '{}- {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            epoch, name, test_loss, correct,
            len(loader.dataset), 100. * float(correct) / len(loader.dataset)),
        file=file_obj)
    if standout == True:
        drop_way = "Standout"
    else:
        drop_way = "Dropout"
   # saveLog(test_loss, test_acc, correct, drop_way, args, epoch)
    return test_acc, y_predict, y_target

def train(model, standout, epoch, file_obj):
    optimizer = optim.Adam(model.parameters(), eps = 1e-4)
    model.train()
    y_predict = []
    y_target = []
    correct = 0
    htime = 0
    btime = 0
    ftime =0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(args.device), target.view(-1).to(args.device)
        data, target = Variable(data), Variable(target.type(torch.LongTensor))
        optimizer.zero_grad()
        start = time.time()
        output = model(data)
        loss = F.nll_loss(output, target)
        ftime += time.time() -start
        start = time.time()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
        start = time.time()
        loss.backward()
        optimizer.step()
        btime += time.time()-start
    print("(feedfoward time: {:.3f} sec, backpropagation time: {:.3f} sec) ".format(ftime, btime), end='')

        # if batch_idx % args.log_interval == 0:
        #     print(batch_idx, len(data))
    test(model, dev_loader, standout, epoch, file_obj)
    #print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
     #           epoch, correct, len(train_loader.dataset),
    #            100. * correct/len(train_loader.dataset), loss.item()))




def plot_confusion_matrix(cm, num_layer, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Reds):
    """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      #  print("Normalized confusion matrix")
    #else:
      #  print('Confusion matrix, without normalization')

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
    plt.savefig('confiusion_matrix' + 'FMNIST' + str(num_layer) + '.png')



def run(standout=False):

    model = Net(standout, layer=3, hidden=1000, input=784, output=10)
    #if torch.cuda.is_available():
    #    model.cuda()
    name = 'MNIST'
    cmap=plt.cm.Greens
    if standout:
        cmap=plt.cm.Reds
        name = 'Standout_MNIST'

    file_object = open('classification_report' + name + str(model.num_layers) + '.txt', 'a')
    test(model, dev_loader, standout, args.epochs, file_object)
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        train(model, standout, epoch, file_object)
        model.forward_time = 0
        print("(wall time: {:.1f} sec) ".format(time.time() - start), end='\n')

    
    curacc, y_pred, y_target = test(model, test_loader, standout, args.epochs, file_object, 'test')
    cnf_matrix = confusion_matrix(y_target, y_pred)
    plot_confusion_matrix(cnf_matrix, model.num_layers, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8,9], normalize=False,
                               title='Confusion Matrix'+name + str(model.num_layers), cmap=cmap)

    file_object.close()
def main():
    print("RUNNING STANDOUT ONE")
    run(standout=True)

    print("RUNNING DROPOUT ONE")
    run()

if __name__ == "__main__":
    main()

