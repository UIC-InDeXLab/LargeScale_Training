
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch.nn.utils import weight_norm as wn
sys.path.append('..')
import numpy as np
from module import LSH_Layer

K=6 #10
L=5 #32
m=3


class MLP(nn.Module):

    def __init__(self, hidden_nodes, layer, inputs, outputs):
        super(MLP, self).__init__()
        self.layer = layer
        self.inputs = inputs
        self.outputs = outputs
        self.model = nn.Sequential(self._create(hidden_nodes, layer,inputs, outputs, K=K, L=L, m=m))
        self.time = 0
        self.forward_time = 0
        self.backward_time = 0
        self.actives = []
        self.inactives =[]
    
    def _create(self, hidden, layer,inputs, outputs, K, L, m):

        d = OrderedDict()
        if layer == 0:
            d['linear0'] = wn(LSH_Layer(inputs, outputs, t=1))
            return d

        for i in range(layer):
            if i == 0:
                d['linear' + str(i)] = wn(LSH_Layer(inputs, hidden, t=0, K=K, L=L, m=m))
                d['relu' + str(i)] = nn.LeakyReLU()#nn.ReLU()
                # d['si' + str(i)] = nn.Sigmoid()


            elif i == layer - 1:
                d['linear' + str(i)] = wn(LSH_Layer(hidden, hidden, t= 0, K=K, L=L, m=m))
                d['relu' + str(i)] =nn.LeakyReLU() #nn.ReLU()
                # d['si' + str(i)] = nn.Sigmoid()

            else:
                d['linear' + str(i)] = wn(LSH_Layer(hidden, hidden, t= 0, K=K, L=L, m=m))
                # d['si' + str(i)] = nn.Sigmoid()
                d['relu' + str(i)] =nn.LeakyReLU()# nn.ReLU()

        d['linear' + str(layer)] = wn(LSH_Layer(hidden, outputs, t=1, K=K, L=L, m=m))
        # d['relu' + str(layer)] = nn.ReLU()

        return d
    def timing(self):
        for module in self.modules():
            if isinstance(module, LSH_Layer):
                self.forward_time += module.f
                module.f = 0
    def head_tails(self):
        data = []
        for module in self.modules():
            if isinstance(module, LSH_Layer):
                self.actives.append(module.actives)
                module.actives = np.zeros(module.actives.shape)
                self.inactives.append(module.inactives)
                module.inactives = np.zeros(module.actives.shape)
        ac = self.actives
        inac = self.inactives
        return ac, inac, data

    def weights_init(self, m):
        nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    def update_tables(self):
        for module in self.modules():
            if isinstance(module, LSH_Layer) and module.lsh != None:
                module.lsh.clear()
                self.time += module.build_lsh_tables()

    def random_nodes(self):
        for module in self.modules():
            if isinstance(module, LSH_Layer) and module.lsh != None:
                module.lsh.stats()

    def forward(self, x):
        t = self.model(x.view(-1, self.inputs))
        output = F.log_softmax(self.model(x.view(-1, self.inputs)), dim=-1)
        return output

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, LSH_Layer):
                self.weights_init(m)
                if m.lsh != None:
                    self.time += m.build_lsh_tables()




