
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from torch import linalg as LA

sys.path.append('..')

class MLP(nn.Module):

    def __init__(self, hidden_nodes, layer):
        super(MLP, self).__init__()
        self.layer = layer
        self.model = nn.Sequential(self._create(hidden_nodes, layer))

    def _create(self, hidden, layer):
        if layer == 0:
            d = OrderedDict()
            d['linear0'] = torch.nn.modules.Linear(784, 10,bias=True)
            d['si0'] = nn.Sigmoid()

            return d
        d = OrderedDict()
        for i in range(layer):
            if i == 0:
                d['linear' + str(i)] = torch.nn.modules.Linear(784, hidden, bias=True)
                # d['relu' + str(i)] = nn.ReLU()
                d['si' + str(i)] = nn.Sigmoid()


            elif i == layer - 1:
                d['linear' + str(i)] = torch.nn.modules.Linear(hidden, hidden, bias=True)
                # d['relu' + str(i)] = nn.ReLU()
                d['si' + str(i)] = nn.Sigmoid()
            # elif i == layer - 2:
            #     d['linear' + str(i)] = LSH_Layer(hidden, hidden, output=t)
            #     d['relu' + str(i)] = nn.ReLU()
            else:
                d['linear' + str(i)] = torch.nn.modules.Linear(hidden, hidden, bias=True)
                d['si' + str(i)] = nn.Sigmoid()
                # d['relu' + str(i)] = nn.ReLU()

        d['linear' + str(layer)] = torch.nn.modules.Linear(hidden, 10, bias=True)
        # d['si' + str(layer)] = nn.Sigmoid()
        # d['relu' + str(layer)] = nn.ReLU()
        return d

    def weights_init(self, m):
        m.weight.data.normal_(mean=0, std=1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.modules.Linear):
                nn.init.normal_(m.weight, mean=0, std=1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    def forward(self, x):
        output = F.log_softmax(self.model(x.view(-1, 784)), dim=-1)
        return output





