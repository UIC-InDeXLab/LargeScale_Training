import time

import torch
from torch.nn.modules import Linear
import numpy as np
import sys
import torch.nn as nn
from numpy import linalg as LA
# sys.path.append('..')
from function import approx_linear_func
from hashfunction import HashFunction
from lsh_hashbucket import LSH
from numpy import linalg as la
#K=5
#L=6
#m=3

class LSH_Layer(torch.nn.modules.Linear):

    def __init__(self, in_features, out_features,t, bias=False, L=6, K=5, m=3):
        self.type = t
        self.lsh = None
        self.K_ = K
        self.L_ = L
        self.f = 0
        self.actives = np.zeros((1,out_features))
        self.active_num = np.zeros((1,out_features))
        self.inactives = np.zeros((1,out_features))
        #self.backward = 0
        if self.type != 1:
            self.lsh = LSH(HashFunction(L, K, in_features, m), out_features, K, L, m)
        super(LSH_Layer, self).__init__(
            in_features,out_features,bias)

    def build_lsh_tables(self):
        weights = self.weight.detach().numpy()
        rows = weights.shape[0]
        start = time.time()
        weights = self.normalize_weights(weights)
        self.weight.data = torch.Tensor(weights)
        for neuron in range(rows):
            item_id = neuron
            self.lsh.insert(item_id, weights[item_id])
        t = time.time()-start
        return t

    @staticmethod
    def normalize_weights(weights):

        weight_norms = LA.norm(weights, axis=1)
        max_norm = max(weight_norms)
        normalizer = 1
        if max_norm > 0.83:
            normalizer = max_norm/0.83
        normalized_weights = weights/normalizer
        return normalized_weights
        

    def forward(self, inputs):
        if self.training is True:
            # Use approximation in training only.
            start = time.time()
            output = approx_linear_func.apply(inputs, self.weight, self.bias, self.lsh,self.actives, self.inactives)
            return output

        else:
            # For evaluation, perform the exact computation
            output = torch.nn.functional.linear(inputs, self.weight, self.bias)
            return output
