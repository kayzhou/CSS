import pickle
import numpy as np
import json
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from tqdm import tqdm


class GCNModel(nn.Module):
    def __init__(self, data, num_features, num_nodes, hidden, support, num_bases, num_classes, 
        text_model='HLSTM', bias_feature=False):
        super(GCNModel, self).__init__()

        self.gc1 = GraphConvolution(num_nodes, hidden, support, num_bases=num_bases,
            activation='tanh')
        self.gc2 = GraphConvolution(hidden, num_features, support, num_bases=num_bases,
                    activation='tanh')
        self.clf_bias = nn.Linear(num_features, num_classes)
       


# This implementation is based on the code at https://github.com/tkipf/relational-gcn/blob/master/rgcn/layers/graph.py
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, support=1, 
                 activation='linear', num_bases=-1, bias=False):
        
        super(GraphConvolution, self).__init__()

        if (activation == 'linear'):
            self.activation = None
        elif (activation == 'sigmoid'):
            self.activation = nn.Sigmoid()
        elif (activation == 'tanh'):
            self.activation = nn.Tanh()
        else:
            print('Error: activation function not available')
            exit()

        self.input_dim = input_dim
        self.output_dim = output_dim  # number of features per node
        self.support = support  # filter support / number of weights

        assert support >= 1

        self.bias = bias
        self.num_bases = num_bases

        if self.num_bases > 0:
            self.W = [Parameter(nn.init.xavier_uniform_(
                torch.FloatTensor(self.input_dim, self.output_dim)))
                for i in range(self.num_bases)]
            self.W_comp = Parameter(nn.init.xavier_uniform_(
                torch.FloatTensor(self.support, self.num_bases)))
        else:
            self.W = [Parameter(nn.init.xavier_uniform_(
                torch.FloatTensor(self.input_dim, self.output_dim)))
                for i in range(self.support)]
        for idx, item in enumerate(self.W):
            self.register_parameter('W_%d' % idx, item)

        if self.bias:
            self.b = Parameter(torch.FloatTensor(self.output_dim, 1))
        
    def forward(self, inputs, mask=None):
        features = inputs[0]
        A = inputs[1:]  # list of basis functions

        # convolve
        supports = []
        if self.num_bases > 0:
            V = []
            for i in range(self.support):
                basis_weights = []
                for j, a_weight in enumerate(self.W):
                    basis_weights.append(self.W_comp[i][j] * a_weight)
                V.append(torch.stack(basis_weights, dim=0).sum(0))
            for i in range(self.support):
                # print(V[i].size())
                supports.append(torch.spmm(features, V[i]))
        else:
            for a_weight in self.W:
                supports.append(torch.spmm(features, a_weight))

        outputs = []
        for i in range(self.support):
            # print(features.size(), A[i].size())
            outputs.append(torch.spmm(A[i], supports[i]))

        output = torch.stack(outputs, dim=1).sum(1)            

        if self.bias:
            output += self.b
        
        if self.activation is not None:
            return self.activation(output)
        else:
            return output
