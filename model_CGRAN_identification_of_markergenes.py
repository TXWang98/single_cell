import torch
import torch.nn as nn
import warnings
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import scipy.sparse as sp
import time
import argparse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import anndata as ad
import scanpy
import random
from numpy.random import seed
from numpy import linalg as la

#seed(30)
#torch.manual_seed(30)
#torch.cuda.manual_seed(30)

warnings.filterwarnings('ignore')

class Classifier(nn.Module):
    def __init__(self, cell_num, in_feature_num, in_embedding_dim, out_embedding_dim, current_layer, current_head):
        super(Classifier, self).__init__()
        self.cell_num = cell_num
        self.in_feature_num = in_feature_num
        self.current_layer = current_layer
        self.in_embedding_dim = in_embedding_dim
        self.current_head = current_head
        self.local_atten_mat = local_atten_mat
        self.out_embedding_dim = out_embedding_dim
        self.activation_func = nn.ReLU()
        self.current_layer = current_layer
        # self.QKV_para_matrix = nn.Parameter(torch.randn(self.in_embedding_dim, self.out_embedding_dim * 3))
        self.Q = nn.Linear(self.in_embedding_dim, self.out_embedding_dim)
        self.K = nn.Linear(self.in_embedding_dim, self.out_embedding_dim)
        self.V = nn.Linear(self.in_embedding_dim, self.out_embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        # nn.init.xavier_normal_(self.c.weight, gain=gain)

    def forward(self, x):
        batch_size = x.shape[0]

        Q = self.Q(self.dropout(x))
        K = self.K(self.dropout(x))
        V = self.V(self.dropout(x))

        K_transpose = K.permute(0, 2, 1)
        atten_weight = torch.matmul(Q, K_transpose)
        atten_weight = torch.div(atten_weight, np.sqrt(self.in_embedding_dim))

        atten_weight_softmax = nn.Softmax(dim=2)(atten_weight)
        atten_weight_local = torch.mul(atten_weight_softmax, self.local_atten_mat)
        atten_weight_final = F.normalize(atten_weight_local, p=2, dim=2)

        compound_feature = torch.matmul(atten_weight_final, V)

        return compound_feature, atten_weight_final


class MultiheadClassifier(nn.Module):
    def __init__(self, cell_num, in_feature_num, in_embedding_dim, out_embedding_dim, current_layer, head_num):
        super(MultiheadClassifier, self).__init__()
        self.cell_num = cell_num
        self.in_feature_num = in_feature_num
        self.in_embedding_dim = in_embedding_dim
        self.head_num = head_num
        self.out_embedding_dim = out_embedding_dim
        self.current_layer = current_layer
        self.attention_head_list = nn.ModuleList([Classifier(self.cell_num, self.in_feature_num, self.in_embedding_dim,
                                                             self.out_embedding_dim, current_layer, current_head=i) for
                                                  i in range(self.head_num)])

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        # nn.init.xavier_normal_(self.c.weight, gain=gain)

    def forward(self, x):
        batch_size = x.shape[0]
        if self.current_layer == 0:

            multi_head_att = [att(x)[0] for att in self.attention_head_list]
            multi_head_att_concat = torch.cat(multi_head_att, dim=2)

            multi_head_atten_weights = [att(x)[1] for att in
                                        self.attention_head_list]
            multi_head_atten_weights = torch.cat(multi_head_atten_weights, dim=0)
            multi_head_atten_weights = multi_head_atten_weights.reshape(10, batch_size, 1001, 1001)
            multi_head_atten_weights = torch.sum(multi_head_atten_weights, dim=0)

            return multi_head_att_concat, multi_head_atten_weights

        elif self.current_layer == 1:

            multi_head_att = [att(x)[0] for att in self.attention_head_list]
            multi_head_att_concat = torch.cat(multi_head_att, dim=2)

            return multi_head_att_concat


class MultilayerClassifier(nn.Module):
    def __init__(self, cell_num, in_feature_num, embedding_dim_list, layer_num, cell_type_num):
        super(MultilayerClassifier, self).__init__()
        self.cell_num = cell_num
        self.in_feature_num = in_feature_num
        self.embedding_dim_list = embedding_dim_list
        self.layer_num = layer_num
        self.cell_type_num = cell_type_num
        self.attentionlayer1 = MultiheadClassifier(self.cell_num, self.in_feature_num, embedding_dim_list[0],
                                                   embedding_dim_list[1], current_layer=0, head_num=10)
        self.residual1 = nn.Linear(embedding_dim_list[0], 10 * embedding_dim_list[1])
        self.attentionlayer2 = MultiheadClassifier(self.cell_num, self.in_feature_num, 10 * embedding_dim_list[1],
                                                   embedding_dim_list[2], current_layer=1, head_num=10)
        self.residual2 = nn.Linear(10 * embedding_dim_list[1], 10 * embedding_dim_list[2])


        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=1)
        self.c = nn.Sequential(nn.Linear(10 * self.embedding_dim_list[2], self.cell_type_num))

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.c.weight, gain=gain)

    def forward(self, x):
        batch_size = x.shape[0]

        x_att, multi_head_atten_weights = self.attentionlayer1(x)
        x_res = self.residual1(self.dropout(x))
        x = torch.add(x_att, x_res)
        x = F.normalize(x, p=2, dim=2)


        x_att = self.attentionlayer2(x)
        x_res = self.residual2(self.dropout(x))
        x = torch.add(x_att, x_res)
        x = F.normalize(x, p=2, dim=2)


        cell_embedding_final = x[:, 0, :].reshape(batch_size, -1)
        res = self.c(self.dropout(cell_embedding_final))

        return res, multi_head_atten_weights

local_atten_mat = np.zeros((1001,1001))

local_atten_mat[0][0] = 1
for j in range(1,1001):
    local_atten_mat[0][j] = 1
    local_atten_mat[j][0] = 1

for k in range(10):
    for l in range(100):
        local_atten_mat[100*k+1+l][100*k+1:100*(k+1)+1] = 1