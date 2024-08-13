import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import torchvision.transforms as transforms
import scipy.sparse as sp
from torch.nn import init
import time
import argparse
import os
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import anndata as ad
import scanpy
import random




class Net(nn.Module):
    def __init__(self ,in_feature ,capsule_feature_dim ,compound_feature_dim ,out_feature):
        super(Net ,self).__init__()
        self.capsule_feature_dim = capsule_feature_dim
        self.compound_feature_dim = compound_feature_dim
        self.activation_func = nn.ReLU()
        self.output_feature = out_feature
        self.capsule_list = nn.ModuleList()
        self.conv_out_channel = 8
        self.conv_kernel_size = 5
        self.c = nn.Linear(self.conv_out_channel *(16 - self.conv_kernel_size + 1) ,self.output_feature)
        self.conv = nn.Conv1d(in_channels = self.compound_feature_dim ,out_channels = self.conv_out_channel
                              ,kernel_size = self.conv_kernel_size)
        for i in range(16):
            self.capsule_list.append(nn.Sequential(nn.Linear(in_feature, capsule_feature_dim) ,self.activation_func))

        self.QKV_para_matrix = nn.Parameter(torch.randn(self.capsule_feature_dim ,self.compound_feature_dim * 3))
        self.dropout = nn.Dropout(0.2)

    def forward(self ,x):
        batch_size = x.shape[0]
        latent = self.capsule_list[0](self.dropout(x)).reshape(batch_size ,self.capsule_feature_dim)
        for num in range(1 ,16):
            latent = torch.cat \
                ((latent ,self.capsule_list[num](self.dropout(x)).reshape(batch_size ,self.capsule_feature_dim)) ,1)
        latent_reshape = latent.reshape(batch_size ,16 ,self.capsule_feature_dim)
        latent_reshape = F.normalize(latent_reshape ,p = 2 ,dim = 2)

        QKV = torch.matmul(latent_reshape ,self.QKV_para_matrix)
        Q = QKV[: ,: ,0:self.compound_feature_dim]
        K = QKV[: ,: ,self.compound_feature_dim: 2 *self.compound_feature_dim]
        V = QKV[: ,: , 2 *self.compound_feature_dim: 3 *self.compound_feature_dim]
        K_transpose = K.permute(0 ,2 ,1)
        atten_weight = torch.matmul(Q ,K_transpose)
        atten_weight_softmax = nn.Softmax(dim = 2)(atten_weight)

        compound_feature = torch.matmul(atten_weight_softmax ,V)
        compound_feature = F.normalize(compound_feature ,p = 2 ,dim = 2)
        compound_feature_permute = compound_feature.permute(0 ,2 ,1)

        conv_res = self.conv(compound_feature_permute)
        conv_res = conv_res.reshape(batch_size ,-1)

        res = self.c(self.dropout(conv_res))

        return res ,latent_reshape ,compound_feature