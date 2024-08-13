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
from data_preprocess import load_data
from model_CGRAN_identification_of_markergenes import Classifier, MultiheadClassifier, MultilayerClassifier

# seed(30)
# torch.manual_seed(30)
# torch.cuda.manual_seed(30)

warnings.filterwarnings('ignore')


class copydataset0(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class copydataset1(Dataset):
    def __init__(self, x, cellindex):
        self.x = x
        self.cellindex = cellindex

    def __len__(self):
        return len(self.cellindex)

    def __getitem__(self, idx):
        return self.x[idx], self.cellindex[idx]


class copydataset2(Dataset):
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w

    def __len__(self):
        return len(self.z)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx], self.w[idx]


def cal_variance(vec):
    vec = np.array(vec)
    vec_2 = np.dot(vec, vec)
    l = len(vec)
    avg = sum(vec) / l
    return (vec_2 / l) - (avg * avg)


if torch.cuda.is_available():
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
print(device)

feature, label, cell_type_num = load_data('./data_filtered.tsv', './subtype.ann')

cell_num = feature.shape[0]
feature_num = feature.shape[1]
most_var_gene_num = feature_num
print(feature.shape)

if feature_num > 1000:
    dic = {}
    for i in range(feature.shape[1]):
        vari = cal_variance(feature[:, i])
        dic[i] = vari
        print(i, vari)
    lis = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    chosen_gene = [ele[0] for ele in lis][0:1000]
    feature = feature[:, chosen_gene]
    most_var_gene_num = 1000

gene_embedding_dim = 128

print(feature.shape)
cell_index_list = np.array([i for i in range(cell_num)])

U, sigma, V_transpose = la.svd(feature)

for i in range(U.shape[0]):
    for j in range(gene_embedding_dim):
        U[i][j] = U[i][j] * np.sqrt(sigma[j])

for i in range(V_transpose.shape[1]):
    for j in range(gene_embedding_dim):
        V_transpose[j][i] = V_transpose[j][i] * np.sqrt(sigma[j])

cell_embedding_matrix = U[:, 0:gene_embedding_dim]
gene_embedding_matrix = V_transpose.T[:, 0:gene_embedding_dim]

input_data = np.zeros((cell_num, most_var_gene_num + 1, gene_embedding_dim))
for i in range(cell_num):
    for j in range(most_var_gene_num):
        if j == 0:
            input_data[i][j] = cell_embedding_matrix[i]
        else:
            input_data[i][j] = cell_embedding_matrix[i] + gene_embedding_matrix[j - 1]

local_atten_mat = np.zeros((1001, 1001))

local_atten_mat[0][0] = 1
for j in range(1, 1001):
    local_atten_mat[0][j] = 1
    local_atten_mat[j][0] = 1

for k in range(10):
    for l in range(100):
        local_atten_mat[100 * k + 1 + l][100 * k + 1:100 * (k + 1) + 1] = 1

local_atten_mat = torch.FloatTensor(local_atten_mat)
local_atten_mat = local_atten_mat.to(device)

#please select training set and test set
datatrain_C = input_data[0:int(0.6 * cell_num), :, :]
datatest_C = input_data[int(0.4 * cell_num):, :, :]
labeltrain_C = label[0:int(0.6 * cell_num)]
labeltest_C = label[int(0.4 * cell_num):]

datatrain_C = torch.FloatTensor(datatrain_C)
datatest_C = torch.FloatTensor(datatest_C)
datatrain_C = datatrain_C.to(device)
datatest_C = datatest_C.to(device)

labeltrain_C = torch.LongTensor(labeltrain_C)
labeltest_C = torch.LongTensor(labeltest_C)
labeltrain_C = labeltrain_C.to(device)
labeltest_C = labeltest_C.to(device)

datasettrain_C = copydataset0(datatrain_C, labeltrain_C)
dataloadertrain_C = DataLoader(datasettrain_C, batch_size=10, shuffle=True)

datasettest_C = copydataset0(datatest_C, labeltest_C)
dataloadertest_C = DataLoader(datasettest_C, batch_size=10, shuffle=False)

model_C = MultilayerClassifier(cell_num=cell_num, in_feature_num=most_var_gene_num + 1,
                               embedding_dim_list=[gene_embedding_dim, 8, 4], layer_num = 2, cell_type_num=cell_type_num)

print(model_C)
model_C = model_C.to(device)
total_params = sum(p.numel() for p in model_C.parameters())
total_trainable_params = sum(p.numel() for p in model_C.parameters() if p.requires_grad)
print("total_params:", total_params)
print("trainable:", total_trainable_params)
opt = torch.optim.Adam(model_C.parameters(), lr=0.0001)
epoch_C = 75




for e in range(epoch_C):
    model_C.train()
    time1 = time.time()
    lossitem = 0
    for index, batch_content in enumerate(dataloadertrain_C):
        opt.zero_grad()
        yhat, multi_head_atten_weights = model_C(batch_content[0])
        loss = nn.CrossEntropyLoss()(yhat, batch_content[1].view(-1))
        lossitem += loss.item()
        loss.backward()
        opt.step()
        if index % 3 == 0:
            with torch.no_grad():
                yhat_cpu = yhat.cpu()
                pred_cpu = torch.max(F.softmax(yhat_cpu, dim = 1), 1)[1].numpy()
                labels_cpu = batch_content[1].cpu()
                labels_cpu = labels_cpu.numpy()
                correct = np.sum(pred_cpu == labels_cpu)
                training_acc = correct / len(labels_cpu)
                print("Epoch = ", e, " index = ", index, " training accuracy = ", training_acc)
    time2 = time.time()
    print("Epoch = ", e, " training loss = ", lossitem, " training time = ", time2 - time1, " s")


model_C = model_C.to('cpu')
torch.save(model_C, "./SVD_MF_CGRAN_model_identification_markers.pkl")




