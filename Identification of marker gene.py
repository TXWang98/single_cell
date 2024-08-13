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



feature, label, cell_type_num = load_data('./data_filtered.tsv','./subtype.ann')


cell_num = feature.shape[0]
feature_num = feature.shape[1]
most_var_gene_num = feature_num
print(feature.shape)

if feature_num > 1000:
    dic = {}
    for i in range(feature.shape[1]):
        vari = cal_variance(feature[:, i])
        dic[i] = vari
        # print(i,vari)
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

data_vs = input_data
y_vs = label


data_vs = torch.FloatTensor(data_vs).to(device)
y_vs = torch.LongTensor(y_vs).to(device)
dataset_vs = copydataset0(data_vs, y_vs)
dataloader_vs = DataLoader(dataset_vs, batch_size = 50, shuffle = False)

# model_C = MultilayerClassifier(cell_num = cell_num,in_feature_num = most_var_gene_num + 1,embedding_dim_list = [gene_embedding_dim,8,4],layer_num = #2,cell_type_num = cell_type_num)


model_C = torch.load("./SVD_MF_CGRAN_model_identification_markers.pkl")
print(model_C)
model_C = model_C.to(device)
model_C.eval()

batch_cnt = 0
with torch.no_grad():
    for ele in dataloader_vs:
        # print(batch_cnt)
        if batch_cnt == 0:
            yhat_all, attention_weights_all = model_C(ele[0])
            yhat_cpu_all = yhat_all.to('cpu')
            pred_cpu_all = torch.max(F.softmax(yhat_cpu_all, dim=1), 1)[1].numpy()
            # print(pred_cpu_all)
            attention_weights_all = attention_weights_all.to('cpu')
            attention_weights_all = attention_weights_all.data.numpy()[:, 0:1, :]
            batch_cnt += 1

        else:
            yhat, attention_weights = model_C(ele[0])
            yhat_cpu = yhat.to('cpu')
            pred_cpu = torch.max(F.softmax(yhat_cpu, dim=1), 1)[1].numpy()
            # print(pred_cpu)
            attention_weights = attention_weights.to('cpu')
            attention_weights = attention_weights.data.numpy()[:, 0:1, :]

            pred_cpu_all = np.concatenate((pred_cpu_all, pred_cpu), axis=0)
            attention_weights_all = np.concatenate((attention_weights_all, attention_weights), axis=0)
            batch_cnt += 1

print(len(pred_cpu_all))
print(attention_weights_all.shape)

#cell_type_number * 1000
pred_cnt_num = [0, 0, 0, 0, 0]
real_cnt_num = [0, 0, 0, 0, 0]
marker_gene = np.zeros((5, 1000))
correct_num_vs = 0

for i in range(cell_num):
    single_att_weights = attention_weights_all[i]
    cell_gene_weights = single_att_weights[0:1, 1:1001]
    dic = {}

    pred_cnt_num[int(pred_cpu_all[i])] += 1
    real_cnt_num[int(y_vs[i])] += 1

    if y_vs[i] == pred_cpu_all[i]:
        correct_num_vs += 1
        for j in range(0, 1000):
            dic[j] = cell_gene_weights[0][j]

        lis = sorted(dic.items(), key=lambda x: x[1], reverse=True)
        chosen_gene = [ele[0] for ele in lis][0:50]

        for k in range(50):
            marker_gene[int(y_vs[i])][int(chosen_gene[k])] += 1

print(pred_cnt_num)
print(real_cnt_num)
print(correct_num_vs / cell_num)
for i in range(5):
    print(list(marker_gene[i]))

print("correct_num_vs:", correct_num_vs)
