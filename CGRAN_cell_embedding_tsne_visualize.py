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
from sklearn.manifold import TSNE
import anndata as ad
import scanpy
import random
from numpy.random import seed
from numpy import linalg as la
from sklearn.cluster import KMeans
from sklearn import metrics
from data_preprocess import load_data
from model_CGRAN import Classifier,MultiheadClassifier,MultilayerClassifier


#seed(30)
#torch.manual_seed(30)
#torch.cuda.manual_seed(30)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
print(device)

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



feature, label, cell_type_num = load_data('./data_filtered.tsv','./subtype.ann')

celltypename = []
for i in range(len(label)):
    if label[i] == 0:
        celltypename.append("Monocytes")
    elif label[i] == 1:
        celltypename.append("NK cells")
    elif label[i] == 2:
        celltypename.append("B cells")
    elif label[i] == 3:
        celltypename.append("T cells")
    elif label[i] == 4:
        celltypename.append("Megakaryocytes")

celltypename = np.array(celltypename)

cell_num = feature.shape[0]
feature_num = feature.shape[1]
input_feature_num = feature_num
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

x_all = input_data
y_all = label

data_all = torch.FloatTensor(x_all)
label_all = torch.LongTensor(y_all)

data_all = data_all.to(device)
label_all = label_all.to(device)

dataset_all = copydataset0(data_all, label_all)
dataloader_all = DataLoader(dataset_all, batch_size = 10, shuffle=False)

model_C = torch.load("./SVD_MF_CGRAN_PBMC.pkl")
print(model_C)
model_C = model_C.to(device)
model_C.eval()
batch_cnt = 0
with torch.no_grad():
    for ele in dataloader_all:
        if batch_cnt == 0:
            batch_cnt += 1
            yhat_all, cell_representation_all = model_C(ele[0])
            yhat_cpu_all = yhat_all.cpu()
            cell_representation_all = cell_representation_all.cpu().data.numpy()
            pred_cpu_all = torch.max(F.softmax(yhat_cpu_all, dim=1), 1)[1].numpy()
            labels_cpu_all = ele[1].cpu()
            labels_cpu_all = labels_cpu_all.numpy()

        else:
            batch_cnt += 1
            yhat, cell_representation = model_C(ele[0])
            yhat_cpu = yhat.cpu()
            cell_representation = cell_representation.cpu().data.numpy()
            pred_cpu = torch.max(F.softmax(yhat_cpu, dim=1), 1)[1].numpy()
            labels_cpu = ele[1].cpu()

            pred_cpu_all = np.concatenate((pred_cpu_all, pred_cpu), axis=0)
            cell_representation_all = np.concatenate((cell_representation_all, cell_representation), axis=0)
            labels_cpu_all = np.concatenate((labels_cpu_all, labels_cpu), axis=0)

    correct = np.sum(pred_cpu_all == labels_cpu_all)
    total_correct = correct.item()
    total_num = len(labels_cpu_all)
    print(total_correct, total_num)
    testing_acc = (total_correct * 1.0) / total_num
    print(" testing accuracy = ", testing_acc)



tsne = TSNE(n_components = 2, init = 'pca', random_state = 30)
cell_representation_2d = tsne.fit_transform(cell_representation_all)

celltypename = np.array(celltypename)

fig, ax = plt.subplots(figsize=(10, 10))

ax.tick_params(labelsize=15)
ax.legend(loc='upper left', prop={'size': 14}, fontsize='large')
plt.rcParams.update({'font.size': 15.5})

labels_cpu_all_name = []

for i in range(len(labels_cpu_all)):
    if labels_cpu_all[i] == 0:
        labels_cpu_all_name.append("Monocytes")
    elif labels_cpu_all[i] == 1:
        labels_cpu_all_name.append("NK cells")
    elif labels_cpu_all[i] == 2:
        labels_cpu_all_name.append("B cells")
    elif labels_cpu_all[i] == 3:
        labels_cpu_all_name.append("T cells")
    elif labels_cpu_all[i] == 4:
        labels_cpu_all_name.append("Megakaryocytes")

labels_cpu_all_name = np.array(labels_cpu_all_name)

for ctn in np.unique(celltypename):
    ax.scatter(cell_representation_2d[labels_cpu_all_name == ctn, 0],
               cell_representation_2d[labels_cpu_all_name == ctn, 1], label=ctn, cmap='plasma', s=8)

ax.legend(frameon = True)
plt.savefig("./SVD_MF_PBMC_tsne_visualize.pdf")



