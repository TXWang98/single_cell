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
from data_preprocess import load_data
from model_CFAN import Net
import matplotlib.pyplot as plt
import anndata as ad
import scanpy
import random


warnings.filterwarnings('ignore')

class copydataset(Dataset):
    def __init__(self,x,y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]





def cal_variance(vec):
    vec = np.array(vec)
    vec_2 = np.dot(vec,vec)
    l = len(vec)
    avg = sum(vec)/l
    return (vec_2/l) - (avg*avg)



feature, label, cell_type_num = load_data('./data_filtered.tsv','./subtype.ann')

cell_num = feature.shape[0]
feature_num = feature.shape[1]
input_feature_num = feature_num
print(feature.shape)

if feature_num > 1000:
    dic = {}
    for i in range(feature.shape[1]):
        
        vari = cal_variance(feature[:,i])
        dic[i] = vari
        #print(i,vari)

    lis = sorted(dic.items(),key = lambda x:x[1],reverse = True)
    chosen_gene = [ele[0] for ele in lis][0:200]
    feature = feature[:,chosen_gene]
    input_feature_num = 1000
        
        
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
print(device)        

#please modify training samples and test samples at your will
datatrain = feature[0:int(0.6 * cell_num),:]
datatest = feature[int(0.4 * cell_num):,:]
labeltrain = label[0:int(0.6 * cell_num)]
labeltest = label[int(0.4 * cell_num):]


datatrain = torch.FloatTensor(datatrain)
labeltrain = torch.LongTensor(labeltrain)

datatrain = datatrain.to(device)
labeltrain = labeltrain.to(device)

datatest = torch.FloatTensor(datatest)
labeltest = torch.LongTensor(labeltest)
datatest = datatest.to(device)
labeltest = labeltest.to(device)


dataset1 = copydataset(datatrain,labeltrain)
dataloader1 = DataLoader(dataset1,batch_size = 10,shuffle = True)

dataset2 = copydataset(datatest,labeltest)
dataloader2 = DataLoader(dataset2,batch_size = 10,shuffle = False)

    

model = Net(in_feature = input_feature_num,capsule_feature_dim = 128,compound_feature_dim = 16,out_feature = cell_type_num)
model = model.to(device)
print(model)
total_params = sum(p.numel() for p in model.parameters())
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("total_params:",total_params)
print("trainable:",total_trainable_params)
loss_func1 = nn.CrossEntropyLoss()

opt = torch.optim.Adam(model.parameters(),lr = 0.0001)

epoch = 50
train_loss_list = []
test_accuracy_list = []
for e in range(epoch):
    model.train()
    time1 = time.time()
    lossitem = 0
    for index, batch_content in enumerate(dataloader1):
        opt.zero_grad()
        yhat,latent,atten_res = model(batch_content[0])
        loss = loss_func1(yhat, batch_content[1].view(-1))
        lossitem += loss.item()
        loss.backward()
        opt.step()
        if index % 3 == 0:
            with torch.no_grad():
                yhat_cpu = yhat.to('cpu')
                pred_cpu = torch.max(F.softmax(yhat_cpu, dim=1), 1)[1].numpy()
                labels_cpu = batch_content[1].to('cpu').numpy()
                correct_cpu = np.sum(pred_cpu == labels_cpu)
                training_acc = correct_cpu / len(labels_cpu)
            print("Epoch = ",e," index = ",index," training accuracy = ",training_acc)
            time2 = time.time()

    time2 = time.time()
    print("Epoch = ", e, " training loss = ", lossitem," training time = ",time2 - time1," s")

    total_correct = 0
    total_num = 0
    with torch.no_grad():
        for ele in dataloader2:
            yhat,latent_reshape,compound_feature = model(ele[0])
            yhat_cpu = yhat.cpu()
            pred_cpu = torch.max(F.softmax(yhat_cpu, dim = 1), 1)[1].numpy()
            labels_cpu = ele[1].cpu()
            labels_cpu = labels_cpu.numpy()
            correct = np.sum(pred_cpu == labels_cpu)
            total_correct += correct.item()
            total_num += len(labels_cpu)
        print(total_correct,total_num)
        testing_acc = (total_correct * 1.0) / total_num
        print("Epoch = ", e, " testing accuracy = ", testing_acc)


model = model.to("cpu")
torch.save(model,"./CFAN.pkl")
            
print("cell_number in this dataset is ",cell_num)
print("original feature number in this dataset is", feature_num)
print("input_feature_number in this dataset is ",input_feature_num)
print("classification number (cell_type number) in this dataset is ",cell_type_num)

                            