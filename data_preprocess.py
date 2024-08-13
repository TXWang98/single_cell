import scanpy
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp



def load_data(data_root = './data_filtered.tsv',cell_type_root = './subtype.ann'):
    data = pd.read_csv(data_root,index_col = 0, sep = '\t')
    cell_type = pd.read_csv(cell_type_root,sep = '\t')

    cells = data.columns.values
    genes = data.index.values
    feature = data.values.T

    cell_num = feature.shape[0]
    feature_num = feature.shape[1]
    print("cell number = ", cell_num, "feature_num = ", feature_num)

    # validation1
    #if cell_num != cell_type.shape[0]:
    #    print("cell number in data and cell_type are not the same")
    #    print("Wrong!!")
    # validation2
    #for i in range(len(cells)):
    #    if str(cells[i]) != str(cell_type.iloc[i, 0]):
    #        print(i, "data cell_type does not correspond with cell_type file's cell type")

    rowsum = feature.sum(1)
    r_inv = np.power(rowsum.astype(float), -1)
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(feature)
    S_ = np.power(10, 5)
    features = features * S_
    feature = np.log2(features + 1)

    label = []
    for i in range(cell_num):
        label.append(cell_type.iloc[i,1])

    flag = []
    cell_type_num = 0
    for ele in label:
        if ele not in flag:
            flag.append(ele)
            cell_type_num += 1
    print(cell_type_num)

    feature = np.array(feature).astype(np.float32)
    label = np.array(label)

    return feature, label, cell_type_num