Here we present the implementations of  CFAN and CGRAN.

Both CFAN and CGRAN needs data preprocess first. The input are the data and labels we supply. You may select training set/validation set/test set at your will. model_CFAN.py, model_CGRAN.py and model_CGRAN_identification_of_markergenes.py are CFAN and CGRAN models.

Combining train_evaluate_CFAN.py with model_CFAN.py, users can train a CFAN model and evaluate its performance.

Combining train_evaluate SVD_MF_CGRAN_identification.py, train_evaluate_NN_MF_CGRAN.py with model_CGRAN.py, users can train CGRAN model and evaluate its performance.
CGRAN_cell_embedding_tsne_visualize.py is for the visualizations of cell embeddings trained in CGRAN.

Identification of marker gene.py is provided for the discovery of marker genes for different cell types in different datasets.

For transfer learning part, users can train one CGRAN/CFAN model on one dataset and transfer the parameters and finetune them on another dataset.