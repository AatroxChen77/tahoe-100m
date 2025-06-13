import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def get_data(file_path, test_size=0.1, val_size=0.1, random_state=42):
    """
    Loads, preprocesses, and splits the data into train, validation, and test sets.
    """
    adata = sc.read_h5ad(file_path)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True, flavor="seurat")
    
    # First split into train+val and test
    train_val_idx, test_idx = train_test_split(
        adata.obs.index, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Then split train+val into train and val
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size/(1-test_size),
        random_state=random_state
    )
    
    adata_train = adata[train_idx].copy()
    adata_val = adata[val_idx].copy()
    adata_test = adata[test_idx].copy()
    
    return adata_train, adata_val, adata_test

class AdataCVAEWrapper(Dataset):
    def __init__(self, adata, cat_features, cont_features):
        self.X = adata.X
        self.cat_data = pd.get_dummies(adata.obs[cat_features], drop_first=False).values.astype(np.float32)
        self.cat_data = torch.from_numpy(self.cat_data)
        
        cont = adata.obs[cont_features].values.astype(np.float32)
        cont = (cont - cont.mean(axis=0)) / cont.std(axis=0)
        self.cont_data = torch.from_numpy(cont)
        
        self.cond = torch.cat([self.cat_data, self.cont_data], dim=1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x_row = torch.tensor(self.X[idx].toarray().squeeze(), dtype=torch.float32)
        c = self.cond[idx]
        return x_row, c 