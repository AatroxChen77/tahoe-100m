import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from datasets import load_dataset
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def create_anndata_from_generator(generator, gene_vocab, sample_size=None):
    """
    Creates an AnnData object from a generator of Tahoe-100M dataset records.
    """
    sorted_vocab_items = sorted(gene_vocab.items())
    token_ids, gene_names = zip(*sorted_vocab_items)
    token_id_to_col_idx = {token_id: idx for idx, token_id in enumerate(token_ids)}

    data, indices, indptr = [], [], [0]
    obs_data = []

    for i, cell in enumerate(generator):
        if sample_size is not None and i >= sample_size:
            break
        genes = cell['genes']
        expressions = cell['expressions']
        if expressions and expressions[0] < 0:
            genes = genes[1:]
            expressions = expressions[1:]

        col_indices = [token_id_to_col_idx[gene] for gene in genes if gene in token_id_to_col_idx]
        valid_expressions = [expr for gene, expr in zip(genes, expressions) if gene in token_id_to_col_idx]

        data.extend(valid_expressions)
        indices.extend(col_indices)
        indptr.append(len(data))

        obs_entry = {k: v for k, v in cell.items() if k not in ['genes', 'expressions']}
        obs_data.append(obs_entry)

    expr_matrix = csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, len(gene_names)))
    obs_df = pd.DataFrame(obs_data)

    adata = anndata.AnnData(X=expr_matrix, obs=obs_df)
    adata.var.index = pd.Index(gene_names, name='ensembl_id')

    return adata

def get_data(file_path, test_size=0.1, random_state=42):
    """
    Loads, preprocesses, and splits the data.
    """
    adata_100m = sc.read_h5ad(file_path)
    sc.pp.normalize_total(adata_100m, target_sum=1e4)
    sc.pp.log1p(adata_100m)
    sc.pp.highly_variable_genes(adata_100m, n_top_genes=2000, subset=True, flavor="seurat")
    
    train_idx, test_idx = train_test_split(adata_100m.obs.index, test_size=test_size, random_state=random_state)
    adata_train = adata_100m[train_idx].copy()
    adata_test = adata_100m[test_idx].copy()
    
    return adata_train, adata_test

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