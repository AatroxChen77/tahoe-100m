from datasets import load_dataset
from scipy.sparse import csr_matrix
from tqdm import tqdm
import anndata
import pandas as pd
import pubchempy as pcp
import numpy as np
import ast

def create_anndata_from_generator(generator, gene_vocab, sample_size=None):
    sorted_vocab_items = sorted(gene_vocab.items())
    token_ids, gene_names = zip(*sorted_vocab_items)
    token_id_to_col_idx = {token_id: idx for idx, token_id in enumerate(token_ids)}

    data, indices, indptr = [], [], [0]
    obs_data = []

    for i, cell in tqdm(enumerate(generator), total = sample_size):
        if sample_size is not None and i >= sample_size:
            break
        genes = cell['genes']
        expressions = cell['expressions']
        if expressions[0] < 0:
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

if __name__ == "__main__":
    # setup dataset
    tahoe_100m_ds = load_dataset('vevotx/Tahoe-100M', streaming=True, split='train')
    gene_metadata = load_dataset("vevotx/Tahoe-100M", name="gene_metadata", split="train")
    gene_vocab = {entry["token_id"]: entry["ensembl_id"] for entry in gene_metadata}
    adata = create_anndata_from_generator(tahoe_100m_ds, gene_vocab, sample_size=1_000_000) # 1_000_000
    sample_metadata = load_dataset("vevotx/Tahoe-100M","sample_metadata", split="train").to_pandas()
    adata.obs = pd.merge(adata.obs, sample_metadata.drop(columns=["drug","plate"]), on="sample")
    adata.obs["drugname_drugconc"] = adata.obs["drugname_drugconc"].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    vals = adata.obs["drugname_drugconc"].values
    drug_array = np.array([x[0] if isinstance(x, list) and len(x) > 0 else (None, None, None) for x in vals])
    # add drug array values
    adata.obs["drug_name"] = drug_array[:, 0]
    adata.obs["drug_conc"] = drug_array[:, 1]
    adata.obs["drug_unit"] = drug_array[:, 2]

    # get drug metadata
    drug_metadata = load_dataset("vevotx/Tahoe-100M","drug_metadata", split="train").to_pandas()

    adata.obs = pd.merge(adata.obs, drug_metadata.drop(columns=["canonical_smiles","pubchem_cid","moa-fine"]), on="drug", how='left')

    drug_name = adata.obs["drug"].values[0]
    cid = int(float(adata.obs["pubchem_cid"].values[0]))
    compound = pcp.Compound.from_cid(cid)

    print(f"Name: {drug_name}")
    print(f"Synonyms: {compound.synonyms[:10]}")
    print(f"Formula: {compound.molecular_formula}")
    print(f"SMILES: {compound.isomeric_smiles}")
    print(f"Mass: {compound.exact_mass}")

    cell_line_metadata = load_dataset("vevotx/Tahoe-100M","cell_line_metadata", split="train").to_pandas()

    # save cell line to csv
    cell_line_metadata.to_csv("/root/autodl-tmp/data/tahoe-100m_5M_cell_line_metadata.csv", index=False)

    # convert targets to string
    adata.obs["targets"] = adata.obs["targets"].astype(str)

    # rename drug_conc as string
    adata.obs["drugname_drugconc"] = adata.obs["drugname_drugconc"].astype(str)
    # save down-sampled adata file
    adata.write_h5ad("/root/autodl-tmp/data/tahoe-100m_5M.h5ad")
