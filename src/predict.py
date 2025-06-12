import torch
from data_utils import get_data, AdataCVAEWrapper
from model import CVAE
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, test_loader, device):
    """
    Evaluates the CVAE model on the test set.
    """
    model.eval()
    all_recon = []
    all_original = []
    
    with torch.no_grad():
        for x_batch, c_batch in test_loader:
            x_batch = x_batch.to(device)
            c_batch = c_batch.to(device)
            recon_x, _, _ = model(x_batch, c_batch)
            all_recon.append(recon_x.cpu().numpy())
            all_original.append(x_batch.cpu().numpy())
            
    all_recon = np.concatenate(all_recon, axis=0)
    all_original = np.concatenate(all_original, axis=0)
    
    return all_original, all_recon


if __name__ == '__main__':
    # You will need to change this to the path of your trained model
    model_path = 'models/best_model.pth' 
    data_path = 'data/tahoe-100m_5M.h5ad'
    
    adata_train, adata_test = get_data(data_path)
    cat_features = ["drug", "cell_line_id"]
    cont_features = ["drug_conc"]
    
    test_dataset = AdataCVAEWrapper(adata_test, cat_features, cont_features)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    model = CVAE(input_dim=adata_test.shape[1], cond_dim=test_dataset.cond.shape[1])
    model.load_state_dict(torch.load(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    original, reconstructed = evaluate_model(model, test_loader, device)
    
    mse_per_gene = mean_squared_error(original, reconstructed, multioutput='raw_values')
    r2_per_gene = r2_score(original, reconstructed, multioutput='raw_values')
    
    print(f"Average MSE per gene: {np.mean(mse_per_gene):.4f}")
    print(f"Average R2 per gene: {np.mean(r2_per_gene):.4f}")