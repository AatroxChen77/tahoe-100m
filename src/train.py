import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_utils import get_data, AdataCVAEWrapper
from model import CVAE, loss_function

def train_model(config):
    """
    Trains the CVAE model.
    """
    adata_train, adata_val, adata_test = get_data(config['data_path'])
    
    cat_features = ["drug", "cell_line_id"]
    cont_features = ["drug_conc"]

    train_dataset = AdataCVAEWrapper(adata_train, cat_features, cont_features)
    val_dataset = AdataCVAEWrapper(adata_val, cat_features, cont_features)
    test_dataset = AdataCVAEWrapper(adata_test, cat_features, cont_features)
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    model = CVAE(input_dim=adata_train.shape[1], cond_dim=train_dataset.cond.shape[1], latent_dim=config['latent_dim'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(config['n_epochs']):
        model.train()
        train_loss = 0
        for x_batch, c_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["n_epochs"]}'):
            x_batch = x_batch.to(device)
            c_batch = c_batch.to(device)
            recon_x, mu, logvar = model(x_batch, c_batch)
            loss = loss_function(recon_x, x_batch, mu, logvar) / x_batch.size(0)  # Normalize by batch size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, c_batch in val_loader:
                x_batch = x_batch.to(device)
                c_batch = c_batch.to(device)
                recon_x, mu, logvar = model(x_batch, c_batch)
                loss = loss_function(recon_x, x_batch, mu, logvar) / x_batch.size(0)  # Normalize by batch size
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config['checkpoint_dir'], 'best_model.pth'))
        else:
            patience_counter += 1

        if patience_counter >= config['patience']:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
            
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config['checkpoint_dir'], 'loss_plot.png'))


if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = f'models/{timestamp}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    config = {
        'data_path': 'data/tahoe-100m_5M.h5ad',
        'checkpoint_dir': checkpoint_dir,
        'latent_dim': 20,
        'batch_size': 128,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'n_epochs': 200,
        'patience': 10,
    }
    
    train_model(config)