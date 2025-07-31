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
    训练CVAE模型的主函数
    
    参数:
        config: 包含训练配置的字典，包括数据路径、超参数等
    """
    # 加载并预处理数据，分割为训练集、验证集和测试集
    print("[INFO] 正在加载数据...")
    adata_train, adata_val, adata_test = get_data(config['data_path'])
    
    # 定义分类特征和连续特征
    cat_features = ["drug", "cell_line_id"]      # 分类特征：药物类型、细胞系ID（需要one-hot编码）
    cont_features = ["drug_conc"]                # 连续特征：药物浓度（需要标准化）

    # 创建数据集包装器，将AnnData对象转换为PyTorch可用的格式
    print("[INFO] 正在创建数据集...")
    train_dataset = AdataCVAEWrapper(adata_train, cat_features, cont_features)
    val_dataset = AdataCVAEWrapper(adata_val, cat_features, cont_features)
    test_dataset = AdataCVAEWrapper(adata_test, cat_features, cont_features)
    
    # 打印数据集信息
    print(f"[INFO] 训练数据集大小: {len(train_dataset)}")
    print(f"[INFO] 验证数据集大小: {len(val_dataset)}")
    print(f"[INFO] 测试数据集大小: {len(test_dataset)}")
    
    # 创建数据加载器，用于批量训练
    # shuffle=True: 训练时随机打乱数据
    # num_workers=4: 使用4个进程并行加载数据
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # 初始化CVAE模型
    # input_dim: 基因表达数据的特征数（从训练数据中获取）
    # cond_dim: 条件信息的特征数（从数据集包装器中获取）
    # latent_dim: 潜在空间维度（从配置中获取）
    print("[INFO] 正在初始化模型...")
    model = CVAE(input_dim=adata_train.shape[1], 
                 cond_dim=train_dataset.cond.shape[1], 
                 latent_dim=config['latent_dim']) # 实例化模型
    
    # 初始化优化器：AdamW优化器，包含权重衰减以防止过拟合
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # 检测并设置设备（GPU或CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 使用设备: {device}")
    model.to(device)

    # 初始化训练记录变量
    train_losses = []        # 记录每个epoch的训练损失
    val_losses = []          # 记录每个epoch的验证损失
    best_val_loss = float('inf')  # 记录最佳验证损失
    patience_counter = 0     # 早停计数器

    print("[INFO] 开始训练...")
    # 主训练循环
    for epoch in range(config['n_epochs']):
        # ==================== 训练阶段 ====================
        model.train()  # 设置为训练模式（启用dropout、batch norm等）
        train_loss = 0
        
        # 遍历训练数据批次
        for batch_idx, (x_batch, c_batch) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["n_epochs"]}')):
            # 将数据移动到指定设备（GPU/CPU）
            x_batch = x_batch.to(device)  # 基因表达数据
            c_batch = c_batch.to(device)  # 条件信息（药物等）
            
            # 添加调试信息（只在第一个epoch的前几个批次打印）
            if epoch == 0 and batch_idx < 3:
                print(f"[DEBUG] Batch {batch_idx}: x_batch shape: {x_batch.shape}, c_batch shape: {c_batch.shape}")
            
            # 前向传播：通过模型获取重建结果和潜在空间参数
            recon_x, mu, logvar = model(x_batch, c_batch)
            
            # 计算损失：重建损失 + KL散度，并按批次大小归一化
            loss = loss_function(recon_x, x_batch, mu, logvar) / x_batch.size(0)
            
            # 反向传播：计算梯度
            optimizer.zero_grad()  # 清空之前的梯度
            loss.backward()        # 计算梯度
            optimizer.step()       # 更新模型参数
            
            # 累积训练损失
            train_loss += loss.item()
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ==================== 验证阶段 ====================
        model.eval()  # 设置为评估模式（禁用dropout、batch norm等）
        val_loss = 0
        
        # 在验证集上评估模型（不计算梯度）
        with torch.no_grad():
            for x_batch, c_batch in val_loader:
                x_batch = x_batch.to(device)
                c_batch = c_batch.to(device)
                # print(f"[DEBUG] x_batch shape: {x_batch.shape}, c_batch shape: {c_batch.shape}")
                recon_x, mu, logvar = model(x_batch, c_batch)
                loss = loss_function(recon_x, x_batch, mu, logvar) / x_batch.size(0)
                val_loss += loss.item()
        
        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # 打印当前epoch的训练和验证损失
        print(f'[INFO] Epoch {epoch+1}/{config["n_epochs"]}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # ==================== 模型保存和早停 ====================
        # 如果验证损失改善，保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # 重置早停计数器
            # 保存模型参数到文件
            torch.save(model.state_dict(), os.path.join(config['checkpoint_dir'], 'best_model.pth'))
            print(f'[INFO] 保存最佳模型，验证损失: {best_val_loss:.4f}')
        else:
            patience_counter += 1  # 增加早停计数器

        # 早停机制：如果验证损失连续patience个epoch没有改善，则停止训练
        if patience_counter >= config['patience']:
            print(f'[INFO] 早停触发，在第 {epoch+1} 个epoch停止训练')
            break
            
    # ==================== 训练结果可视化 ====================
    print("[INFO] 正在生成训练曲线...")
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    # 保存损失曲线图
    plt.savefig(os.path.join(config['checkpoint_dir'], 'loss_plot.png'))
    # plt.close()
    
    print(f"[INFO] 训练完成！最佳验证损失: {best_val_loss:.4f}")
    print(f"[INFO] 模型和图表已保存到: {config['checkpoint_dir']}")


if __name__ == '__main__':
    # 创建带时间戳的模型保存目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = f'models/{timestamp}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 训练配置参数
    config = {
        'data_path': 'data/tahoe-100m_5M.h5ad',  # 数据文件路径
        'checkpoint_dir': checkpoint_dir,         # 模型保存目录
        'latent_dim': 20,                         # 潜在空间维度
        'batch_size': 128,                        # 批次大小
        'learning_rate': 1e-3,                    # 学习率
        'weight_decay': 1e-4,                     # 权重衰减（L2正则化）
        'n_epochs': 1000,                          # 最大训练轮数
        'patience': 200,                           # 早停耐心值
    }
    
    # 开始训练
    train_model(config)