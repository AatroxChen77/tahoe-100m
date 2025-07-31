import torch
from data_utils import get_data, AdataCVAEWrapper
from model import CVAE
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(model, test_loader, device):
    """
    评估CVAE模型在测试集上的性能
    
    参数:
        model: 训练好的CVAE模型
        test_loader: 测试数据的数据加载器
        device: 计算设备（GPU或CPU）
    
    返回:
        all_original: 原始输入数据的numpy数组
        all_recon: 重构数据的numpy数组
    """
    # 将模型设置为评估模式，关闭dropout等训练时的随机性
    model.eval()
    all_recon = []      # 存储所有重构结果
    all_original = []   # 存储所有原始输入
    
    # 不计算梯度，节省内存和计算时间
    with torch.no_grad():
        # 遍历测试数据批次
        for x_batch, c_batch in test_loader:
            # 将数据移动到指定设备（GPU/CPU）
            x_batch = x_batch.to(device)  # 基因表达数据
            c_batch = c_batch.to(device)  # 条件信息（药物、细胞系、浓度等）
            
            # 通过模型前向传播，获得重构结果
            # model返回: (重构数据, 均值, 方差)
            recon_x, _, _ = model(x_batch, c_batch)
            
            # 将结果转换为numpy数组并存储
            all_recon.append(recon_x.cpu().numpy())
            all_original.append(x_batch.cpu().numpy())
    
    # 将所有批次的结果拼接成完整的数据集
    all_recon = np.concatenate(all_recon, axis=0)
    all_original = np.concatenate(all_original, axis=0)
    
    return all_original, all_recon


if __name__ == '__main__':
    # 主程序入口点
    
    # 模型文件路径 - 需要根据实际情况修改
    model_path = 'models/20250731_014559/best_model.pth' 
    # model_path = 'models/best_model.pth' 
    # 数据文件路径 - 包含基因表达数据的AnnData文件
    data_path = 'data/tahoe-100m_5M.h5ad'
    
    # 加载训练和测试数据
    adata_train, adata_val, adata_test = get_data(data_path)
    
    # 定义分类特征（离散变量）
    cat_features = ["drug", "cell_line_id"]  # 药物类型和细胞系ID
    # 定义连续特征（数值变量）
    cont_features = ["drug_conc"]  # 药物浓度
    
    # 创建测试数据集包装器，将AnnData转换为PyTorch可用的格式
    test_dataset = AdataCVAEWrapper(adata_test, cat_features, cont_features)
    # 创建数据加载器，设置批次大小和并行处理
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # 初始化CVAE模型
    # input_dim: 输入维度（基因数量）
    # cond_dim: 条件维度（编码后的条件特征维度）
    model = CVAE(input_dim=adata_test.shape[1], 
                 cond_dim=test_dataset.cond.shape[1])
    
    # 加载训练好的模型权重
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # 设置计算设备（优先使用GPU，否则使用CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 将模型移动到指定设备
    model.to(device)
    
    # 评估模型，获得原始数据和重构数据
    original, reconstructed = evaluate_model(model, test_loader, device)
    
    # 计算每个基因的均方误差（MSE）
    # multioutput='raw_values' 返回每个输出维度（基因）的单独MSE值
    mse_per_gene = mean_squared_error(original, reconstructed, multioutput='raw_values')
    
    # 计算每个基因的决定系数（R²）
    # R²值越接近1，表示重构效果越好
    r2_per_gene = r2_score(original, reconstructed, multioutput='raw_values')
    
    # 输出平均性能指标
    print(f"[INFO] Average MSE per gene: {np.mean(mse_per_gene):.4f}")
    print(f"[INFO] Average R2 per gene: {np.mean(r2_per_gene):.4f}")