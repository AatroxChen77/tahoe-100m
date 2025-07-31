import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def get_data(file_path, test_size=0.1, val_size=0.1, random_state=42):
    """
    加载、预处理并分割数据为训练集、验证集和测试集
    
    参数:
        file_path: AnnData文件路径（.h5ad格式）
        test_size: 测试集比例（默认0.1，即10%）
        val_size: 验证集比例（默认0.1，即10%）
        random_state: 随机种子，确保结果可重现
        
    返回:
        adata_train: 训练集AnnData对象
        adata_val: 验证集AnnData对象  
        adata_test: 测试集AnnData对象
    """
    # 加载AnnData文件（单细胞基因表达数据）
    print(f"[INFO] 正在加载数据文件: {file_path}")
    adata = sc.read_h5ad(file_path)
    print(f"[INFO] 原始数据形状: {adata.shape}")
    
    # 数据预处理步骤
    print("[INFO] 正在进行数据预处理...")
    
    # 1. 标准化：将每个细胞的总计数标准化到10,000
    # 这是单细胞数据分析的标准做法，用于消除测序深度的影响
    sc.pp.normalize_total(adata, target_sum=1e4)
    
    # 2. 对数变换：log1p变换（log(1+x)）
    # 将基因表达数据转换为对数尺度，减少高表达基因的权重
    sc.pp.log1p(adata)
    
    # 3. 高变异基因选择：选择表达变异最大的2000个基因
    # 这些基因通常包含更多的生物学信息
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True, flavor="seurat")
    
    # 数据集分割
    print("[INFO] 正在进行数据集分割...")
    
    # 第一步：将数据分为 (训练+验证集) 和 测试集
    train_val_idx, test_idx = train_test_split(
        adata.obs.index,  # 使用观察值的索引进行分割
        test_size=test_size,  # 测试集占10%
        random_state=random_state  # 确保结果可重现
    )
    
    # 第二步：将 (训练+验证集) 分为训练集和验证集
    # 注意：这里的test_size需要调整，因为是在剩余数据中的比例
    train_idx, val_idx = train_test_split(
        train_val_idx,  # 训练+验证集的索引
        test_size=val_size/(1-test_size),  # 调整比例：0.1/(1-0.1) = 0.111
        random_state=random_state
    )
    
    # 根据索引创建三个数据集
    adata_train = adata[train_idx].copy()  # 训练集（约80%的数据）
    adata_val = adata[val_idx].copy()      # 验证集（约10%的数据）
    adata_test = adata[test_idx].copy()    # 测试集（约10%的数据）
    
    # print(f"数据集分割完成:")
    # print(f"  训练集: {adata_train.shape[0]} 个细胞, {adata_train.shape[1]} 个基因")
    # print(f"  验证集: {adata_val.shape[0]} 个细胞, {adata_val.shape[1]} 个基因")
    # print(f"  测试集: {adata_test.shape[0]} 个细胞, {adata_test.shape[1]} 个基因")
    
    return adata_train, adata_val, adata_test

class AdataCVAEWrapper(Dataset):
    """
    AnnData到PyTorch数据集的包装器
    
    这个类将AnnData对象转换为PyTorch可以使用的Dataset格式，
    同时处理分类特征（如药物类型、细胞系）和连续特征（如药物浓度）。
    
    参数:
        adata: AnnData对象，包含基因表达数据和元数据
        cat_features: 分类特征列表（如["drug", "cell_line_id"]）
        cont_features: 连续特征列表（如["drug_conc"]）
    """
    def __init__(self, adata, cat_features, cont_features):
        # 存储基因表达数据（稀疏矩阵格式）
        self.X = adata.X
        print(f"[DEBUG] 基因表达数据形状: {self.X.shape}")
        
        # 处理分类特征：进行one-hot编码
        # 例如：药物类型["A", "B", "C"] -> [[1,0,0], [0,1,0], [0,0,1]]
        print(f"处理分类特征: {cat_features}")
        self.cat_data = pd.get_dummies(adata.obs[cat_features], drop_first=False).values.astype(np.float32)
        self.cat_data = torch.from_numpy(self.cat_data)
        print(f"分类特征one-hot编码后形状: {self.cat_data.shape}")
        
        # 处理连续特征：进行标准化（Z-score标准化）
        # 公式：(x - mean) / std
        print(f"处理连续特征: {cont_features}")
        cont = adata.obs[cont_features].values.astype(np.float32)
        cont = (cont - cont.mean(axis=0)) / cont.std(axis=0)  # Z-score标准化
        self.cont_data = torch.from_numpy(cont)
        print(f"连续特征标准化后形状: {self.cont_data.shape}")
        
        # 将分类特征和连续特征拼接成条件信息
        # 维度：[样本数, 分类特征数 + 连续特征数]
        self.cond = torch.cat([self.cat_data, self.cont_data], dim=1)
        print(f"条件信息最终形状: {self.cond.shape}")

    def __len__(self):
        """
        返回数据集中的样本数量
        """
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        获取指定索引的样本
        
        参数:
            idx: 样本索引
            
        返回:
            x_row: 基因表达数据（一维张量）
            c: 条件信息（一维张量，包含药物信息等）
        """
        # 获取基因表达数据
        # self.X[idx]是稀疏矩阵的一行，需要转换为密集格式
        # toarray()将稀疏矩阵转换为numpy数组
        # squeeze()移除多余的维度，确保是一维数组
        x_row = torch.tensor(self.X[idx].toarray().squeeze(), dtype=torch.float32)
        
        # 获取条件信息（药物类型、细胞系、药物浓度等）
        c = self.cond[idx]

        # 添加调试信息（只在第一个样本时打印）
        # if idx == 0:
        #     print(f"[DEBUG] 基因表达数据形状: {x_row.shape}")
        #     print(f"[DEBUG] 条件信息形状: {c.shape}")
        
        return x_row, c 