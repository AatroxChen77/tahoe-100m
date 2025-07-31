import torch.nn as nn
import torch
import torch.nn.functional as F

class CVAE(nn.Module):
    """
    条件变分自编码器 (Conditional Variational AutoEncoder)
    
    这个模型用于学习在给定条件信息（如药物类型、浓度等）下的基因表达数据的潜在表示。
    通过编码器将输入数据和条件信息映射到潜在空间，然后通过解码器重建原始数据。
    
    参数:
        input_dim: 输入维度（基因表达数据的特征数）
        cond_dim: 条件维度（药物等条件信息的特征数）
        latent_dim: 潜在空间维度（默认20）
    """
    def __init__(self, input_dim, cond_dim, latent_dim=20):
        super().__init__()
        
        # 编码器网络：将输入数据和条件信息编码为潜在空间的参数
        # 输入: [基因表达数据, 条件信息] -> 输出: 潜在空间的均值和方差
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + cond_dim, 512),  # 第一层：输入维度+条件维度 -> 512
            nn.ReLU(),                             # 激活函数
            nn.Linear(512, 128),                   # 第二层：512 -> 128
            nn.BatchNorm1d(128),                   # 批归一化，提高训练稳定性
            nn.ReLU()                              # 激活函数
        )
        
        # 潜在空间参数化层
        # 从编码器输出计算潜在空间的均值和方差
        self.fc_mu = nn.Linear(128, latent_dim)      # 均值层：128 -> latent_dim
        self.fc_logvar = nn.Linear(128, latent_dim)  # 对数方差层：128 -> latent_dim

        # 解码器网络：从潜在空间和条件信息重建原始数据
        # 输入: [潜在向量, 条件信息] -> 输出: 重建的基因表达数据
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, 128),   # 第一层：潜在维度+条件维度 -> 128
            nn.ReLU(),                               # 激活函数
            nn.Linear(128, 512),                     # 第二层：128 -> 512
            nn.BatchNorm1d(512),                     # 批归一化
            nn.ReLU(),                               # 激活函数
            nn.Linear(512, input_dim),               # 输出层：512 -> 原始输入维度
            nn.ReLU()                                # 激活函数，确保输出非负（基因表达通常非负）
        )

    def encode(self, x, c):
        """
        编码函数：将输入数据和条件信息编码为潜在空间的参数
        
        参数:
            x: 输入数据（基因表达数据）
            c: 条件信息（药物信息等）
            
        返回:
            mu: 潜在空间的均值
            logvar: 潜在空间的对数方差
        """
        # 将输入数据和条件信息拼接
        # print("[DEBUG] x + c:", torch.cat([x, c], dim=1).shape)
        h = self.encoder(torch.cat([x, c], dim=1))
        # 计算潜在空间的均值和方差
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """
        重参数化技巧：从均值和方差采样得到潜在向量
        这是VAE的核心技巧，使得模型可以进行反向传播
        
        参数:
            mu: 潜在空间的均值
            logvar: 潜在空间的对数方差
            
        返回:
            z: 采样的潜在向量
        """
        # 计算标准差
        std = torch.exp(0.5 * logvar)
        # 从标准正态分布采样噪声
        eps = torch.randn_like(std)
        # 重参数化：z = μ + σ * ε
        return mu + eps * std

    def decode(self, z, c):
        """
        解码函数：从潜在向量和条件信息重建原始数据
        
        参数:
            z: 潜在向量
            c: 条件信息
            
        返回:
            recon_x: 重建的输入数据
        """
        # 将潜在向量和条件信息拼接，然后解码
        return self.decoder(torch.cat([z, c], dim=1))

    def forward(self, x, c):
        """
        前向传播：完整的编码-解码过程
        
        参数:
            x: 输入数据（基因表达数据）
            c: 条件信息（药物信息等）
            
        返回:
            recon_x: 重建的输入数据
            mu: 潜在空间的均值
            logvar: 潜在空间的对数方差
        """
        # 编码：获取潜在空间的参数
        mu, logvar = self.encode(x, c)
        # 重参数化：采样潜在向量
        z = self.reparameterize(mu, logvar)
        # 解码：重建原始数据
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    """
    VAE损失函数：重建损失 + KL散度
    
    参数:
        recon_x: 重建的输入数据
        x: 原始输入数据
        mu: 潜在空间的均值
        logvar: 潜在空间的对数方差
        
    返回:
        total_loss: 总损失（重建损失 + KL散度）
    """
    # 重建损失：衡量重建数据与原始数据的差异
    # 使用MSE损失，reduction='sum'表示对所有元素求和
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL散度：衡量学习到的潜在分布与标准正态分布的差异
    # KL散度公式：-0.5 * Σ(1 + log(σ²) - μ² - σ²)
    # 这里logvar = log(σ²)，所以exp(logvar) = σ²
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # 总损失 = 重建损失 + KL散度
    # 重建损失确保模型能准确重建数据
    # KL散度确保潜在空间接近标准正态分布，便于采样和生成
    return recon_loss + kl_div
