# model.py (修改版)
import torch
import torch.nn as nn


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 压缩通道维度：最大池化 + 平均池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return x * self.sigmoid(out)


class DQN_Model(nn.Module):
    def __init__(self, n_actions=19):
        super(DQN_Model, self).__init__()

        # 1. 全局画面处理 (输入 6帧 x 100 x 100)
        # 两层卷积 + 空间注意力
        self.global_net = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=2),  # -> 32, 48, 48
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # -> 64, 23, 23
            nn.ReLU(),
            SpatialAttention(),  # 插入注意力
            nn.Flatten()
        )

        # 2. 局部画面处理 (输入 6帧 x 20 x 20)
        # 两个独立的两层卷积网络
        self.local_net = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # 3. 标量处理 (敌我坐标)
        self.scalar_net = nn.Sequential(
            nn.Linear(6 * 6, 64),  # 6帧 * 6个标量
            nn.ReLU()
        )

        # 4. 全连接融合
        # 计算一下 Flatten 后的维度，这里只是估算，建议运行一次 print(x.shape) 确认
        # global: 64*23*23 = 33856
        # local: 32*16*16 = 8192 (x2)
        # scalar: 64
        total_features = 33856 + 8192 * 2 + 64

        self.fc = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, obs):
        # obs 是一个字典，包含转成 Tensor 的数据
        g = self.global_net(obs['global'])
        l1 = self.local_net(obs['local1'])
        l2 = self.local_net(obs['local2'])
        s = self.scalar_net(obs['scalar'])

        # 拼接所有特征
        combined = torch.cat([g, l1, l2, s], dim=1)
        return self.fc(combined)