import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SEQUENCE_LENGTH, NUM_ACTIONS


class SpatialAttention(nn.Module):
    """空间注意力机制模块"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 输入2通道(Max+Avg Pooling)，输出1通道权重图
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [Batch*Seq, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return x * self.sigmoid(out)


class HybridNetwork(nn.Module):
    def __init__(self):
        super(HybridNetwork, self).__init__()

        # 1. 整体画面分支 (1x100x100)
        self.global_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU()
        )
        self.attention = SpatialAttention()

        # 2. 局部画面分支 (20x20) - 权重共享或独立均可，这里独立定义
        self.local_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1), nn.ReLU()
        )

        # 3. 向量分支 (FC) - 输入: [Batch, Seq, 5] (x1,y1, x2,y2, dist)
        self.vec_fc = nn.Sequential(nn.Linear(5, 32), nn.ReLU())

        # 计算 Flatten 后的维度 (需根据实际卷积输出计算)
        # 假设: Global->32x23x23, Local->32x16x16
        self.cnn_flat_dim = (32 * 23 * 23) + 2 * (32 * 16 * 16)
        self.lstm_input_dim = self.cnn_flat_dim + 32  # +Vector embedding

        # 4. LSTM 层
        self.lstm = nn.LSTM(input_size=self.lstm_input_dim, hidden_size=512, batch_first=True)

        # 5. 输出层
        self.fc_out = nn.Linear(512, NUM_ACTIONS)

    def forward(self, g_img, l1_img, l2_img, vec):
        # 输入形状: [Batch, Seq, C, H, W]
        B, S, C, H, W = g_img.size()

        # 合并 Batch 和 Seq 维度以进行 CNN 处理
        g_in = g_img.view(B * S, C, H, W)
        l1_in = l1_img.view(B * S, 1, 20, 20)
        l2_in = l2_img.view(B * S, 1, 20, 20)
        v_in = vec.view(B * S, -1)

        # CNN Forward
        g_feat = self.global_cnn(g_in)
        g_feat = self.attention(g_feat)  # 空间注意力
        g_feat = g_feat.view(B * S, -1)

        l1_feat = self.local_cnn(l1_in).view(B * S, -1)
        l2_feat = self.local_cnn(l2_in).view(B * S, -1)

        v_feat = self.vec_fc(v_in)

        # Concat
        combined = torch.cat([g_feat, l1_feat, l2_feat, v_feat], dim=1)

        # LSTM Forward
        # 恢复 [Batch, Seq, Features]
        lstm_in = combined.view(B, S, -1)
        lstm_out, _ = self.lstm(lstm_in)

        # 取最后一个时间步
        q_values = self.fc_out(lstm_out[:, -1, :])
        return q_values