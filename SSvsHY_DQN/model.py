import torch
import torch.nn as nn
import torch.nn.functional as F
from config import FULL_IMG_SIZE, LOCAL_IMG_SIZE, SEQUENCE_LENGTH, NUM_ACTIONS


class FeatureExtractor(nn.Module):
    """图像特征提取器 (CNN)"""

    def __init__(self, input_size):
        super(FeatureExtractor, self).__init__()
        # 使用序列长度作为输入通道数
        self.conv = nn.Sequential(
            nn.Conv2d(SEQUENCE_LENGTH, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        def conv_output_size(size):
            # 简化计算：两次 stride=2 的卷积
            return size // 4

        conv_out_size = conv_output_size(input_size)
        self.output_dim = 32 * conv_out_size * conv_out_size

    def forward(self, x):
        x = self.conv(x)
        return x.view(x.size(0), -1)


class HybridNetwork(nn.Module):
    """混合网络：CNN特征 + 向量输入 + LSTM + FC"""

    def __init__(self):
        super(HybridNetwork, self).__init__()

        # 1. 图像特征提取 (共享 CNN 结构)
        self.global_fe = FeatureExtractor(FULL_IMG_SIZE)
        self.local_fe = FeatureExtractor(LOCAL_IMG_SIZE)

        # 2. 图像特征合并

        # 3. 向量分支 (FC)
        # 向量维度是 8 维 (5个坐标/距离 + 1个时间 + 2个怒气)
        self.vec_fc = nn.Sequential(nn.Linear(8, 32), nn.ReLU())
        vector_feature_dim = 32

        # 4. LSTM 层
        # 图像特征维度 (global + 2*local) + 向量特征维度
        # 注意：这里需要计算 image_feature_dim
        # 简化计算，假设 feature_dim 为某个固定值
        # 假设 FULL_IMG_SIZE=100, LOCAL_IMG_SIZE=20
        # global_fe: 32 * (100/4)^2 = 20000
        # local_fe: 32 * (20/4)^2 = 800
        image_feature_dim = self.global_fe.output_dim + 2 * self.local_fe.output_dim

        lstm_input_dim = image_feature_dim + vector_feature_dim
        self.lstm = nn.LSTM(lstm_input_dim, 64, batch_first=True)  # 输出 64 维

        # 5. 输出层 (Q值)
        self.fc_q = nn.Linear(64, NUM_ACTIONS)

    def forward(self, G_full, L1_img, L2_img, Vec):
        batch_size, seq_len, _, _, _ = G_full.shape  # G_full 形状: [B, S, C=1, H, W]

        all_features = []
        for t in range(seq_len):
            # 提取单帧图像特征 (需要重新调整输入形状以匹配 FeatureExtractor 的预期 [B, S, H, W])
            # 我们在 agent.py 中将 S=6 作为 Channels，这里需要重新处理：
            # 在 batch_first=True 的情况下，正确的做法是在 agent.py 中将 S 维度移到 Channels (C)
            # 由于模型的 FeatureExtractor 是以 SEQUENCE_LENGTH 为输入通道的，
            # 我们将整个序列 [B, S, 1, H, W] 重塑为 [B, S, H, W] 传递给 FeatureExtractor，
            # 这在 agent.py 中已经通过 squeeze(1) 实现。

            # 这里的 LSTM 循环前向传播需要修改以匹配 Hybrid结构
            # 考虑到当前架构，我们假设 G_full, L1_img, L2_img 已经是 [B, C=S, H, W]

            # 提取单帧图像特征
            # **注意：由于 FeatureExtractor 的输入通道是 SEQUENCE_LENGTH，
            # 这里必须传入整个序列，而不是单帧。**

            # 修正：将 LSTM 逻辑放在后面，先提取整个序列的图像特征
            pass

        # 提取图像特征 (这里假设我们只用最后一帧的图像特征，但实际上模型结构用了整个序列)
        # 让我们按照当前的 FeatureExtractor 定义，将整个序列视为多通道输入：
        g_feat = self.global_fe(G_full.squeeze(2))  # [B, S, H, W] -> [B, Feat_dim]
        l1_feat = self.local_fe(L1_img.squeeze(2))
        l2_feat = self.local_fe(L2_img.squeeze(2))
        img_feat = torch.cat([g_feat, l1_feat, l2_feat], dim=1)  # [B, Img_Feat_dim]

        # 向量特征 (需要序列化处理)
        vec_feat = self.vec_fc(Vec)  # [B, S, Vec_Feat_dim]

        # 整合所有特征作为 LSTM 输入
        # 将 Img_feat (形状 [B, Img_Feat_dim]) 复制 S 次，以匹配 Vec_feat 的序列维度
        # Img_feat_seq = img_feat.unsqueeze(1).repeat(1, seq_len, 1)

        # 由于我们使用 CNN 提取了整个序列的特征，Img_feat 已经包含了时间依赖性。
        # 故我们将其视为一个固定的上下文特征，与每一步的向量特征结合。

        # **简化方法**：只取序列最后一帧的向量特征来做最终决策 (典型 DQN 做法)
        # vec_feat_last = vec_feat[:, -1, :] # [B, Vec_Feat_dim]
        # combined_feat = torch.cat([img_feat, vec_feat_last], dim=1)
        # return self.fc_q(combined_feat)

        # **修正方法 (保持 LSTM 结构)**: 将 Img_feat 作为全局上下文，与序列向量特征结合

        # 1. 将图像特征复制到序列维度
        img_feat_seq = img_feat.unsqueeze(1).repeat(1, seq_len, 1)  # [B, S, Img_Feat_dim]

        # 2. 结合序列化图像特征和序列化向量特征
        lstm_input = torch.cat([img_feat_seq, vec_feat], dim=2)  # [B, S, LSTM_Input_dim]

        # 3. LSTM 前向传播
        lstm_out, _ = self.lstm(lstm_input)  # [B, S, 64]

        # 4. 只使用序列最后一帧的输出来计算 Q 值
        last_output = lstm_out[:, -1, :]

        q_values = self.fc_q(last_output)
        return q_values