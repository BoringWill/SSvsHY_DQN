import torch
import random
import numpy as np
from collections import deque
from config import DEVICE, NUM_ACTIONS, REPLAY_BUFFER_SIZE, SEQUENCE_LENGTH, MAX_RAGE, ULTIMATE_ACTIONS


# --- 辅助函数 ---
def _frames_to_tensor(frames_list, device):
    """将帧列表转换为用于模型的张量，包含归一化的时间和怒气值。"""

    # 提取向量数据 (5维: p1_x, p1_y, p2_x, p2_y, distance)
    vecs = np.stack([f['vec'] for f in frames_list], axis=0)

    # 提取时间数据并归一化 (最大时间假设为 99)
    times = np.array([f['time'] for f in frames_list], dtype=np.float32) / 99.0
    times = times.reshape(-1, 1)

    # 提取怒气值数据并归一化 (使用 MAX_RAGE=3)
    rage_p1 = np.array([f['rage_vals'][0] for f in frames_list], dtype=np.float32) / MAX_RAGE
    rage_p2 = np.array([f['rage_vals'][1] for f in frames_list], dtype=np.float32) / MAX_RAGE
    rage_p1 = rage_p1.reshape(-1, 1)
    rage_p2 = rage_p2.reshape(-1, 1)

    # 拼接向量、时间、怒气数据
    # 维度: 5 (vec) + 1 (time) + 2 (rage) = 8
    full_vecs = np.concatenate([vecs, times, rage_p1, rage_p2], axis=1)

    # 提取图像数据
    g_imgs = np.stack([f['global'] for f in frames_list], axis=0)
    l1_imgs = np.stack([f['local_p1'] for f in frames_list], axis=0)
    l2_imgs = np.stack([f['local_p2'] for f in frames_list], axis=0)

    # 转换为 PyTorch 张量 (增加 Batch 维度)
    # 图像张量形状: [B=1, Seq_len, Channels=1, H, W]
    G_full = torch.from_numpy(g_imgs).unsqueeze(0).unsqueeze(2).to(device)
    L1_img = torch.from_numpy(l1_imgs).unsqueeze(0).unsqueeze(2).to(device)
    L2_img = torch.from_numpy(l2_imgs).unsqueeze(0).unsqueeze(2).to(device)

    # 向量张量形状: [B=1, Seq_len, Feature_dim=8]
    Vec = torch.from_numpy(full_vecs).unsqueeze(0).to(device)
    return G_full, L1_img, L2_img, Vec


# --- 代理类 ---
class BVNAgent:
    def __init__(self, model):
        self.model = model
        self.memory = deque(maxlen=REPLAY_BUFFER_SIZE)
        self.epsilon = 1.0
        self.current_state_data = None

    def get_action_mask(self):
        """
        根据当前 P1 怒气值和不同动作的消耗，生成动作掩码。
        """
        mask = np.ones(NUM_ACTIONS, dtype=bool)

        if self.current_state_data is None:
            return mask

            # 提取最新帧数据中的 P1 怒气值
        current_rage = self.current_state_data[-1]['rage_vals'][0]

        # 遍历所有需要怒气消耗的动作
        for idx, cost in ULTIMATE_ACTIONS.items():
            if current_rage < cost:
                # 如果怒气不足以支付该动作的消耗，则禁用
                mask[idx] = False

        return mask

    def select_action(self, state_sequence, mask):
        """ε-greedy 选择动作"""
        self.current_state_data = state_sequence

        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            # 如果所有动作都被禁用，则默认选择一个安全动作（例如左移动 'a', 索引 0）
            return 0

        if random.random() < self.epsilon:
            return np.random.choice(valid_indices)
        else:
            G_full, L1_img, L2_img, Vec = _frames_to_tensor(state_sequence, DEVICE)
            with torch.no_grad():
                q_values = self.model(G_full, L1_img, L2_img, Vec).squeeze(0)

            mask_tensor = torch.from_numpy(mask).to(DEVICE)
            # 使用掩码将无效动作的 Q 值设置为极小值
            q_values_masked = q_values.masked_fill(~mask_tensor, -float('inf'))
            return torch.argmax(q_values_masked).item()

    def store_transition(self, transition):
        _, action, _, _, _ = transition
        if action is not None:
            self.memory.append(transition)