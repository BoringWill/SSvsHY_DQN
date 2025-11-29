import torch
import random
import numpy as np
from config import *


class BVNAgent:
    def __init__(self, model):
        self.model = model
        self.memory = []  # 简化列表，实际请用 Deque 或 SumTree
        self.epsilon = 1.0

    def get_action_mask(self, info):
        """
        根据当前状态生成 mask。
        1 = 可用, 0 = 不可用
        """
        mask = np.ones(NUM_ACTIONS)

        # 假设 info 中包含了从画面 OCR 提取的怒气值 (这里简化为假设逻辑)
        # current_energy = info['energy'] 
        current_energy = 0  # 示例：假设没能量

        # 如果能量不足，屏蔽必杀技
        if current_energy < 50:
            for idx in ULTIMATE_ACTIONS:
                mask[idx] = 0

        return mask

    def select_action(self, state_sequence, mask):
        """
        Action Selection with Masking
        """
        valid_actions = np.where(mask == 1)[0]

        # 要求：当前可用动作为空则模型不输出
        if len(valid_actions) == 0:
            return None

        if random.random() < self.epsilon:
            return np.random.choice(valid_actions)

        # 模型预测
        with torch.no_grad():
            # 需要将 state_sequence 转换为 Tensor 并增加 Batch 维度
            # 这里省略数据转换代码...
            g_img = torch.tensor(...)
            q_values = self.model(g_img, ...)

            # Masking: 将无效动作的 Q 值设为负无穷
            # q_values[0, i] = -inf where mask[i] == 0
            for i in range(NUM_ACTIONS):
                if mask[i] == 0:
                    q_values[0, i] = -float('inf')

            return torch.argmax(q_values).item()

    def store_transition(self, transition):
        # 要求：动作无效时不纳入记忆库
        state, action, reward, next_state, done = transition
        if action is not None:
            self.memory.append(transition)