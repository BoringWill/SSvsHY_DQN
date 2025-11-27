# train.py
import win32gui
import torch
import numpy as np
import random
from collections import deque
import torch.optim as optim
import torch.nn.functional as F

from game_env import DeadBleachEnv
from model import DQN_Model
import config

# --- 超参数 ---
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 10000
TARGET_UPDATE = 10
MEMORY_SIZE = 5000
LR = 1e-4


# --- 经验回放池 ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def train():
    # 1. 初始化
    hwnd = win32gui.FindWindow(None, config.GAME_WINDOW_TITLE)
    if not hwnd: print("未找到窗口"); return

    env = DeadBleachEnv(hwnd)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_net = DQN_Model().to(device)
    target_net = DQN_Model().to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    steps_done = 0

    # 2. 训练循环
    for i_episode in range(1000):
        state = env.reset()
        # 状态字典转 Tensor
        # 注意：这里需要把 numpy 转成 torch 并且增加 batch 维度 (unsqueeze)
        # 为了代码简洁，这里省略了具体的转换函数，你需要写一个 helper function

        while True:
            # --- 选择动作 (Epsilon-Greedy) ---
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                      np.exp(-1. * steps_done / EPSILON_DECAY)
            steps_done += 1

            if random.random() < epsilon:
                action = random.randint(0, 18)
            else:
                with torch.no_grad():
                    # 这里 state 需要转 tensor
                    # q_vals = policy_net(state_tensor)
                    # action = q_vals.max(1)[1].item()
                    action = 0  # 占位

            # --- 执行 ---
            next_state, reward, done, _ = env.step(action)

            # --- 存储 ---
            memory.push(state, action, reward, next_state, done)
            state = next_state

            # --- 优化 (Learn) ---
            if len(memory) > BATCH_SIZE:
                transitions = memory.sample(BATCH_SIZE)
                # ... 这里执行标准的 DQN Loss 计算和 backward() ...
                # 1. 提取 batch
                # 2. 计算 Q(s, a)
                # 3. 计算 Target Q = r + gamma * max Q(s', a')
                # 4. Loss = MSE(Q, Target Q)
                # 5. optimizer.step()
                pass

            if done:
                break

        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            print(f"Episode {i_episode} finished.")


if __name__ == "__main__":
    train()