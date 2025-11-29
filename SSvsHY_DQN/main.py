from environment import BVNEnv
from model import HybridNetwork
from agent import BVNAgent
from config import *
import torch
import time


def main():
    env = BVNEnv()
    net = HybridNetwork().to(DEVICE)
    agent = BVNAgent(net)

    for episode in range(1000):
        state_seq = env.reset()
        done = False

        while not done:
            start_ts = time.time()

            # 1. 获取 Mask (基于当前最后一帧的信息)
            # 注意：实际中可能需要单独提取能量值传入
            mask = agent.get_action_mask({})

            # 2. 选择动作
            action = agent.select_action(state_seq, mask)

            if action is not None:
                # 3. 执行并获取下一步
                next_state_seq, reward, done, _ = env.step(action)

                # 4. 存储
                agent.store_transition((state_seq, action, reward, next_state_seq, done))

                state_seq = next_state_seq

                # 5. 训练 (省略)
                # agent.learn()
            else:
                # 如果无动作执行，仅更新画面但不执行 step (或执行空操作)
                # 视具体逻辑而定
                time.sleep(0.01)

            # 控制帧率 50-150ms
            dt = time.time() - start_ts
            if dt < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - dt)


if __name__ == "__main__":
    main()