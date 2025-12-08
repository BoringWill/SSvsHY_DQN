import torch
import time
import os
import cv2
from environment import BVNEnv
from model import HybridNetwork
from agent import BVNAgent
from config import DEVICE, FRAME_INTERVAL, NUM_ACTIONS


def main():
    if not os.path.exists('checkpoints'): os.makedirs('checkpoints')

    print(f"--- 初始化环境 (Device: {DEVICE}) ---")
    try:
        env = BVNEnv()
    except Exception as e:
        print(f"错误: {e}")
        return

    # 初始化模型和Agent (如果需要训练，请在 Agent 中添加优化器和学习率)
    net = HybridNetwork().to(DEVICE)
    agent = BVNAgent(net)

    print("--- 开始运行 (按 Ctrl+C 停止) ---")
    print("--- 当前处于调试模式：程序不会自动停止 ---")

    try:
        state_seq = env.reset()
        # 初始状态下 done 变量被忽略，因为我们强制使用 while True

        while True:
            start_time = time.time()

            # 1. 选择动作 (使用动作掩码)
            mask = agent.get_action_mask()
            action = agent.select_action(state_seq, mask)

            if action is not None:
                # 2. 执行一步
                next_state_seq, reward, done, _ = env.step(action)
                state_seq = next_state_seq
            else:
                # 如果没有有效动作，暂停一下
                time.sleep(0.01)
                # done = False  # 确保如果 action=None，不会因遗留 done 状态而出错

            # 3. 打印调试信息：显示 OCR 识别到的真实数值
            p1_val = state_seq[-1]['hp_vals'][0]
            p2_val = state_seq[-1]['hp_vals'][1]
            print(f"Action: {action} | HP P1: {p1_val} | HP P2: {p2_val} | Reward: {reward:.4f}")

            # 4. 调试检查：如果检测到游戏结束条件
            #if done:
                # print("检测到游戏结束条件 (HP <= 0)。请检查 OCR 坐标是否准确。程序继续运行...")
                # pass

            # 5. 帧率控制
            dt = time.time() - start_time
            if dt < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - dt)

    except KeyboardInterrupt:
        print("\n 收到停止信号，程序退出。")
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()