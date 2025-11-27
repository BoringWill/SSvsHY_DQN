# game_env.py
import numpy as np
import cv2
import time
from collections import deque
import config
import utils_capture
import utils_control
from perception import GamePerception


class DeadBleachEnv:
    def __init__(self, hwnd):
        self.hwnd = hwnd
        self.perception = GamePerception()
        self.action_space_n = 19  # 动作数量

        # 6帧堆叠的队列
        self.frames = {
            'global': deque(maxlen=6),
            'local1': deque(maxlen=6),
            'local2': deque(maxlen=6),
            'scalar': deque(maxlen=6)
        }

        self.last_hp_own = 1.0
        self.last_hp_enemy = 1.0

    def reset(self):
        """重置游戏，开始新一局"""
        print("环境重置...")
        # 1. 发送重启按键 (比如不断按J确认)
        utils_control.execute_model_action(self.hwnd, 5, config.ACTION_MAP)
        time.sleep(3.0)  # 等待开场

        # 2. 抓取初始帧并填充队列
        frame = utils_capture.capture_frame_by_hwnd(self.hwnd)
        g, l1, l2, s = self.perception.process(frame)

        for _ in range(6):
            self.frames['global'].append(g)
            self.frames['local1'].append(l1)
            self.frames['local2'].append(l2)
            self.frames['scalar'].append(s)

        self.last_hp_own = 1.0
        self.last_hp_enemy = 1.0

        return self._get_observation()

    def step(self, action_idx):
        """
        Gym 核心：执行动作 -> 获取反馈
        """
        # 1. 执行动作
        utils_control.execute_model_action(self.hwnd, action_idx, config.ACTION_MAP)

        # 2. 截图 & 感知
        frame = utils_capture.capture_frame_by_hwnd(self.hwnd)
        if frame is None:  # 容错
            return self._get_observation(), 0, True, {}

        g, l1, l2, s = self.perception.process(frame)

        # 3. 更新队列
        self.frames['global'].append(g)
        self.frames['local1'].append(l1)
        self.frames['local2'].append(l2)
        self.frames['scalar'].append(s)

        # 4. 计算奖励 (核心难点!!!)
        curr_hp_own, curr_hp_enemy = self._get_hp_from_pixels(frame)

        # 奖励公式：(我对敌人造成的伤害 * 权重) - (我受到的伤害 * 权重)
        damage_dealt = (self.last_hp_enemy - curr_hp_enemy)
        damage_taken = (self.last_hp_own - curr_hp_own)

        reward = (damage_dealt * 200) - (damage_taken * 100)

        # 存活奖励 (每活一帧给一点点分，防止它消极避战或自杀)
        reward += 0.1

        # 更新血量记录
        self.last_hp_own = curr_hp_own
        self.last_hp_enemy = curr_hp_enemy

        # 5. 判断结束
        done = False
        if curr_hp_own <= 0 or curr_hp_enemy <= 0:
            done = True
            if curr_hp_enemy <= 0: reward += 1000  # 胜利大奖

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """将队列堆叠成 numpy 数组"""
        return {
            'global': np.array(self.frames['global']),  # Shape: (6, 100, 100)
            'local1': np.array(self.frames['local1']),  # Shape: (6, 20, 20)
            'local2': np.array(self.frames['local2']),
            'scalar': np.array(self.frames['scalar']).flatten()  # Shape: (36,)
        }

    def _get_hp_from_pixels(self, frame):
        """
        【重要】你需要修改这里的坐标！
        打开画图，鼠标指到游戏血条位置，看左下角坐标。
        """
        # 示例：假设我方血条在 (x=50, y=30) 到 (x=250, y=40)
        # 红色像素判断
        # my_hp_bar = frame[30:40, 50:250]
        # hp_percent = ... (计算红色点数量 / 总像素数)

        # 现在先返回假数据，否则跑不通
        return 1.0, 1.0