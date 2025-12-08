# game_interface.py

import cv2
import numpy as np
import mss
import pygetwindow as gw
import time
from config import GAME_TITLE


class GameInterface:
    def __init__(self):
        self.sct = mss.mss()
        self.monitor = self._find_game_window()

        if self.monitor is None:
            raise RuntimeError(f"未找到窗口名为 '{GAME_TITLE}' 的游戏窗口，请确保游戏已启动。")

        # 颜色阈值 (BGR格式，OpenCV默认)
        self.lower_yellow = np.array([0, 150, 150])
        self.upper_yellow = np.array([100, 255, 255])
        self.lower_blue = np.array([100, 0, 0])
        self.upper_blue = np.array([255, 100, 100])

        self.width = self.monitor['width']
        self.height = self.monitor['height']

    def _find_game_window(self):
        """根据窗口标题查找游戏窗口，并返回其 mss 格式的坐标。"""
        try:
            windows = gw.getWindowsWithTitle(GAME_TITLE)

            if not windows:
                return None

            game_window = windows[0]
            game_window.activate()
            time.sleep(0.5)

            print(f"成功锁定窗口：{GAME_TITLE}")

            # 返回 mss 兼容的字典格式
            return {
                "top": game_window.top,
                "left": game_window.left,
                "width": game_window.width,
                "height": game_window.height,
            }

        except Exception as e:
            print(f"查找窗口时发生错误: {e}")
            return None

    def _find_center(self, img, lower, upper):
        """通过颜色阈值查找角色中心坐标 (用于黄/蓝框识别)。"""
        mask = cv2.inRange(img, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            return (int(x + w / 2), int(y + h / 2))
        return None

    def _crop_local(self, gray_img, center, size):
        """裁剪以中心点为基准的局部 (20x20) 灰度图。"""
        x, y = center
        half = size // 2
        # 处理边界填充
        padded = np.pad(gray_img, ((half, half), (half, half)), 'constant')
        x_pad, y_pad = x + half, y + half
        crop = padded[y_pad - half: y_pad + half, x_pad - half: x_pad + half]
        return cv2.resize(crop, (size, size)) / 255.0

    def get_frame_data(self):
        """捕获帧，识别角色位置，并返回所有输入数据。"""
        raw_screen = np.array(self.sct.grab(self.monitor))
        frame_gray = cv2.cvtColor(raw_screen, cv2.COLOR_BGRA2GRAY)
        frame_bgr = cv2.cvtColor(raw_screen, cv2.COLOR_BGRA2BGR)

        p1_pos = self._find_center(frame_bgr, self.lower_yellow, self.upper_yellow)
        p2_pos = self._find_center(frame_bgr, self.lower_blue, self.upper_blue)

        # 失败处理：使用屏幕的默认位置
        if p1_pos is None: p1_pos = (self.width // 4, self.height - 50)
        if p2_pos is None: p2_pos = (self.width * 3 // 4, self.height - 50)

        # 全局图 (缩放到 100x100)
        global_view = cv2.resize(frame_gray, (100, 100)) / 255.0

        # 局部图 (20x20)
        local_self = self._crop_local(frame_gray, p1_pos, 20)
        local_enemy = self._crop_local(frame_gray, p2_pos, 20)

        # 向量数据 (归一化)
        vec_data = [
            p1_pos[0] / self.width, p1_pos[1] / self.height,  # P1 x, y
            p2_pos[0] / self.width, p2_pos[1] / self.height,  # P2 x, y
            (p1_pos[0] - p2_pos[0]) / self.width,  # dx
            (p1_pos[1] - p2_pos[1]) / self.height  # dy
        ]

        # 转换为 PyTorch 要求的 [C, H, W] 格式 (这里是 [1, H, W])
        global_view = np.expand_dims(global_view, axis=0)
        local_self = np.expand_dims(local_self, axis=0)
        local_enemy = np.expand_dims(local_enemy, axis=0)

        return global_view, local_self, local_enemy, np.array(vec_data, dtype=np.float32)

