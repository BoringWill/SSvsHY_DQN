import gym
import numpy as np
import cv2
import win32gui, win32ui, win32con
from config import *
from collections import deque


class BVNEnv(gym.Env):
    def __init__(self):
        super(BVNEnv, self).__init__()
        self.hwnd = win32gui.FindWindow(None, GAME_TITLE)
        if not self.hwnd: raise Exception(f"未找到窗口: {GAME_TITLE}")

        # 帧缓冲区 (6帧)
        self.frames = deque(maxlen=SEQUENCE_LENGTH)

        # 初始化 OpenCV 识别器（颜色追踪人物）
        # 示例：黄色追踪P1，蓝色追踪P2 (需根据实际调整 HSV 范围)
        self.p1_lower = np.array([20, 100, 100])
        self.p1_upper = np.array([40, 255, 255])
        self.p2_lower = np.array([100, 100, 100])
        self.p2_upper = np.array([120, 255, 255])

    def _grab_screen_win32(self):
        """Win32 高速截图"""
        left, top, right, bot = win32gui.GetClientRect(self.hwnd)
        w, h = right - left, bot - top

        hwindc = win32gui.GetWindowDC(self.hwnd)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, w, h)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (w, h), srcdc, (0, 0), win32con.SRCCOPY)

        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (h, w, 4)  # BGRA

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # 转 BGR

    def _read_hp_bar(self, img, rect):
        """
        OpenCV 读取血条：降低颜色，去除空槽，计算比例。
        """
        x, y, w, h = rect
        # 1. 裁剪
        roi = img[y:y + h, x:x + w]
        # 2. 转 HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # 3. 颜色阈值 (提取有效血量颜色)
        mask = cv2.inRange(hsv, np.array(HP_COLOR_LOWER), np.array(HP_COLOR_UPPER))
        # 4. 计算非零像素比例 (即血量百分比)
        valid_pixels = cv2.countNonZero(mask)
        total_pixels = w * h
        hp_ratio = valid_pixels / total_pixels
        return hp_ratio, (x, y, w, h)

    def _find_character(self, img, lower, upper):
        """简单颜色识别定位人物"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            return (x + w // 2, y + h // 2), (x, y, w, h)
        return (0, 0), (0, 0, 0, 0)  # 未找到

    def _process_frame(self, raw_img):
        """核心处理逻辑：提取特征 + 可视化数据"""
        # 1. 血量读取
        p1_hp, p1_hp_rect = self._read_hp_bar(raw_img, HP_BAR_P1_RECT)
        p2_hp, p2_hp_rect = self._read_hp_bar(raw_img, HP_BAR_P2_RECT)

        # 2. 人物定位
        p1_center, p1_rect = self._find_character(raw_img, self.p1_lower, self.p1_upper)
        p2_center, p2_rect = self._find_character(raw_img, self.p2_lower, self.p2_upper)

        # 3. 图像预处理 (用于 CNN)
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        global_view = cv2.resize(gray, (FULL_IMG_SIZE, FULL_IMG_SIZE))

        # 局部裁剪 (简单实现：以人物为中心裁剪并缩放)
        def crop_local(center):
            cx, cy = center
            if cx == 0: return np.zeros((LOCAL_IMG_SIZE, LOCAL_IMG_SIZE))
            # 简化裁剪逻辑，实际需处理边界
            crop = gray[max(0, cy - 50):cy + 50, max(0, cx - 50):cx + 50]
            if crop.size == 0: return np.zeros((LOCAL_IMG_SIZE, LOCAL_IMG_SIZE))
            return cv2.resize(crop, (LOCAL_IMG_SIZE, LOCAL_IMG_SIZE))

        local_p1 = crop_local(p1_center)
        local_p2 = crop_local(p2_center)

        # 4. 向量数据
        vec = [p1_center[0], p1_center[1], p2_center[0], p2_center[1],
               np.linalg.norm(np.array(p1_center) - np.array(p2_center))]

        # 5. 可视化 (在原图画框)
        vis_img = raw_img.copy()
        # 画血条识别框
        cv2.rectangle(vis_img, (p1_hp_rect[0], p1_hp_rect[1]),
                      (p1_hp_rect[0] + p1_hp_rect[2], p1_hp_rect[1] + p1_hp_rect[3]), (0, 255, 0), 2)
        cv2.putText(vis_img, f"P1 HP: {p1_hp:.2f}", (p1_hp_rect[0], p1_hp_rect[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 1)
        # 画人物框
        cv2.rectangle(vis_img, (p1_rect[0], p1_rect[1]), (p1_rect[0] + p1_rect[2], p1_rect[1] + p1_rect[3]),
                      (0, 255, 255), 2)

        return {
            "global": global_view,
            "local_p1": local_p1,
            "local_p2": local_p2,
            "vec": vec,
            "hp": (p1_hp, p2_hp),
            "vis_img": vis_img
        }

    def reset(self):
        self.frames.clear()
        # 重置游戏略
        img = self._grab_screen_win32()
        data = self._process_frame(img)
        for _ in range(SEQUENCE_LENGTH):
            self.frames.append(data)
        return list(self.frames)

    def step(self, action):
        from game_actions import execute_action
        execute_action(action)

        img = self._grab_screen_win32()
        new_data = self._process_frame(img)
        self.frames.append(new_data)

        # 自定义奖励：HP 差值变化
        curr_hp_diff = new_data['hp'][0] - new_data['hp'][1]
        prev_hp_diff = self.frames[-2]['hp'][0] - self.frames[-2]['hp'][1]
        reward = (curr_hp_diff - prev_hp_diff) * 10

        done = new_data['hp'][0] <= 0 or new_data['hp'][1] <= 0

        # 显示可视化窗口
        cv2.imshow("AI Vision Debug", new_data['vis_img'])
        cv2.waitKey(1)

        return list(self.frames), reward, done, {}