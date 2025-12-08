import gym
import numpy as np
import cv2
import time
import win32gui, win32ui, win32con
from collections import deque
import pytesseract
import platform
# 确保所有配置常量都已导入，这里假设 config.py 已经就绪
from config import *

# --- 强制设置 Tesseract 路径 ---
if platform.system() == 'Windows':
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


class BVNEnv(gym.Env):
    def __init__(self):
        super(BVNEnv, self).__init__()
        self.hwnd = win32gui.FindWindow(None, GAME_TITLE)
        if not self.hwnd:
            raise Exception(f"未找到窗口: {GAME_TITLE}")

        if win32gui.IsIconic(self.hwnd):
            raise Exception("窗口已最小化，请还原窗口。")

        self.frames = deque(maxlen=SEQUENCE_LENGTH)

    def _grab_screen_win32(self):
        left, top, right, bot = win32gui.GetClientRect(self.hwnd)
        w, h = right - left, bot - top
        if w == 0 or h == 0: return np.zeros((100, 100, 3), dtype=np.uint8)

        hwindc = win32gui.GetWindowDC(self.hwnd)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, w, h)
        memdc.SelectObject(bmp)

        # 修正 BitBlt 参数：确保正确截图 w x h 区域
        memdc.BitBlt((0, 0), (w, h), srcdc, (0, 0), win32con.SRCCOPY)

        signedIntsArray = bmp.GetBitmapBits(True)
        img = np.frombuffer(signedIntsArray, dtype='uint8')
        img.shape = (h, w, 4)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())

        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def _read_ocr_value(self, img, rect, max_val=None):
        """
        通用 OCR 读取整数数字 (仅用于 HP/Time，使用单色 DIGIT_COLOR 阈值)。
        """
        x, y, w, h = rect
        # 边界检查
        if y + h > img.shape[0] or x + w > img.shape[1] or w <= 0 or h <= 0:
            return 0, 0.0, (x, y, w, h)

        roi = img[y:y + h, x:x + w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 单色过滤（用于白色数字）
        mask = cv2.inRange(hsv, np.array(DIGIT_COLOR_LOWER), np.array(DIGIT_COLOR_UPPER))

        # 反转：得到白底黑字（Tesseract 偏爱格式）
        final_img = cv2.bitwise_not(mask)

        # 识别整数
        config = '--psm 7 -c tessedit_char_whitelist=0123456789'
        try:
            text = pytesseract.image_to_string(final_img, config=config).strip()
            val = int(text) if text else 0
        except Exception as e:
            val = 0

        ratio = np.clip(val / max_val, 0.0, 1.0) if max_val else 0.0

        return val, ratio, (x, y, w, h)

    # 🔥 新增/替换：通过颜色统计读取怒气/气量等级 (0, 1, 2, 或 3)
    def _read_gauge_level_by_color(self, raw_img, rect):
        """
        通过颜色统计读取怒气/气量等级 (1, 2, 或 3)。
        【已添加调试输出】以帮助校准 HSV 阈值。
        """
        x, y, w, h = rect
        if y + h > raw_img.shape[0] or x + w > raw_img.shape[1] or w <= 0 or h <= 0:
            return 0, (x, y, w, h)

        roi = raw_img[y:y + h, x:x + w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # --- DEBUG 1: 打印 ROI 的平均 HSV 值 ---
        # 观察这些值，然后去修改 config.py 中的 H 范围
        # h_mean, s_mean, v_mean = cv2.mean(hsv, mask=None)[:3]
        # print(f"--- RAGE ROI ({x},{y}) MEAN HSV: H={h_mean:.2f}, S={s_mean:.2f}, V={v_mean:.2f} ---")

        # 定义颜色范围和对应的等级 (从最高等级 3 开始检查)
        color_levels = [
            # 3级 (红色) - 检查两个 H 范围
            ("3R_1", GAUGE_3_COLOR_LOWER_1, GAUGE_3_COLOR_UPPER_1, 3),
            ("3R_2", GAUGE_3_COLOR_LOWER_2, GAUGE_3_COLOR_UPPER_2, 3),

            # 2级 (橙色)
            ("2O", GAUGE_2_COLOR_LOWER, GAUGE_2_COLOR_UPPER, 2),

            # 1级 (绿色)
            ("1G", GAUGE_1_COLOR_LOWER, GAUGE_1_COLOR_UPPER, 1),
        ]

        # 遍历：从 3 级气开始，如果发现足够多的目标颜色像素，则立即返回该等级
        for name, lower, upper, level in color_levels:
            lower_np = np.array(lower)
            upper_np = np.array(upper)

            mask = cv2.inRange(hsv, lower_np, upper_np)
            color_count = cv2.countNonZero(mask)

            # --- DEBUG 2: 打印每个等级的像素计数 ---
            # print(f"  Level {name} Count: {color_count} (Min={GAUGE_MIN_PIXEL_COUNT})")

            # 如果找到足够多的像素（高于阈值），则立即认为这是当前等级
            if color_count >= GAUGE_MIN_PIXEL_COUNT:
                # print(f"*** DETECTED LEVEL {level} by color {name} ***")
                return level, (x, y, w, h)

        # 如果以上所有等级 (1, 2, 3) 都不满足最小像素阈值，返回默认值 0
        return 0, (x, y, w, h)

    def _read_ready_state(self, raw_img, rect):
        """通过颜色统计读取 Ready 状态 (布尔值)"""
        x, y, w, h = rect
        if y + h > raw_img.shape[0] or x + w > raw_img.shape[1] or w <= 0 or h <= 0:
            return False, (x, y, w, h)

        roi = raw_img[y:y + h, x:x + w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        total_pixels = w * h

        # 使用 READY_COLOR 阈值创建掩码
        lower_np = np.array(READY_COLOR_LOWER)
        upper_np = np.array(READY_COLOR_UPPER)
        mask = cv2.inRange(hsv, lower_np, upper_np)

        # 统计匹配目标颜色的像素数
        color_count = cv2.countNonZero(mask)

        # 如果目标颜色像素占比超过阈值，则认为 Ready 状态为 True
        is_ready = (color_count / total_pixels) >= READY_COLOR_MIN_RATIO

        return is_ready, (x, y, w, h)

    def _find_character_box(self, img, lower, upper):
        """利用游戏自带的人物框颜色定位人物"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        # 形态学处理，去除噪声
        mask = cv2.dilate(mask, None, iterations=4)
        mask = cv2.erode(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            # 基础过滤：排除极小的噪声
            if w < 10 or h < 10:
                return (0, 0), (0, 0, 0, 0)

            center = (x + w // 2, y + h // 2)
            return center, (x, y, w, h)

        return (0, 0), (0, 0, 0, 0)

    # 关键修改：替换怒气读取逻辑
    def _process_frame(self, raw_img):
        # 1. 读取血量、时间和怒气

        # HP/Time 使用默认单色 OCR
        p1_val, p1_ratio, p1_rect_ocr = self._read_ocr_value(raw_img, OCR_P1_HP_RECT, max_val=MAX_HP)
        p2_val, p2_ratio, p2_rect_ocr = self._read_ocr_value(raw_img, OCR_P2_HP_RECT, max_val=MAX_HP)
        time_val, _, time_rect_ocr = self._read_ocr_value(raw_img, OCR_TIME_RECT)

        #  怒气值：使用新的颜色统计识别函数
        p1_rage_val, p1_rage_rect_ocr = self._read_gauge_level_by_color(raw_img, OCR_P1_RAGE_RECT)
        p2_rage_val, p2_rage_rect_ocr = self._read_gauge_level_by_color(raw_img, OCR_P2_RAGE_RECT)

        #  Ready 状态读取
        p1_ready_state, p1_ready_rect = self._read_ready_state(raw_img, OCR_P1_READY_RECT)
        p2_ready_state, p2_ready_rect = self._read_ready_state(raw_img, OCR_P2_READY_RECT)

        # 2. 人物定位 (保持不变)
        p1_center, p1_box_rect = self._find_character_box(raw_img, np.array(P1_BOX_LOWER), np.array(P1_BOX_UPPER))
        p2_center, p2_box_rect = self._find_character_box(raw_img, np.array(P2_BOX_LOWER), np.array(P2_BOX_UPPER))

        # 3. 图像预处理 (保持不变)
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        global_view = cv2.resize(gray, (FULL_IMG_SIZE, FULL_IMG_SIZE))

        def crop_local(center):
            cx, cy = center
            x1, y1 = max(0, cx - 50), max(0, cy - 50)
            x2, y2 = min(raw_img.shape[1], cx + 50), min(raw_img.shape[0], cy + 50)
            crop = gray[y1:y2, x1:x2]
            if crop.shape[0] < 10 or crop.shape[1] < 10:
                return np.zeros((LOCAL_IMG_SIZE, LOCAL_IMG_SIZE), dtype=np.uint8)
            return cv2.resize(crop, (LOCAL_IMG_SIZE, LOCAL_IMG_SIZE))

        local_p1 = crop_local(p1_center)
        local_p2 = crop_local(p2_center)

        # 向量数据：5个坐标/距离
        vec = np.array([p1_center[0], p1_center[1], p2_center[0], p2_center[1],
                        np.linalg.norm(np.array(p1_center) - np.array(p2_center))], dtype=np.float32)

        # 4. 可视化绘制 (保持不变)
        vis_img = raw_img.copy()

        # [P1 HP] (绿色)
        cv2.rectangle(vis_img, (p1_rect_ocr[0], p1_rect_ocr[1]),
                      (p1_rect_ocr[0] + p1_rect_ocr[2], p1_rect_ocr[1] + p1_rect_ocr[3]), (0, 255, 0), 2)
        cv2.putText(vis_img, f"P1 HP: {p1_val}", (p1_rect_ocr[0], p1_rect_ocr[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # [P2 HP] (红色)
        cv2.rectangle(vis_img, (p2_rect_ocr[0], p2_rect_ocr[1]),
                      (p2_rect_ocr[0] + p2_rect_ocr[2], p2_rect_ocr[1] + p2_rect_ocr[3]), (0, 0, 255), 2)
        cv2.putText(vis_img, f"P2 HP: {p2_val}", (p2_rect_ocr[0], p2_rect_ocr[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # [Time] (青色)
        cv2.rectangle(vis_img, (time_rect_ocr[0], time_rect_ocr[1]),
                      (time_rect_ocr[0] + time_rect_ocr[2], time_rect_ocr[1] + time_rect_ocr[3]), (255, 255, 0), 2)
        cv2.putText(vis_img, f"Time: {time_val}", (time_rect_ocr[0], time_rect_ocr[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # [P1 Rage] (紫色)
        cv2.rectangle(vis_img, (p1_rage_rect_ocr[0], p1_rage_rect_ocr[1]),
                      (p1_rage_rect_ocr[0] + p1_rage_rect_ocr[2], p1_rage_rect_ocr[1] + p1_rage_rect_ocr[3]),
                      (128, 0, 128), 2)
        cv2.putText(vis_img, f"R1: {p1_rage_val}", (p1_rage_rect_ocr[0], p1_rage_rect_ocr[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 0, 128), 2)

        # [P2 Rage] (橙色)
        cv2.rectangle(vis_img, (p2_rage_rect_ocr[0], p2_rage_rect_ocr[1]),
                      (p2_rage_rect_ocr[0] + p2_rage_rect_ocr[2], p2_rage_rect_ocr[1] + p2_rage_rect_ocr[3]),
                      (0, 165, 255), 2)
        cv2.putText(vis_img, f"R2: {p2_rage_val}", (p2_rage_rect_ocr[0], p2_rage_rect_ocr[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        # [P1 Ready] (蓝色)
        cv2.rectangle(vis_img, (p1_ready_rect[0], p1_ready_rect[1]),
                      (p1_ready_rect[0] + p1_ready_rect[2], p1_ready_rect[1] + p1_ready_rect[3]), (255, 0, 0), 2)
        cv2.putText(vis_img, f"P1 Rdy: {p1_ready_state}", (p1_ready_rect[0], p1_ready_rect[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # [P2 Ready] (浅蓝色)
        cv2.rectangle(vis_img, (p2_ready_rect[0], p2_ready_rect[1]),
                      (p2_ready_rect[0] + p2_ready_rect[2], p2_ready_rect[1] + p2_ready_rect[3]), (255, 255, 0), 2)
        cv2.putText(vis_img, f"P2 Rdy: {p2_ready_state}", (p2_ready_rect[0], p2_ready_rect[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # [P1/P2 人物追踪框]
        cv2.rectangle(vis_img, (p1_box_rect[0], p1_box_rect[1]),
                      (p1_box_rect[0] + p1_box_rect[2], p1_box_rect[1] + p1_box_rect[3]), (255, 0, 0), 2)
        cv2.rectangle(vis_img, (p2_box_rect[0], p2_box_rect[1]),
                      (p2_box_rect[0] + p2_box_rect[2], p2_box_rect[1] + p2_box_rect[3]), (0, 255, 255), 2)

        return {
            "global": global_view, "local_p1": local_p1, "local_p2": local_p2,
            "vec": vec,
            "hp_vals": (p1_val, p2_val),
            "hp_ratios": (p1_ratio, p2_ratio),
            "time": time_val,
            "rage_vals": (p1_rage_val, p2_rage_val),
            "ready_states": (p1_ready_state, p2_ready_state),
            "vis_img": vis_img
        }

    def reset(self):
        self.frames.clear()
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

        r_curr = new_data['hp_ratios'][0] - new_data['hp_ratios'][1]
        r_prev = self.frames[-2]['hp_ratios'][0] - self.frames[-2]['hp_ratios'][1]
        reward = (r_curr - r_prev) * 10

        done = new_data['hp_ratios'][0] <= 0 or new_data['hp_ratios'][1] <= 0 or new_data['time'] <= 0

        cv2.imshow("AI Vision Debug", new_data['vis_img'])
        cv2.waitKey(1)

        return list(self.frames), reward, done, {}