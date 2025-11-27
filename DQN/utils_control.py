# utils_control.py
import win32gui
import win32con
import time

# 虚拟键码表 (VK_CODE 保持不变)
VK_CODE = {
    'w': 0x57, 'a': 0x41, 's': 0x53, 'd': 0x44,
    'j': 0x4A, 'k': 0x4B, 'l': 0x4C,
    'u': 0x55, 'i': 0x49, 'o': 0x4F,
    'enter': 0x0D
}


def send_key_event(hwnd, key, event_type):
    """向指定句柄的窗口发送键盘事件。"""
    if key in VK_CODE:
        win32gui.PostMessage(hwnd, event_type, VK_CODE[key], 0)


def execute_model_action(hwnd, action_idx, action_map):
    """
    执行模型预测出的动作索引
    """
    if not hwnd or not win32gui.IsWindow(hwnd):
        return

    keys = action_map.get(action_idx)

    if keys is None:
        return

    if isinstance(keys, str):
        keys_list = [keys]
    else:
        keys_list = keys

    # 按下
    for k in keys_list:
        send_key_event(hwnd, k, win32con.WM_KEYDOWN)

    time.sleep(0.05)  # 模拟按键时长

    # 抬起
    for k in keys_list:
        send_key_event(hwnd, k, win32con.WM_KEYUP)