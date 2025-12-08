import time
import win32api, win32con
from config import ACTION_MAP

# 键盘映射：Win32 虚拟键码
KEY_MAP = {
    'a': 0x41, 'd': 0x44, 's': 0x53, 'w': 0x57, 'j': 0x4A,
    'k': 0x4B, 'l': 0x4C, 'u': 0x55, 'i': 0x49, 'o': 0x4F,
    # 确保所有动作键都已映射
}


def press_key(key):
    """按下指定的键."""
    vk_code = KEY_MAP.get(key.lower())
    if vk_code is not None:
        win32api.keybd_event(vk_code, 0, 0, 0)


def release_key(key):
    """释放指定的键."""
    vk_code = KEY_MAP.get(key.lower())
    if vk_code is not None:
        win32api.keybd_event(vk_code, 0, win32con.KEYEVENTF_KEYUP, 0)


def execute_action(action_index):
    """
    根据动作索引执行按键序列。支持同时按下多个键。
    """
    keys_to_press = ACTION_MAP.get(action_index)
    if not keys_to_press:
        return

    # 1. 按下所有键
    for key in keys_to_press:
        press_key(key)

    # 2. 保持按下状态一小段时间 (模拟按键时间)
    time.sleep(0.05)

    # 3. 释放所有键
    for key in keys_to_press:
        release_key(key)

    # 确保组合键释放后，不会影响下一帧的输入
    time.sleep(0.01)