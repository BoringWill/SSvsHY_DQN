import keyboard
import time
from config import ACTION_MAP

def execute_action(action_idx):
    """根据索引执行动作"""
    keys = ACTION_MAP.get(action_idx)
    if keys is None: return

    # 简单模拟按键
    if isinstance(keys, str):
        keyboard.press(keys)
        time.sleep(0.02)
        keyboard.release(keys)
    elif isinstance(keys, list):
        for k in keys: keyboard.press(k)
        time.sleep(0.05)
        for k in keys: keyboard.release(k)