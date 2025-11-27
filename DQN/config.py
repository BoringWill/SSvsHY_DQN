# config.py

# --- 游戏窗口设置 ---
GAME_WINDOW_TITLE = "死神vs火影 竞技版 - 3.8.6"  # 请修改为您实际的游戏窗口标题
SCREEN_WIDTH = 800     # 假设的游戏分辨率宽
SCREEN_HEIGHT = 600    # 假设的游戏分辨率高

# --- 模型输入参数 ---
GLOBAL_IMG_SIZE = (100, 100)
LOCAL_IMG_SIZE = (20, 20)
NUM_ACTIONS = 19       # 动作数量

# --- YOLO 设置 ---
YOLO_MODEL_PATH = 'yolov8n.pt' # 或者您自己训练好的 'best.pt'
CONF_THRESHOLD = 0.5

# --- 动作映射 (Action Mapping) ---
# 将模型的输出索引 (0-18) 映射到具体的按键操作
# 格式: Index: 'key_name' 或 ['key1', 'key2']
ACTION_MAP = {
    0: None,       # 站立/无操作
    1: 'w',        # 跳
    2: 's',        # 防御/下
    3: 'a',        # 左移
    4: 'd',        # 右移
    5: 'j',        # 普攻
    6: 'k',        # 跳跃攻击/特殊
    7: 'l',        # 瞬步/冲刺
    8: 'u',        # 远攻
    9: 'i',        # 大招
    10: 'o',       # 辅助
    # 组合键示例 (如果您的控制函数支持列表)
    11: ['w', 'j'], # 跳攻
    12: ['s', 'j'],
    13: ['w', 'u'],
    # ... 根据您的需求补充至 18
}