import torch

# --- 基础配置 ---
GAME_TITLE = "死神vs火影 竞技版 - 3.8.6"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 训练超参数 ---
NUM_ACTIONS = 19
SEQUENCE_LENGTH = 6       # 输入6帧
FRAME_INTERVAL = 0.1      # 帧间隔 100ms (0.05 - 0.15)
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 5000

# --- 图像处理参数 ---
FULL_IMG_SIZE = 100       # 整体画面缩放尺寸
LOCAL_IMG_SIZE = 20       # 局部画面尺寸

# --- OpenCV 血条识别配置  ---
# 格式: (x, y, w, h) 相对于游戏窗口客户区左上角
HP_BAR_P1_RECT = (100, 60, 260, 20)  # P1 血条区域
HP_BAR_P2_RECT = (400, 60, 260, 20) # P2 血条区域
ENERGY_BAR_RECT = (100, 550, 200, 15) # 能量/怒气区域

# 颜色阈值 (HSV格式) - 用于识别血条有效部分
# 血条是蓝色的 (HUE 范围大约在 90 到 130 之间)
# H (色相): 0-179, S (饱和度): 0-255, V (亮度): 0-255
HP_COLOR_LOWER = (90, 100, 100)
HP_COLOR_UPPER = (130, 255, 255)

# --- 动作映射 ---
ACTION_MAP = {

    # 基础移动与攻击 (0-9)
    0: 'j',  # 普攻 (J)
    1: 'k',  # 跳跃 (K)
    2: 'l',  # 冲刺/闪烁 (L)
    3: 'u',  # 技能 (U)
    4: 'i',  # 技能 (I)
    5: 'o',  # 技能 (O)
    6: 'w',  # 上
    7: 's',  # 下
    8: 'a',  # 左
    9: 'd',  # 右



    # 组合动作 (10-14)
    10: ['d', 'd'],  # 前冲
    11: ['a', 'a'],  # 后退
    12: ['s', 'j'],  # 下蹲+普攻
    13: ['w', 'k'],  # 跳跃+上
    14: ['w', 'j'],  # 跳跃+普攻
    # 必杀技/查克拉 (15-18)
    15: 'space',  # 查克拉 (默认空格)
    16: ['s', 'u'],  # 必杀技1 (需能量)
    17: ['s', 'i'],  # 必杀技2 (需能量)
    18: ['s', 'o'],  # 终极必杀 (高能量)

}
# 定义哪些是需要能量/怒气的动作索引
ULTIMATE_ACTIONS = [17, 18]