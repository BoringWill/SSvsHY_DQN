import torch
import numpy as np

# --- 基础配置 ---
GAME_TITLE = "死神vs火影 竞技版 - 3.8.6"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# !!! Tesseract OCR 引擎路径 (必须修改为您电脑上的实际路径) !!!
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- 训练超参数 ---
NUM_ACTIONS = 20
SEQUENCE_LENGTH = 6
FRAME_INTERVAL = 0.05
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 5000

# --- 图像处理参数 ---
FULL_IMG_SIZE = 100
LOCAL_IMG_SIZE = 20
MAX_HP = 2000.0  # 最大血量是 2000 (用于计算比例)

# --- 怒气值配置 ---
MAX_RAGE = 3  # 最大怒气值为 3
RAGE_COST = 1  # 释放 1 格气技能消耗
SUPER_RAGE_COST = 3  # 释放 3 格气技能消耗

# --- OCR 数字识别区域配置 (x, y, w, h) ---
# HP/Time 区域（用于 OCR 数字）
OCR_P1_HP_RECT = (315, 55, 50, 40)
OCR_P2_HP_RECT = (448, 55, 50, 40)
OCR_TIME_RECT = (380, 50, 60, 50)

# 怒气值区域（用于颜色识别，需要框住数字）
OCR_P1_RAGE_RECT = (245, 590, 15, 10)  # 245
OCR_P2_RAGE_RECT = (552, 590, 15, 10)  # 552

# Ready 单词区域（用于 OCR 单词）
OCR_P1_READY_RECT = (55, 560, 40, 15)  # P1 怒气左边，更宽一些以容纳 "Ready"
OCR_P2_READY_RECT = (720, 560, 40, 15)  # P2 怒气右边
READY_COLOR_LOWER = (10, 150, 150)
READY_COLOR_UPPER = (40, 255, 255)

# 识别目标颜色占区域总像素的最低比例 (用于 Ready 状态判定)
READY_COLOR_MIN_RATIO = 0.03 # 设定一个较低的比例，防止边缘模糊导致识别失败

# OCR 颜色过滤 (白色数字/Ready 单词)
DIGIT_COLOR_LOWER = (0, 0, 200)
DIGIT_COLOR_UPPER = (179, 50, 255)

# --- 怒气值颜色阈值配置  ---
# S_LOWER 和 V_LOWER 保持 80 不变，以捕获光晕

# 1 级: 绿色/霓虹绿 (H: 40-80) - 保持不变
GAUGE_1_COLOR_LOWER = (40, 80, 80)
GAUGE_1_COLOR_UPPER = (80, 255, 255)

# 2 级: 橙色/黄色 (H: 13-40)
GAUGE_2_COLOR_LOWER = (13, 80, 80)
GAUGE_2_COLOR_UPPER = (40, 255, 255)

# 3 级: 红色 (H: 0-12 & 165-180)
GAUGE_3_COLOR_LOWER_1 = (0, 150, 150)  # S/V 提高，只抓核心亮红
GAUGE_3_COLOR_UPPER_1 = (12, 255, 255) # H 上限降低

GAUGE_3_COLOR_LOWER_2 = (165, 150, 150) # S/V 提高
GAUGE_3_COLOR_UPPER_2 = (180, 255, 255)

# 最小像素数量阈值：保持不变
GAUGE_MIN_PIXEL_COUNT = 5


# --- 游戏自带人物框颜色阈值 ---
# --- P1 角色追踪框 (黄色 #FFFF00) 的 HSV 阈值 ---
P1_BOX_LOWER = [25, 180, 180]
P1_BOX_UPPER = [35, 255, 255]

# --- P2 角色追踪框 (蓝色 #0000FF) 的 HSV 阈值 ---
P2_BOX_LOWER = [115, 180, 180]
P2_BOX_UPPER = [125, 255, 255]

# --- 动作映射 (共 20 个独立动作) ---
# 格式: 动作索引: [按下键序列]
ACTION_MAP = {
    # 0-7: 基础移动与防御
    0: ['a'],  # A: 左移动
    1: ['d'],  # D: 右移动
    2: ['s'],  # S: 防御/下蹲
    3: ['k'],  # K: 跳跃
    4: ['l'],  # L: 冲刺
    5: ['j'],  # J: 近攻
    6: ['u'],  # U: 远攻
    7: ['s', 'k'],  # S+K: 从台阶上下来

    # 8-15: 必杀、援助与组合技
    8: ['i'],  # I: 必杀 (1格气)
    9: ['o'],  # O: 召唤援助 (1格气)
    10: ['w', 'j'],  # W+J: 升龙/特殊技
    11: ['s', 'j'],  # S+J: 下段/特殊技
    12: ['w', 'u'],  # W+U: 对空远攻/特殊技
    13: ['s', 'u'],  # S+U: 地面远攻/特殊技

    # 14-17: 怒气消耗动作
    14: ['w', 'i'],  # W+I: 必杀 (1格气)
    15: ['s', 'i'],  # S+I: 超级必杀 (3格气)
    16: ['j', 'k'],  # J+K: 升级 (3格气)

    # 17-19: 特殊动作
    17: ['s', 'l'],  # S+L: 幽步
    18: ['w', 'l'],  # W+L: 幽步
    19: ['l'],  # L: 击倒起身（需要在特定时间调用）
}

# --- 需要消耗怒气的动作索引及其消耗值 ---
ULTIMATE_ACTIONS = {
    8: RAGE_COST,  # I (1格气)
    9: RAGE_COST,  # O (1格气)
    14: RAGE_COST,  # W+I (1格气)
    15: SUPER_RAGE_COST,  # S+I (3格气)
    16: SUPER_RAGE_COST,  # J+K (3格气)
}