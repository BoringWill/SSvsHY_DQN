import win32gui
import win32ui
import win32con
import numpy as np
import cv2  # 需要 opencv-python
import time


# --- 屏幕捕获函数 (保持不变，已高效) ---

def capture_window(window_title):
    """
    使用 win32 API 截取指定窗口的图像。

    [... 函数内部代码与您提供的一致 ...]
    """
    # 1. 获取窗口句柄
    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd:
        # print(f"错误：未找到标题为 '{window_title}' 的窗口。")
        return None

    # 2. 获取窗口大小
    try:
        left, top, right, bot = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bot - top
    except win32gui.error:
        # print(f"错误：获取窗口 '{window_title}' 矩形失败。")
        return None

    if width <= 0 or height <= 0:
        # print(f"错误：窗口 '{window_title}' 大小异常。")
        return None

    # 3. 获取窗口设备上下文 (Window Device Context)
    hwndDC = win32gui.GetWindowDC(hwnd)
    if not hwndDC:
        # print("错误：获取窗口DC失败。")
        return None

    mfcDC = win32ui.CreateDCFromHandle(hwndDC)

    # 4. 创建内存设备上下文 (Memory Device Context)
    saveDC = mfcDC.CreateCompatibleDC()

    # 5. 创建位图对象
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)

    # 6. 将位图选入内存DC
    saveDC.SelectObject(saveBitMap)

    # 7. 复制窗口DC的图像到内存DC
    # SRCCOPY 表示直接复制源矩形到目标矩形
    try:
        saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
    except win32ui.error as e:
        # print(f"BitBlt 失败: {e}")
        # 释放资源
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        return None

    # 8. 提取位图数据
    bmp_info = saveBitMap.GetInfo()
    bmp_str = saveBitMap.GetBitmapBits(True)

    # 9. 将数据转换为 NumPy 数组
    img = np.frombuffer(bmp_str, dtype='uint8')
    img.shape = (height, width, 4)

    # 10. 释放GDI资源 (非常重要!)
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    # 返回 BGRA 图像
    return img[:, :, :3]  # 截取 BGR 三通道，模型通常不需要 Alpha 通道


# --- 核心逻辑：帧堆叠与时间控制 ---

def capture_and_stack_frames(window_title, num_frames, delay_ms, target_size):
    """
    连续捕获指定数量的帧，并将其堆叠成一个列表。

    :param window_title: 目标窗口标题
    :param num_frames: 需要捕获的帧数 (例如 6)
    :param delay_ms: 帧与帧之间尝试保持的延迟时间 (毫秒)
    :param target_size: 缩放后的图像尺寸 (width, height)
    :return: 包含 6 帧处理后图像的列表，如果失败则返回 None
    """
    delay_sec = delay_ms / 1000.0  # 毫秒转秒
    stacked_frames = []

    for i in range(num_frames):
        start_time_frame = time.perf_counter()

        # 1. 捕获
        frame = capture_window(window_title)

        if frame is None:
            # 捕获失败或窗口未找到
            if i == 0:
                # 第一次捕获失败，返回 None
                return None
            else:
                # 捕获过程中失败，用前一帧的图像填充 (常见做法)
                if stacked_frames:
                    processed_frame = stacked_frames[-1]
                else:
                    return None
        else:
            # 2. 预处理 (缩放)
            processed_frame = cv2.resize(frame, target_size)
            # 强化学习通常还会进行灰度化:
            # processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            # processed_frame = processed_frame.reshape(target_size[1], target_size[0], 1)

        stacked_frames.append(processed_frame)

        # 3. 精确延迟计算和等待
        elapsed_time = time.perf_counter() - start_time_frame
        wait_time = delay_sec - elapsed_time

        if wait_time > 0:
            time.sleep(wait_time)

        # 可选：打印实际帧间隔用于调试
        # print(f"Frame {i+1} 实际耗时: {(time.perf_counter() - start_time_frame)*1000:.2f}ms")

    return stacked_frames


# --- 主程序入口 ---

if __name__ == "__main__":
    # --- 配置项 ---
    game_window_title = "死神vs火影 竞技版 - 3.8.6"  # **请替换成你的游戏窗口标题**
    TARGET_IMAGE_SIZE = (512, 512)
    NUM_FRAMES_TO_STACK = 6
    FRAME_INTERVAL_MS = 100  # 设定间隔 100ms (在 50ms 到 150ms 之间)

    # 模拟AI模型，DQN的输入张量通常是 (H, W, C * N)
    # 如果原始是 (100, 100, 3)，则 DQN 输入应该是 (100, 100, 18)

    print(f"目标：捕获 {NUM_FRAMES_TO_STACK} 帧，每帧间隔 {FRAME_INTERVAL_MS}ms。")
    print(f"等待窗口 '{game_window_title}' ...")

    while True:
        # 捕获 6 帧
        frame_stack = capture_and_stack_frames(
            game_window_title,
            NUM_FRAMES_TO_STACK,
            FRAME_INTERVAL_MS,
            TARGET_IMAGE_SIZE
        )

        if frame_stack is not None and len(frame_stack) == NUM_FRAMES_TO_STACK:

            # ** 强化学习模型的输入准备 **
            # 将 6 帧图像在通道(axis=-1)维度上拼接起来，作为模型的输入
            dqn_input = np.concatenate(frame_stack, axis=-1)

            # --- AI决策和控制（此处仅为演示） ---
            # action = model.predict(dqn_input)
            # execute_action(hwnd, action)
            # -----------------------------------

            # 显示最后一帧（用于调试）
            cv2.imshow("Last Captured Frame (DQN Input)", frame_stack[-1])

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("等待窗口...")
            # 仅在捕获失败时进行较长的等待
            time.sleep(1)

    cv2.destroyAllWindows()