# utils_capture.py
import win32gui
import win32ui
import win32con
import numpy as np
import cv2  # 需要 opencv-python
import time


def set_window_title(hwnd, new_title):
    """
    修改指定句柄 (HWND) 窗口的标题。
    """
    if hwnd and win32gui.IsWindow(hwnd):
        try:
            old_title = win32gui.GetWindowText(hwnd)
            win32gui.SetWindowText(hwnd, new_title)
            print(f"窗口标题成功修改: '{old_title}' -> '{new_title}'")
            return True
        except Exception as e:
            print(f"警告：修改窗口标题失败: {e}")
            return False
    return False


def capture_frame_by_hwnd(hwnd):
    """
    直接使用句柄 (HWND) 截取指定窗口的图像，绕过标题查找。
    返回: BGR 格式的 OpenCV NumPy 数组图像，如果失败则返回 None
    """
    if not hwnd or not win32gui.IsWindow(hwnd):
        return None

    # 1. 获取窗口大小 (使用 GetWindowRect 截取整个窗口区域)
    try:
        left, top, right, bot = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bot - top
    except win32gui.error:
        # HWND 有效但状态异常 (例如最小化)
        return None

    if width <= 0 or height <= 0:
        return None

    # 2. 获取窗口设备上下文
    hwndDC = win32gui.GetWindowDC(hwnd)
    if not hwndDC: return None

    mfcDC = win32ui.CreateDCFromHandle(hwndDC)
    saveDC = mfcDC.CreateCompatibleDC()

    # 3. 创建位图对象
    saveBitMap = win32ui.CreateBitmap()
    saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
    saveDC.SelectObject(saveBitMap)

    # 4. 复制窗口DC的图像到内存DC (使用 PrintWindow 更稳定)
    # 0 代表截取整个窗口，包括非客户区 (标题栏、边框)
    try:
        # 使用 PrintWindow 替代 BitBlt，对现代游戏兼容性更好
        win32gui.PrintWindow(hwnd, saveDC.GetHandle(), 0)
    except Exception as e:
        # 如果 PrintWindow 失败，回退到 BitBlt (如果 PrintWindow 成功，这一行不会执行)
        try:
            saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)
        except win32ui.error:
            # print(f"截图失败: {e}")
            pass

    # 5. 提取位图数据
    bmp_str = saveBitMap.GetBitmapBits(True)
    img = np.frombuffer(bmp_str, dtype='uint8')
    img.shape = (height, width, 4)

    # 6. 释放GDI资源
    win32gui.DeleteObject(saveBitMap.GetHandle())
    saveDC.DeleteDC()
    mfcDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwndDC)

    # 返回 BGR (去掉 Alpha 通道)
    return img[:, :, :3]