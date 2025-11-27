# perception.py (修改版)
import cv2
import numpy as np
from ultralytics import YOLO
# 如果不想安装 filterpy，我们可以用一个简化的线性追踪类
from filterpy.kalman import KalmanFilter


class ObjectTracker:
    """简单的卡尔曼滤波包装器，用于追踪 (x, y, w, h)"""

    def __init__(self):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        # 状态向量 [x, y, w, h, vx, vy, vw, vh]
        self.kf.F = np.eye(8)  # 状态转移矩阵
        for i in range(4):
            self.kf.F[i, i + 4] = 1  # 简单的匀速模型
        self.kf.H = np.eye(4, 8)  # 测量矩阵
        self.kf.P *= 1000.
        self.kf.R *= 10
        self.kf.Q *= 0.01

    def update(self, box):
        """box: [x, y, w, h]"""
        self.kf.predict()
        self.kf.update(box)
        return self.kf.x[:4].flatten()  # 返回预测后的 x, y, w, h

    def predict_only(self):
        """当YOLO没检测到时，使用预测值"""
        self.kf.predict()
        return self.kf.x[:4].flatten()


class GamePerception:
    def __init__(self):
        self.yolo = YOLO('yolov8n.pt')
        # 为敌我双方分别建立追踪器
        self.tracker_p1 = ObjectTracker()
        self.tracker_p2 = ObjectTracker()
        # 记录上一帧是否存在，用于判断是否使用预测
        self.p1_seen = False
        self.p2_seen = False

    def process(self, frame):
        # 1. YOLO 检测
        results = self.yolo(frame, verbose=False)[0]
        boxes = results.boxes.data.cpu().numpy()

        # 提取原始测量值
        meas_p1 = None
        meas_p2 = None

        for box in boxes:
            if box[4] < 0.5: continue
            x1, y1, x2, y2 = box[:4]
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2, y1 + h / 2
            cls_id = int(box[5])

            # 这里假设 class 0 是我方，class 1 是敌方
            if cls_id == 0 and meas_p1 is None:
                meas_p1 = np.array([cx, cy, w, h])
            elif cls_id == 1 and meas_p2 is None:
                meas_p2 = np.array([cx, cy, w, h])

        # 2. 卡尔曼滤波更新
        if meas_p1 is not None:
            final_p1 = self.tracker_p1.update(meas_p1)
            self.p1_seen = True
        else:
            final_p1 = self.tracker_p1.predict_only()  # 漏检时用预测

        if meas_p2 is not None:
            final_p2 = self.tracker_p2.update(meas_p2)
            self.p2_seen = True
        else:
            final_p2 = self.tracker_p2.predict_only()

        # 3. 转换回坐标框 [x1, y1, x2, y2] 用于截图
        def to_box(state):
            cx, cy, w, h = state
            return [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)]

        box_p1 = to_box(final_p1)
        box_p2 = to_box(final_p2)

        # 4. 生成图像和标量 (复用之前的逻辑，略微调整)
        # 全局图 100x100
        global_view = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (100, 100)) / 255.0

        # 局部图裁剪 (20x20)
        local_p1 = self._crop(frame, box_p1)
        local_p2 = self._crop(frame, box_p2)

        # 标量信息 (归一化坐标 + 距离)
        # ... (此处代码参考之前给出的 _calc_scalars) ...

        # 为了演示简洁，这里直接返回处理好的数据
        # 实际代码请把之前写的 scalar 计算逻辑放进来
        scalars = np.array([0] * 6)

        return global_view, local_p1, local_p2, scalars

    def _crop(self, frame, box):
        # ... (参考之前的 _crop_resize 代码) ...
        return np.zeros((20, 20))  # 占位