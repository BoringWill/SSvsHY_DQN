## 📄 SSvsHY-BVN: 基于视觉感知的深度强化学习智能体

### 🚀 项目概述

本项目旨在开发一个深度强化学习（DQN）智能体，通过**Win32 API**实时捕获游戏画面，并利用**异构神经网络**（包含卷积网络、空间注意力机制和LSTM）提取特征，以学习如何在《死神 vs 火影 竞技版》中进行对战。

核心特点包括：

1.  **多模态输入**：融合全局视觉特征、局部动作感知特征和相对坐标向量。
2.  **时序处理**：使用 LSTM 处理连续 6 帧的画面序列。
3.  **视觉感知**：利用 **OpenCV** 实时提取血条状态和人物位置，无需依赖内存读取。
4.  **动作屏蔽 (Action Masking)**：通过外部状态信息过滤无效动作，避免污染经验记忆库。
5.  **Gym 接口封装**：环境封装符合 OpenAI Gym 标准，便于训练和扩展。

-----

### 🛠️ 环境配置

本项目的运行环境基于 Windows 操作系统，并依赖特定的 Python 库。

#### 1\. 软件依赖

| 软件 | 说明 |
| :--- | :--- |
| **Python 3.8+** | 推荐使用 Anaconda 或 Miniconda 创建独立环境。 |
| **Tesseract OCR Engine** | (可选，如果需要使用OCR提取能量/怒气) 需单独安装并配置路径。 |
| **游戏** | 《死神 vs 火影 竞技版》 (需确保窗口标题与 `config.py` 中的 `GAME_TITLE` 一致)。 |

#### 2\. Python 依赖安装

建议使用项目根目录下的 `requirements.txt` 进行安装。

```bash
# 1. 创建并激活虚拟环境
python -m venv venv_bvn
.\venv_bvn\Scripts\activate

# 2. 安装基础依赖
pip install -r requirements.txt

# 3. 安装 PyTorch (根据您的 CUDA/CPU 配置单独安装)
# 示例 (CPU版本):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

-----

### 📂 代码结构概览

| 文件/目录 | 描述 |
| :--- | :--- |
| `config.py` | **核心配置文件。** 包含超参数、游戏窗口标题、图像裁剪尺寸、**OpenCV 颜色阈值和坐标**等，运行前必须根据您的游戏画面调整。 |
| `environment.py` | **Gym 环境封装。** 实现 `win32` 截图、`OpenCV` 血条读取和人物定位，并提供 `render()` 方法进行可视化。 |
| `model.py` | **神经网络定义。** 实现 2层CNN、空间注意力 (`SpatialAttention`)、全连接层和 LSTM 的复杂异构结构。 |
| `agent.py` | **智能体逻辑。** 包含经验回放逻辑和关键的 **`select_action`** 方法，实现动作屏蔽和 Q-值选择。 |
| `game_actions.py` | 键盘操作模块，负责将动作索引转换为实际的键盘输入 (使用 `keyboard` 库)。 |
| `main.py` | **项目运行入口。** 默认配置为**环境测试模式**，用于验证配置和数据流是否正常。 |
| `checkpoints/` | 用于存放训练好的模型权重。 |

-----

### ▶️ 运行与调试指南

本项目默认运行在**环境测试模式**，用于验证所有组件（截图、CV 识别、可视化）是否正常工作。

#### 1\. 准备工作

  * 确保游戏窗口已打开，并且窗口标题与 `config.py` 中的 `GAME_TITLE` 严格匹配。
  * 根据您的游戏分辨率，**在 `config.py` 中精确设置** `HP_BAR_RECT` 和人物颜色追踪的 `HSV` 阈值。

#### 2\. 环境测试 (验证配置)

运行 `main.py` 将启动环境测试循环。

```bash
python main.py
```

**预期结果：**

1.  命令行会打印环境初始化信息和每一步的血量、奖励和 Mask 状态。
2.  将弹出一个 `OpenCV` 窗口，实时显示游戏画面，并在画面上：
      * **血条区域**被绿色/红色矩形框标记。
      * **人物位置**被黄色/蓝色矩形框包围。
      * 这验证了 `win32` 截图和 `OpenCV` 识别功能正常。

#### 3\. 切换到训练模式

若环境测试成功，您可以修改 `main.py` 中的循环逻辑，启用 DQN 训练和模型优化部分 (目前在 `agent.py` 中被省略，但留有接口)。

```python
# main.py (在实际训练时)
# ...
# agent.store_transition(...) 
# agent.learn() # <--- 启用模型优化/训练步骤
# ...
```