# YOLOv8 和 OpenCV 使用指南

## 准备工作

### 1. 设置脚本执行权限
首次使用前，确保所有脚本都有可执行权限：
```bash
# 为主安装脚本添加执行权限
chmod +x install_yolo.sh

# 为所有Python脚本和环境脚本添加执行权限
chmod +x yolo_setup.sh
chmod +x yolo_examples/*.py
```

如果在Windows环境下编辑过文件并传输到Linux系统，可能还需要修复行尾问题：
```bash
# 修复可能的Windows行尾问题
sed -i 's/\r$//' install_yolo.sh
sed -i 's/\r$//' yolo_setup.sh
sed -i 's/\r$//' yolo_examples/*.py
```

### 2. 准备测试图片
YOLO测试脚本需要一张名为`1.jpg`的图片放在当前目录下。您可以通过以下方式获取测试图片：

1. **使用摄像头拍摄**（推荐）:
   ```bash
   # 先运行摄像头测试脚本拍摄照片
   source ./yolo_setup.sh
   python3 ./yolo_examples/test_camera.py
   
   # 将拍摄的照片重命名为1.jpg
   mv photo1.jpg 1.jpg
   ```

2. **下载测试图片**:
   ```bash
   # 下载一张网络测试图片
   wget https://ultralytics.com/images/zidane.jpg -O 1.jpg
   # 或
   curl https://ultralytics.com/images/zidane.jpg -o 1.jpg
   ```

3. **手动创建测试图片**：
   - 将您自己的图片复制到当前目录并重命名为`1.jpg`
   - 通过电脑传输图片到树莓派

> **注意**：确保图片格式为`.jpg`，并且图片质量合适（不要太大或太小，建议分辨率在640x480到1920x1080之间）。

## 快速开始

### 1. 一键安装配置
使用以下命令运行一键安装配置脚本：
```bash
./install_yolo.sh
```
这个脚本会自动：
- 安装所有必要的依赖
- 配置Python环境
- 创建所有必要的测试脚本
- 下载YOLOv8模型

> **注意**：如果出现`权限被拒绝`的错误，请确保已经执行了上面的`chmod +x install_yolo.sh`命令。

### 2. 测试摄像头
使用以下命令测试摄像头并拍摄照片：
```bash
source ./yolo_setup.sh
python3 ./yolo_examples/test_camera.py
```
这个脚本会：
- 打开摄像头
- 拍摄两张照片
- 保存为 `photo1.jpg` 和 `photo2.jpg`
- 在有图形界面的情况下显示照片

### 3. 测试YOLOv8目标检测
在运行目标检测前，请确保当前目录下存在一张名为`1.jpg`的测试图片（见上方"准备测试图片"部分）。

使用以下命令测试YOLOv8目标检测：
```bash
source ./yolo_setup.sh
python3 ./yolo_examples/test_yolo.py
```
这个脚本会：
- 检测当前目录下的 `1.jpg` 图片
- 在检测到的对象上绘制边界框
- 保存结果到 `1_result.jpg`

> **错误排查**：如果出现"找不到图像文件"的错误，请确认当前目录下是否有`1.jpg`文件，可使用`ls -la 1.jpg`命令检查。

### 4. 与ROS集成
```bash
# 启动ROS核心
roscore

# 在新终端中运行YOLOv8节点
source ./yolo_setup.sh
python3 ./yolo_examples/yolo_ros_simple.py
```

## 常见问题排查

1. 如果出现`权限被拒绝`错误:
   ```bash
   # 检查文件权限
   ls -la install_yolo.sh yolo_setup.sh yolo_examples/

   # 如果权限不正确，设置可执行权限
   chmod +x install_yolo.sh yolo_setup.sh yolo_examples/*.py
   ```

2. 如果出现模块导入错误:
   ```bash
   sudo pip3 install ultralytics opencv-python-headless
   ```

3. 如果摄像头无法打开:
   - 确保摄像头已连接
   - 检查摄像头权限: `ls -la /dev/video*`
   - 如果需要，添加当前用户到video组: `sudo usermod -a -G video $USER`
   - 尝试修改摄像头索引: 编辑 `./yolo_examples/test_camera.py` 中的 `cv2.VideoCapture(0)` 改为 `cv2.VideoCapture(1)` 或其他索引

4. 如果ROS节点失败:
   - 确保ROS环境已正确设置: `source /opt/ros/noetic/setup.bash`
   - 安装缺失的依赖: `sudo apt install ros-noetic-cv-bridge python3-opencv`

## 文件结构说明

安装完成后，你会得到以下文件结构：
```
|-- install_yolo.sh       # 一键安装脚本
|-- yolo_setup.sh         # 环境设置脚本
|-- YOLO_USAGE_GUIDE.md   # 本使用指南
|-- yolo_examples/        # 示例脚本目录
|   |-- test_camera.py    # 摄像头测试脚本
|   |-- test_yolo.py      # YOLO检测测试脚本
|   `-- yolo_ros_simple.py # ROS集成脚本
```

## 高级用法

### 使用不同的YOLOv8模型

YOLOv8提供多种型号的模型，适合不同的设备性能:

- Nano: `model = YOLO('yolov8n.pt')` (最快, 准确度较低)
- Small: `model = YOLO('yolov8s.pt')`
- Medium: `model = YOLO('yolov8m.pt')`
- Large: `model = YOLO('yolov8l.pt')`
- XLarge: `model = YOLO('yolov8x.pt')` (最慢, 准确度最高)

在Raspberry Pi上建议使用Nano型号以获得最佳性能。

### 修改摄像头参数

在 `test_camera.py` 中，你可以修改以下参数来调整摄像头设置：

```python
# 设置分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 宽度
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # 高度

# 设置帧率
cap.set(cv2.CAP_PROP_FPS, 30)  # 30帧每秒
```

### 修改YOLO检测参数

在 `test_yolo.py` 中，你可以修改以下参数来调整检测设置：

```python
# 设置置信度阈值
model.conf = 0.25  # 默认值

# 设置IOU阈值
model.iou = 0.45  # 默认值

# 设置检测类别
model.classes = [0, 1, 2]  # 只检测特定类别
``` 