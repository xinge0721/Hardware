#!/bin/bash

echo "===== YOLO和OpenCV一键安装脚本 ====="

# 激活ROS环境
echo "正在加载ROS环境..."
source /opt/ros/noetic/setup.bash
if [ -d "$HOME/catkin_ws/devel" ]; then
  source $HOME/catkin_ws/devel/setup.bash
fi

# 确保基本依赖
echo "正在安装基础系统依赖..."
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev python3-opencv

# 先卸载可能冲突的包
echo "卸载可能存在冲突的旧版本包..."
sudo pip3 uninstall -y matplotlib pillow ultralytics opencv-python-headless opencv-python || true

# 安装必要的Python包
echo "正在安装和更新必要的Python包..."
# 使用--no-deps避免拉入所有依赖项
sudo pip3 install --upgrade pip
sudo pip3 install --upgrade --no-deps --force-reinstall numpy

# 安装matplotlib和pillow的兼容版本
echo "安装与ultralytics兼容的库..."
sudo pip3 install --upgrade matplotlib==3.7.1 pillow==9.5.0 || sudo pip3 install --upgrade matplotlib>=3.3.0 pillow>=7.1.2

# 安装其他依赖
echo "安装其他依赖..."
sudo pip3 install --upgrade tqdm pyyaml scipy

# 安装torch (不带CUDA，适用于树莓派)
echo "安装PyTorch (CPU版)..."
sudo pip3 install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu || echo "PyTorch安装失败，但可以继续"

# 安装ultralytics
echo "安装YOLOv8 (ultralytics)..."
sudo pip3 install --upgrade ultralytics || sudo pip3 install --upgrade ultralytics==8.0.0

# 安装OpenCV
echo "安装OpenCV..."
sudo pip3 install --upgrade opencv-python-headless

# 创建兼容性脚本
echo "设置环境变量以增加兼容性..."
cat > ~/.yolo_compat.py << 'EOF'
# YOLOv8 兼容性辅助脚本
import os
os.environ['ULTRALYTICS_OLD_MATPLOTLIB'] = '1'  # 尝试兼容旧版matplotlib
try:
    from ultralytics import YOLO
    print("YOLOv8已成功加载")
except Exception as e:
    print(f"加载YOLOv8时出错: {e}")
    print("尝试进行兼容性修复...")
    os.environ['PYTHONPATH'] = os.environ.get('PYTHONPATH', '') + ':/usr/local/lib/python3.8/dist-packages'
    try:
        from ultralytics import YOLO
        print("修复成功，YOLOv8已加载")
    except Exception as e:
        print(f"修复失败: {e}")
EOF

# 验证安装
echo "验证安装..."
python3 -c "
import os
os.environ['ULTRALYTICS_OLD_MATPLOTLIB'] = '1'
try:
    import numpy
    print(f'✓ NumPy已安装: {numpy.__version__}')
except:
    print('✗ NumPy安装失败')

try:
    import cv2
    print(f'✓ OpenCV已安装: {cv2.__version__}')
except:
    print('✗ OpenCV安装失败')

try:
    from ultralytics import YOLO
    print(f'✓ YOLOv8已安装')
except Exception as e:
    print(f'✗ YOLOv8安装失败: {e}')
"

echo ""
echo "===== 一键安装完成! ====="
echo "如果存在库版本不兼容的问题，已经采取了最大兼容性措施"
echo "使用方法:"
echo "- 检测图像: python3 ./yolo_examples/test_yolo.py"
echo "- 与ROS集成: python3 ./yolo_examples/yolo_ros_simple.py"
echo "" 