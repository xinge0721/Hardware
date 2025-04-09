#!/bin/bash

# 设置脚本以在出错时停止执行
set -e

echo "===== 开始安装YOLOv8和OpenCV ====="
echo "更新软件包列表"
sudo apt update

echo "安装基本依赖项"
sudo apt install -y python3-pip python3-dev python3-numpy build-essential cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev

echo "安装OpenCV相关依赖项"
sudo apt install -y libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev libv4l-dev

echo "安装ROS依赖项"
sudo apt install -y ros-noetic-cv-bridge python3-opencv python3-empy python3-catkin-pkg python3-rospkg

# 确保ROS环境变量已设置
if [ -z "$ROS_DISTRO" ]; then
    source /opt/ros/noetic/setup.bash
fi

echo "安装更多依赖"
sudo apt install -y ros-noetic-cv-bridge python3-opencv

# 确保empy正确安装
echo "确保empy正确安装"
sudo apt install -y python3-empy
sudo pip3 install em

echo "===== 创建简单配置文件 ====="
# 创建直接使用系统Python的版本，更简单可靠
cat > ~/yolo_setup.sh << 'EOF'
#!/bin/bash

# 激活ROS环境
source /opt/ros/noetic/setup.bash
if [ -d "$HOME/catkin_ws/devel" ]; then
  source $HOME/catkin_ws/devel/setup.bash
fi

# 安装必要的Python包
sudo pip3 install ultralytics opencv-python-headless matplotlib pillow

echo "YOLOv8和OpenCV环境已准备就绪!"
EOF

chmod +x ~/yolo_setup.sh

# 创建测试脚本
mkdir -p ~/yolo_examples

cat > ~/yolo_examples/test_yolo.py << 'EOF'
#!/usr/bin/env python3
import cv2
from ultralytics import YOLO
import time

def main():
    # 加载YOLO模型 (首次运行会自动下载模型)
    print("正在加载YOLOv8模型...")
    model = YOLO('yolov8n.pt')
    print("模型加载完成!")
    
    # 打开摄像头
    print("正在打开摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头，尝试使用图片进行测试...")
        # 使用图片进行测试
        import numpy as np
        # 创建一个测试图像
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Camera not available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 保存测试图像
        cv2.imwrite("test_image.jpg", img)
        
        # 使用测试图像
        results = model("test_image.jpg")
        img_result = results[0].plot()
        cv2.imwrite("result.jpg", img_result)
        print("已将结果保存到 result.jpg")
        return
    
    print("摄像头已打开，按 'q' 键退出...")
    
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧")
            break
        
        # 使用YOLO进行目标检测
        start_time = time.time()
        results = model(frame)
        inference_time = time.time() - start_time
        
        # 可视化结果
        annotated_frame = results[0].plot()
        
        # 添加FPS和性能信息
        fps = 1.0 / inference_time
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"推理时间: {inference_time*1000:.1f}ms", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 显示结果
        cv2.imshow("YOLOv8 检测", annotated_frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("测试完成！")

if __name__ == "__main__":
    main()
EOF

# 创建与ROS集成的简单示例
cat > ~/yolo_examples/yolo_ros_simple.py << 'EOF'
#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from ultralytics import YOLO
from sensor_msgs.msg import Image
from std_msgs.msg import String
import json

def numpy_to_ros_image(img_array, encoding="bgr8"):
    """将numpy数组转换为ROS图像消息"""
    msg = Image()
    msg.height = img_array.shape[0]
    msg.width = img_array.shape[1]
    msg.encoding = encoding
    msg.is_bigendian = 0
    msg.step = img_array.shape[1] * 3
    msg.data = img_array.tobytes()
    return msg

def ros_image_to_numpy(ros_msg):
    """将ROS图像消息转换为numpy数组"""
    if ros_msg.encoding != "bgr8":
        rospy.logwarn(f"不支持的图像编码格式: {ros_msg.encoding}, 尝试按bgr8处理")
    
    # 直接从消息数据中创建numpy数组
    return np.frombuffer(ros_msg.data, dtype=np.uint8).reshape((ros_msg.height, ros_msg.width, 3))

class YoloRosNode:
    def __init__(self):
        rospy.init_node('yolo_detector', anonymous=True)
        
        # 加载YOLO模型
        self.model = YOLO('yolov8n.pt')
        
        # 创建结果发布者
        self.detection_pub = rospy.Publisher('/yolo/detections', String, queue_size=10)
        self.image_pub = rospy.Publisher('/yolo/annotated_image', Image, queue_size=10)
        
        # 订阅图像
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)
        
        rospy.loginfo("YOLOv8检测节点已初始化")
    
    def image_callback(self, msg):
        try:
            # 将ROS图像消息转换为numpy数组
            cv_image = ros_image_to_numpy(msg)
            
            # 使用YOLO进行检测
            results = self.model(cv_image)
            
            # 处理检测结果
            result_data = {"detections": []}
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    b = box.xyxy[0].tolist()  # 边界框
                    c = box.cls.item()        # 类别ID
                    conf = box.conf.item()    # 置信度
                    label = r.names[int(c)]   # 类别名称
                    
                    result_data["detections"].append({
                        "class": label,
                        "confidence": conf,
                        "bbox": b
                    })
            
            # 发布JSON结果
            self.detection_pub.publish(json.dumps(result_data))
            
            # 发布带注释的图像
            annotated_frame = results[0].plot()
            self.image_pub.publish(numpy_to_ros_image(annotated_frame))
            
        except Exception as e:
            rospy.logerr(f"处理图像时发生错误: {e}")

def main():
    try:
        node = YoloRosNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"运行节点时发生错误: {e}")

if __name__ == '__main__':
    main()
EOF

# 设置执行权限
chmod +x ~/yolo_examples/test_yolo.py
chmod +x ~/yolo_examples/yolo_ros_simple.py

echo "===== 创建使用指南 ====="
cat > ~/YOLO_USAGE_GUIDE.md << 'EOF'
# YOLOv8 和 OpenCV 使用指南

## 快速开始

按照以下步骤快速开始使用 YOLOv8:

1. 设置环境 (每次使用前运行):
   ```bash
   source ~/yolo_setup.sh
   ```

2. 测试 YOLOv8 (使用摄像头):
   ```bash
   python3 ~/yolo_examples/test_yolo.py
   ```

3. 与 ROS 集成:
   ```bash
   # 启动 ROS 核心
   roscore

   # 在新终端中运行 YOLOv8 节点
   source ~/yolo_setup.sh
   python3 ~/yolo_examples/yolo_ros_simple.py
   ```

4. 查看检测结果:
   ```bash
   rostopic echo /yolo/detections
   ```

5. 可视化检测结果:
   ```bash
   # 安装图像查看器
   sudo apt install ros-noetic-image-view

   # 查看带注释的图像
   rosrun image_view image_view image:=/yolo/annotated_image
   ```

## 常见问题排查

1. 如果出现模块导入错误:
   ```bash
   sudo pip3 install ultralytics opencv-python-headless
   ```

2. 如果摄像头无法打开:
   - 确保摄像头已连接
   - 尝试修改摄像头索引: 编辑 `~/yolo_examples/test_yolo.py` 中的 `cv2.VideoCapture(0)` 改为 `cv2.VideoCapture(1)` 或其他索引

3. 如果 ROS 节点失败:
   - 确保 ROS 环境已正确设置: `source /opt/ros/noetic/setup.bash`
   - 安装缺失的依赖: `sudo apt install ros-noetic-cv-bridge python3-opencv`

## 高级用法

### 使用不同的 YOLOv8 模型

YOLOv8 提供多种型号的模型，适合不同的设备性能:

- Nano: `model = YOLO('yolov8n.pt')` (最快, 准确度较低)
- Small: `model = YOLO('yolov8s.pt')`
- Medium: `model = YOLO('yolov8m.pt')`
- Large: `model = YOLO('yolov8l.pt')`
- XLarge: `model = YOLO('yolov8x.pt')` (最慢, 准确度最高)

在 Raspberry Pi 上建议使用 Nano 型号以获得最佳性能。
EOF

echo "===== 安装完成 ====="
echo "一切准备就绪! 请按照以下步骤使用 YOLOv8:"
echo "1. 运行 'source ~/yolo_setup.sh' 设置环境"
echo "2. 运行 'python3 ~/yolo_examples/test_yolo.py' 测试 YOLOv8"
echo ""
echo "详细使用说明请查看: ~/YOLO_USAGE_GUIDE.md" 