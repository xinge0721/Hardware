#!/bin/bash
set -e  # 遇到错误立即停止

# 打印彩色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # 恢复默认颜色

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}       YOLOv8 和 OpenCV 一键安装脚本       ${NC}"
echo -e "${GREEN}============================================${NC}"

# 检查是否为root权限，如果不是则使用sudo
SUDO=''
if [ "$EUID" -ne 0 ]; then
  SUDO='sudo'
  echo -e "${YELLOW}使用sudo执行某些命令...${NC}"
fi

# 检查并更新软件包
echo -e "\n${GREEN}[1/7] 检查并更新软件包列表...${NC}"
$SUDO apt-get update

# 安装系统级依赖
echo -e "\n${GREEN}[2/7] 安装系统级依赖...${NC}"
$SUDO apt-get install -y python3-pip python3-dev python3-opencv

# 安装Python环境
echo -e "\n${GREEN}[3/7] 安装Python依赖...${NC}"
# 优先使用--no-deps避免拉入可能冲突的依赖
$SUDO pip3 install --upgrade pip

echo -e "${YELLOW}安装基础Python包...${NC}"
# 安装matplotlib、pillow和numpy
$SUDO pip3 install --upgrade --no-deps matplotlib==3.7.1 pillow==9.5.0 numpy || \
$SUDO pip3 install --upgrade matplotlib>=3.3.0 pillow>=7.1.2 numpy

echo -e "${YELLOW}安装PyTorch (CPU版本)...${NC}"
$SUDO pip3 install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu || \
  echo -e "${YELLOW}PyTorch安装可能有问题，但继续安装其他组件...${NC}"

echo -e "${YELLOW}安装YOLOv8...${NC}"
$SUDO pip3 install --upgrade ultralytics || $SUDO pip3 install --upgrade ultralytics==8.0.0

echo -e "${YELLOW}安装OpenCV...${NC}"
$SUDO pip3 install --upgrade opencv-python-headless

# 创建目录结构
echo -e "\n${GREEN}[4/7] 创建项目目录...${NC}"
mkdir -p yolo_examples

# 创建YOLO设置脚本
echo -e "\n${GREEN}[5/7] 创建环境设置脚本...${NC}"
cat > yolo_setup.sh << 'EOF'
#!/bin/bash

# 激活ROS环境
source /opt/ros/noetic/setup.bash
if [ -d "$HOME/catkin_ws/devel" ]; then
  source $HOME/catkin_ws/devel/setup.bash
fi

# 设置PYTHONPATH以确保模块可见性
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.8/dist-packages

# 输出环境信息
echo "YOLOv8和OpenCV环境已准备就绪!"
echo "PYTHONPATH=$PYTHONPATH"
EOF
chmod +x yolo_setup.sh

# 创建YOLO测试脚本
echo -e "\n${GREEN}[6/7] 创建YOLO测试脚本...${NC}"
cat > yolo_examples/test_yolo.py << 'EOF'
#!/usr/bin/env python3
import os
# 设置环境变量以增加与旧版matplotlib的兼容性
os.environ['ULTRALYTICS_OLD_MATPLOTLIB'] = '1'

import sys
import time

# 尝试加载必要模块，如果失败则提供友好提示
try:
    import cv2
except ImportError:
    print("错误: 无法导入OpenCV (cv2)")
    print("请尝试运行: sudo pip3 install opencv-python-headless")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("错误: 无法导入NumPy")
    print("请尝试运行: sudo pip3 install numpy")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("错误: 无法导入YOLOv8 (ultralytics)")
    print("请尝试运行: sudo pip3 install ultralytics==8.0.0")
    
    # 尝试修复路径问题
    print("尝试修复PYTHONPATH...")
    if '/usr/local/lib/python3.8/dist-packages' not in sys.path:
        sys.path.append('/usr/local/lib/python3.8/dist-packages')
    
    try:
        from ultralytics import YOLO
        print("修复成功！已加载YOLOv8")
    except ImportError:
        print("修复失败，请检查安装")
        sys.exit(1)

def main():
    # 加载YOLO模型 (首次运行会自动下载模型)
    print("正在加载YOLOv8模型...")
    try:
        model = YOLO('yolov8n.pt')
        print("模型加载完成!")
    except Exception as e:
        print(f"模型加载失败: {e}")
        try:
            # 如果普通加载失败，尝试使用本地文件
            if os.path.exists('yolov8n.pt'):
                print("尝试加载本地模型文件...")
                model = YOLO('yolov8n.pt', task='detect')
            else:
                print("尝试使用最简单的模式加载...")
                model = YOLO('yolov8n.yaml')  # 尝试从配置文件加载
        except Exception as e:
            print(f"所有尝试均失败: {e}")
            return
    
    # 打开摄像头
    print("正在打开摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头，尝试使用图片进行测试...")
        # 使用图片进行测试
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_img, "Camera not available", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 保存测试图像
        test_img_path = "test_image.jpg"
        cv2.imwrite(test_img_path, test_img)
        
        # 使用测试图像
        try:
            results = model(test_img_path)
            img_result = results[0].plot()
            cv2.imwrite("result.jpg", img_result)
            print("已将结果保存到 result.jpg")
        except Exception as e:
            print(f"处理图像时出错: {e}")
        return
    
    print("摄像头已打开，按 'q' 键退出...")
    
    # 失败次数计数
    failures = 0
    max_failures = 5
    
    while failures < max_failures:
        try:
            # 读取一帧
            ret, frame = cap.read()
            if not ret:
                print("无法获取帧")
                failures += 1
                time.sleep(0.5)
                continue
            
            # 重置失败计数
            failures = 0
            
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
                
        except Exception as e:
            print(f"处理帧时出错: {e}")
            failures += 1
            time.sleep(0.5)
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("测试完成！")

if __name__ == "__main__":
    main()
EOF
chmod +x yolo_examples/test_yolo.py

# 创建ROSyolo示例
cat > yolo_examples/yolo_ros_simple.py << 'EOF'
#!/usr/bin/env python3
import os
# 设置环境变量以增加与旧版matplotlib的兼容性
os.environ['ULTRALYTICS_OLD_MATPLOTLIB'] = '1'

import sys
import time

# 尝试导入必要的模块，提供错误处理
try:
    import rospy
except ImportError:
    print("错误: 无法导入ROS Python客户端 (rospy)")
    print("请确保ROS已安装并且环境已设置: source /opt/ros/noetic/setup.bash")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("错误: 无法导入OpenCV或NumPy")
    print("请尝试运行: sudo pip3 install opencv-python-headless numpy")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("错误: 无法导入YOLOv8 (ultralytics)")
    print("请尝试运行: sudo pip3 install ultralytics==8.0.0")
    try:
        # 尝试修复PYTHONPATH
        if '/usr/local/lib/python3.8/dist-packages' not in sys.path:
            sys.path.append('/usr/local/lib/python3.8/dist-packages')
        from ultralytics import YOLO
        print("修复成功！")
    except ImportError:
        print("修复失败，请检查安装")
        sys.exit(1)

try:
    from sensor_msgs.msg import Image
    from std_msgs.msg import String
except ImportError:
    print("错误: 无法导入ROS消息类型")
    print("请安装: sudo apt install ros-noetic-std-msgs ros-noetic-sensor-msgs")
    sys.exit(1)

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
    
    try:
        # 直接从消息数据中创建numpy数组
        return np.frombuffer(ros_msg.data, dtype=np.uint8).reshape((ros_msg.height, ros_msg.width, 3))
    except Exception as e:
        rospy.logerr(f"转换图像格式失败: {e}")
        return None

class YoloRosNode:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('yolo_detector', anonymous=True)
        
        try:
            # 加载YOLO模型
            rospy.loginfo("正在加载YOLOv8模型...")
            self.model = YOLO('yolov8n.pt')
            rospy.loginfo("YOLOv8模型加载完成!")
        except Exception as e:
            rospy.logerr(f"加载YOLOv8模型失败: {e}")
            try:
                # 尝试替代加载方法
                if os.path.exists('yolov8n.pt'):
                    rospy.loginfo("尝试直接加载本地模型文件...")
                    self.model = YOLO('yolov8n.pt', task='detect')
                else:
                    rospy.loginfo("模型文件不存在，尝试从配置加载...")
                    self.model = YOLO('yolov8n.yaml')
            except Exception as e:
                rospy.logerr(f"所有加载尝试均失败: {e}")
                raise RuntimeError("无法初始化YOLOv8，请检查安装")
        
        # 创建ROS发布者和订阅者
        self.detection_pub = rospy.Publisher('/yolo/detections', String, queue_size=10)
        self.image_pub = rospy.Publisher('/yolo/annotated_image', Image, queue_size=10)
        
        # 延迟初始化订阅者，确保模型已加载
        rospy.loginfo("等待图像话题...")
        rospy.sleep(1) # 给系统一些时间
        
        # 创建订阅者
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback, queue_size=1)
        
        # 记录初始化状态
        self.initialized = True
        rospy.loginfo("YOLOv8检测节点已初始化并正在运行")
        
        # 添加定时检查
        self.timer = rospy.Timer(rospy.Duration(10), self.health_check)
    
    def health_check(self, event):
        """定期检查节点状态"""
        if self.initialized:
            rospy.loginfo("YOLO节点正常运行中")
    
    def image_callback(self, msg):
        try:
            start_time = time.time()
            
            # 转换图像格式
            cv_image = ros_image_to_numpy(msg)
            if cv_image is None:
                rospy.logerr("无法转换ROS图像消息")
                return
            
            # 使用YOLO进行检测
            results = self.model(cv_image)
            
            # 处理检测结果
            result_data = {"detections": []}
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    try:
                        b = box.xyxy[0].tolist()  # 边界框
                        c = box.cls.item()        # 类别ID
                        conf = box.conf.item()    # 置信度
                        label = r.names[int(c)]   # 类别名称
                        
                        result_data["detections"].append({
                            "class": label,
                            "confidence": float(conf),  # 确保可JSON序列化
                            "bbox": [float(x) for x in b]  # 确保可JSON序列化
                        })
                    except Exception as e:
                        rospy.logwarn(f"处理单个检测结果时出错: {e}")
            
            # 计算处理时间
            process_time = time.time() - start_time
            result_data["process_time_ms"] = process_time * 1000
            
            # 发布JSON结果
            try:
                self.detection_pub.publish(json.dumps(result_data))
            except Exception as e:
                rospy.logerr(f"发布检测结果时出错: {e}")
            
            # 发布带注释的图像
            try:
                annotated_frame = results[0].plot()
                # 添加性能信息
                fps = 1.0 / process_time
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.image_pub.publish(numpy_to_ros_image(annotated_frame))
            except Exception as e:
                rospy.logerr(f"发布带注释图像时出错: {e}")
            
        except Exception as e:
            rospy.logerr(f"处理图像时发生错误: {e}")

def main():
    try:
        rospy.loginfo("正在启动YOLOv8 ROS节点...")
        node = YoloRosNode()
        
        # 设置关闭处理
        def shutdown_hook():
            rospy.loginfo("关闭YOLOv8节点...")
        
        rospy.on_shutdown(shutdown_hook)
        
        # 启动节点
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS中断异常，节点停止")
    except Exception as e:
        rospy.logerr(f"运行节点时发生错误: {e}")
        print(f"发生错误: {e}")
        print("请检查ROS环境和依赖库安装")

if __name__ == '__main__':
    main()
EOF
chmod +x yolo_examples/yolo_ros_simple.py

# 创建使用指南
echo -e "\n${GREEN}[7/7] 创建使用指南...${NC}"
cat > YOLO_USAGE_GUIDE.md << 'EOF'
# YOLOv8 和 OpenCV 使用指南

## 快速开始

按照以下步骤快速开始使用 YOLOv8:

1. 设置环境 (每次使用前运行):
   ```bash
   source ./yolo_setup.sh
   ```

2. 测试 YOLOv8 (使用摄像头):
   ```bash
   python3 ./yolo_examples/test_yolo.py
   ```

3. 与 ROS 集成:
   ```bash
   # 启动 ROS 核心
   roscore

   # 在新终端中运行 YOLOv8 节点
   source ./yolo_setup.sh
   python3 ./yolo_examples/yolo_ros_simple.py
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
   - 尝试修改摄像头索引: 编辑 `./yolo_examples/test_yolo.py` 中的 `cv2.VideoCapture(0)` 改为 `cv2.VideoCapture(1)` 或其他索引

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

# 验证安装
echo -e "\n${GREEN}验证安装...${NC}"
source ./yolo_setup.sh
echo -e "\n${GREEN}尝试导入YOLOv8...${NC}"
python3 -c "
import os
os.environ['ULTRALYTICS_OLD_MATPLOTLIB'] = '1'
try:
    from ultralytics import YOLO
    print('✅ YOLOv8安装成功')
except Exception as e:
    print('❌ YOLOv8导入失败:', e)
"

# 完成
echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}       YOLOv8 和 OpenCV 安装完成!           ${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "使用指南请参考: ${YELLOW}YOLO_USAGE_GUIDE.md${NC}"
echo -e "运行以下命令开始使用:"
echo -e "${YELLOW}source ./yolo_setup.sh${NC}"
echo -e "${YELLOW}python3 ./yolo_examples/test_yolo.py${NC}"
echo -e "\n" 