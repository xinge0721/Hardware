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