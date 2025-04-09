#!/usr/bin/env python3
import os
# 设置环境变量以增加与旧版matplotlib的兼容性
os.environ['ULTRALYTICS_OLD_MATPLOTLIB'] = '1'

import sys
import time
import cv2
import numpy as np

# 尝试导入YOLO
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
    # 获取图像路径 - 与脚本同目录下的1.jpg
    script_dir = os.getcwd()
    image_path = os.path.join(script_dir, "1.jpg")
    
    # 检查文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 找不到图像文件 {image_path}")
        print("请确保在当前目录下有名为1.jpg的图像文件")
        return
    
    print(f"找到图像文件: {image_path}")
    
    # 加载YOLO模型
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
    
    # 读取图像
    print("正在读取图像...")
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return
        
        # 获取图像尺寸
        height, width = image.shape[:2]
        print(f"图像尺寸: {width}x{height}")
    except Exception as e:
        print(f"读取图像时出错: {e}")
        return
    
    # 使用YOLO进行检测
    print("正在进行对象检测...")
    start_time = time.time()
    try:
        results = model(image)
        inference_time = time.time() - start_time
        print(f"检测完成！耗时: {inference_time*1000:.1f}ms")
    except Exception as e:
        print(f"对象检测时出错: {e}")
        return
    
    # 获取检测结果数据
    detection_count = 0
    try:
        for r in results:
            boxes = r.boxes
            detection_count = len(boxes)
            for box in boxes:
                b = box.xyxy[0].tolist()  # 边界框
                c = int(box.cls.item())   # 类别
                conf = box.conf.item()    # 置信度
                label = r.names[c]        # 类别名称
                print(f"检测到: {label}, 置信度: {conf:.2f}, 位置: {b}")
    except Exception as e:
        print(f"处理检测结果时出错: {e}")
    
    # 可视化结果
    try:
        annotated_image = results[0].plot()
        
        # 添加性能信息
        fps = 1.0 / inference_time
        cv2.putText(annotated_image, f"FPS: {fps:.1f}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_image, f"检测到 {detection_count} 个对象", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 保存带注释的图像
        result_path = os.path.join(script_dir, "1_result.jpg")
        cv2.imwrite(result_path, annotated_image)
        print(f"已保存检测结果到: {result_path}")
        
        # 尝试显示结果（可能需要图形界面）
        try:
            cv2.imshow("YOLOv8 检测结果", annotated_image)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"无法显示图像: {e}")
            print("但结果已保存到文件")
    except Exception as e:
        print(f"生成结果图像时出错: {e}")

if __name__ == "__main__":
    main()
    print("处理完成!") 