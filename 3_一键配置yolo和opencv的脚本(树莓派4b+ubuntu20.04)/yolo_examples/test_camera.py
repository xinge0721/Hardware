#!/usr/bin/env python3
import cv2
import time
import os

def main():
    # 获取当前目录
    script_dir = os.getcwd()
    
    # 打开摄像头
    print("正在打开摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return
    
    print("摄像头已打开，准备拍摄...")
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 拍摄第一张照片
    print("拍摄第一张照片...")
    ret, frame1 = cap.read()
    if ret:
        # 保存第一张照片
        photo1_path = os.path.join(script_dir, "photo1.jpg")
        cv2.imwrite(photo1_path, frame1)
        print(f"已保存第一张照片到: {photo1_path}")
        
        # 显示第一张照片的信息
        height, width = frame1.shape[:2]
        print(f"照片尺寸: {width}x{height}")
    else:
        print("拍摄第一张照片失败")
    
    # 等待2秒
    print("等待2秒...")
    time.sleep(2)
    
    # 拍摄第二张照片
    print("拍摄第二张照片...")
    ret, frame2 = cap.read()
    if ret:
        # 保存第二张照片
        photo2_path = os.path.join(script_dir, "photo2.jpg")
        cv2.imwrite(photo2_path, frame2)
        print(f"已保存第二张照片到: {photo2_path}")
    else:
        print("拍摄第二张照片失败")
    
    # 释放摄像头
    cap.release()
    print("摄像头已关闭")
    
    # 尝试显示拍摄的照片（如果有图形界面）
    try:
        if 'frame1' in locals():
            cv2.imshow("第一张照片", frame1)
            print("按任意键查看第二张照片...")
            cv2.waitKey(0)
        
        if 'frame2' in locals():
            cv2.imshow("第二张照片", frame2)
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"无法显示图像: {e}")
        print("但照片已保存到文件")

if __name__ == "__main__":
    main()
    print("测试完成!") 