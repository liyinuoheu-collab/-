# area_finder.py
# 一个用于帮助您确定追踪物体像素面积的辅助工具

import cv2
import numpy as np

# --- 1. 参数设置 (请在此处修改) ---

# 视频文件路径
video_path = 'video_20250910_095607.mp4'  # <--- !!! 修改为您的视频文件名

# 蓝色小球的HSV阈值 (请确保这里的设置与您主脚本中的一致)
lower_blue = np.array([0, 82, 19])
upper_blue = np.array([24, 255, 255])


# --- 2. 主程序 ---

# 打开视频
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"错误：无法打开视频文件 {video_path}")
    exit()

print("\n--- 像素面积查找工具 ---")
print("用法:")
print("  - 按 [空格键] 前进到下一帧。")
print("  - 按 [Q] 键退出程序。")
print("--------------------------\n")

# 创建显示窗口
cv2.namedWindow("Frame")
cv2.namedWindow("Mask")

while cap.isOpened():
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        print("视频播放完毕或读取下一帧失败。")
        break

    # --- 核心识别逻辑 (与主脚本一致) ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # 寻找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 如果找到了轮廓
    if contours:
        # 找到面积最大的那个轮廓，我们假设它就是小球
        max_contour = max(contours, key=cv2.contourArea)
        
        # 计算这个轮廓的精确面积
        area = cv2.contourArea(max_contour)

        # 在原始画面上把这个轮廓画出来 (绿色，粗细为2)
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)

        # 准备要显示的文字
        area_text = f"Detected Area: {int(area)} pixels"

        # 在左上角显示文字
        cv2.putText(frame, area_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 也在终端打印一次，方便复制
        print(f"当前帧找到目标，像素面积: {int(area)}")

    else:
        # 如果没找到，也在屏幕上提示一下
        cv2.putText(frame, "No object detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # 显示处理后的图像
    cv2.imshow("Frame", frame) # 显示带有面积信息的原始图像
    cv2.imshow("Mask", mask)   # 显示二值化的掩码图像，用于判断颜色设置是否准确

    # 等待按键
    key = cv2.waitKey(0) & 0xFF # waitKey(0) 会暂停，直到有按键按下

    # 按 'q' 退出
    if key == ord('q'):
        print("用户请求退出。")
        break
    # 按空格键继续到下一帧
    elif key == ord(' '):
        continue

# 释放资源
cap.release()
cv2.destroyAllWindows()