# video_hsv_tuner.py (Fixed and Improved with Resizable Windows)
import cv2
import numpy as np

# --- 1. 参数设置 ---
# !!! 请将这里替换为您的视频文件名 !!!
video_path = '1.8v-400mv1.7hz.mp4'

# 【新功能】在这里设置预览窗口的最大宽度（单位：像素），程序将自动按比例缩放。
# 您可以根据您屏幕的大小，自由地修改这个数值。
PREVIEW_MAX_WIDTH = 800
# ---------------------


def nothing(x):
    """一个空的回调函数，是 cv2.createTrackbar 语法所必需的。"""
    pass


# --- 2. 主程序 ---

# 读取视频的第一帧用于调试
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"错误：无法打开视频文件 '{video_path}'")
    exit()

ret, frame = cap.read()
if not ret:
    print("错误：无法从视频中读取帧。请检查文件是否损坏或路径是否正确。")
    cap.release()
    exit()

# 读取完第一帧后即可释放，因为我们只在静态图像上调参
cap.release()

# 【新】根据设定的最大宽度，计算最终要显示的窗口尺寸
original_height, original_width = frame.shape[:2]
if original_width > PREVIEW_MAX_WIDTH:
    # 计算缩放比例
    ratio = PREVIEW_MAX_WIDTH / original_width
    # 计算新的高度
    new_height = int(original_height * ratio)
    display_size = (PREVIEW_MAX_WIDTH, new_height)
    print(f"视频原始尺寸过大 ({original_width}x{original_height})，预览窗口已按比例缩放至 {display_size[0]}x{display_size[1]}。")
else:
    # 如果视频本身不大，则按原尺寸显示
    display_size = (original_width, original_height)

# 创建一个窗口和6个用于调节HSV的滑动条
cv2.namedWindow('HSV Tuner for Video')
cv2.createTrackbar('H Lower', 'HSV Tuner for Video', 0, 179, nothing)
cv2.createTrackbar('S Lower', 'HSV Tuner for Video', 50, 255, nothing)
cv2.createTrackbar('V Lower', 'HSV Tuner for Video', 50, 255, nothing)
cv2.createTrackbar('H Upper', 'HSV Tuner for Video', 179, 179, nothing)
cv2.createTrackbar('S Upper', 'HSV Tuner for Video', 255, 255, nothing)
cv2.createTrackbar('V Upper', 'HSV Tuner for Video', 255, 255, nothing)

print("\n--- 视频专用HSV调参器 ---")
print(f"已加载视频 '{video_path}' 的第一帧。")
print("1. 调整6个滑块，让'Mask'窗口中只有目标物体是纯白色，其他所有背景都是纯黑色。")
print("2. 调整完毕后，记下6个滑块的数值。")
print("3. 按键盘上的 'q' 键退出程序。\n")

while True:
    # 将第一帧的颜色空间从 BGR 转换为 HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 从滑动条实时获取当前的HSV阈值
    h_lower = cv2.getTrackbarPos('H Lower', 'HSV Tuner for Video')
    s_lower = cv2.getTrackbarPos('S Lower', 'HSV Tuner for Video')
    v_lower = cv2.getTrackbarPos('V Lower', 'HSV Tuner for Video')
    h_upper = cv2.getTrackbarPos('H Upper', 'HSV Tuner for Video')
    s_upper = cv2.getTrackbarPos('S Upper', 'HSV Tuner for Video')
    v_upper = cv2.getTrackbarPos('V Upper', 'HSV Tuner for Video')

    # 将获取到的值组合成Numpy数组
    lower_bound = np.array([h_lower, s_lower, v_lower])
    upper_bound = np.array([h_upper, s_upper, v_upper])

    # 根据上下限创建掩膜（Mask）
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # 【修改后】在显示前，将原始帧和Mask都缩放到计算好的尺寸
    display_frame = cv2.resize(frame, display_size)
    display_mask = cv2.resize(mask, display_size)

    # 显示缩放后的图像
    cv2.imshow('First Frame of Video', display_frame)
    cv2.imshow('Mask', display_mask)

    # 检测按键，如果按下 'q' 就退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("程序退出。")
        break

# 销毁所有创建的窗口
cv2.destroyAllWindows()