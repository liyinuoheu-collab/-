# multi_object_tracker_v41_combined_plot_gem.py
# 版本: V41 合并图表版 (由嵌入-式专家Gem整合)
# 特性:
#   1. 此版本将速度与加速度的图表重新合并为一张双Y轴图，便于观察对应关系。
#   2. 保留了V41版本完整的“双重物理阈值 + Hampel”鲁棒滤波功能。
#   3. 所有其他功能（如单位转换、数据导出等）均保持不变。
# ==============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial import distance as sp_dist
from collections import deque
import pandas as pd

# --- (辅助函数部分) ---
calibration_points = []; resize_ratio = 1.0
def mouse_callback(event, x, y, flags, param):
    global calibration_points, resize_ratio
    if event == cv2.EVENT_LBUTTONDOWN and len(calibration_points) < 2:
        original_x = int(x / resize_ratio); original_y = int(y / resize_ratio); calibration_points.append((original_x, original_y))
        cv2.circle(param['image'], (x, y), 5, (0, 255, 0), -1); cv2.imshow(param['window_name'], param['image'])
        print(f"已选择标定点 {len(calibration_points)}: (原图坐标: {original_x}, {original_y})")

def get_pixel_to_cm_ratio(frame, known_distance_cm, max_width):
    global calibration_points, resize_ratio; calibration_points = []; window_name = "CALIBRATION: Click on two points"
    original_height, original_width = frame.shape[:2]
    if original_width > max_width: resize_ratio = max_width / original_width; display_frame = cv2.resize(frame, (max_width, int(original_height * resize_ratio)))
    else: resize_ratio = 1.0; display_frame = frame.copy()
    callback_param = {'image': display_frame, 'window_name': window_name}; cv2.namedWindow(window_name); cv2.setMouseCallback(window_name, mouse_callback, callback_param)
    print("\n" + "="*50); print("--- 步骤1: 物理尺寸标定 ---"); print(f"请在弹出的窗口中:"); print(f"  1. 用鼠标左键依次点击两个点，这两个点在真实世界中的距离应为 {known_distance_cm} cm。"); print("  2. 完成点击后，按键盘上的 'Enter' 键确认并开始处理。"); print("="*50 + "\n")
    cv2.imshow(window_name, callback_param['image']); cv2.waitKey(0); cv2.destroyWindow(window_name)
    if len(calibration_points) != 2: print("错误：标定点选择不足2个。"); return None
    pixel_distance = sp_dist.euclidean(calibration_points[0], calibration_points[1]); pixels_per_cm = pixel_distance / known_distance_cm
    print(f"\n标定完成: 像素距离 {pixel_distance:.2f} px 对应物理距离 {known_distance_cm} cm。"); print(f"计算出的比例尺为: {pixels_per_cm:.2f} 像素/厘米。\n"); return pixels_per_cm

def reject_outliers_by_threshold(time_series_data, threshold, data_name="数据"):
    if not time_series_data: return []
    original_count = len(time_series_data)
    filtered_data = [item for item in time_series_data if abs(item[2]) <= threshold]
    num_removed = original_count - len(filtered_data)
    if num_removed > 0:
        print(f"  -> 物理剔除: 已移除 {num_removed} 个超出 {data_name} 阈值 {threshold} 的点。")
    return filtered_data

def apply_hampel_filter(time_series_data, window_size=11, n_sigmas=2.5):
    if not time_series_data: return []
    if window_size % 2 == 0: window_size += 1
    frames, times, values = zip(*time_series_data)
    values = np.array(values)
    filtered_values = values.copy()
    k = 1.4826 
    for i in range(len(values)):
        start = max(0, i - window_size // 2)
        end = min(len(values), i + window_size // 2 + 1)
        window = values[start:end]
        local_median = np.median(window)
        mad = np.median(np.abs(window - local_median))
        if mad == 0: continue
        threshold = n_sigmas * k * mad
        if np.abs(values[i] - local_median) > threshold:
            filtered_values[i] = local_median
    return list(zip(frames, times, filtered_values))

# GEM: 恢复为统一的合并绘图函数
def generate_kinematic_plot(track_id, velocity_data, acceleration_data, output_path, start_frame=None, end_frame=None):
    if not velocity_data:
        print(f"错误: 轨迹 {track_id} 在此片段内没有有效的速度数据。")
        return
    
    print(f"--- 正在为轨迹ID: {track_id} (帧 {start_frame}-{end_frame}) 生成运动学图表 ---")
    
    _, v_times, velocities = zip(*velocity_data)
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    title = f'Kinematic Analysis for Track ID: {track_id}'
    if start_frame is not None and end_frame is not None:
        title += f' (Frames {start_frame}-{end_frame})'
    fig.suptitle(title, fontsize=16)
    
    # 绘制速度
    color1 = 'tab:blue'
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Velocity (cm/s)', color=color1, fontsize=12)
    ax1.plot(v_times, velocities, color=color1, marker='o', linestyle='-', markersize=4, label='Velocity')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # 绘制加速度
    if acceleration_data:
        _, a_times, accelerations = zip(*acceleration_data)
        ax2 = ax1.twinx()  # 创建共享x轴的第二个y轴
        color2 = 'tab:red'
        ax2.set_ylabel('Acceleration (cm/s²)', color=color2, fontsize=12)
        ax2.plot(a_times, accelerations, color=color2, marker='x', linestyle='--', markersize=5, label='Acceleration')
        ax2.tick_params(axis='y', labelcolor=color2)
        
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"--- 运动学图表已成功保存至: {output_path} ---")

def export_track_data_to_txt(track_id, trajectories, relative_trajectories, cleaned_velocity_data, cleaned_acceleration_data, pixels_per_cm, output_path):
    print(f"\n--- 正在为轨迹ID: {track_id} 导出数据 ---")
    if track_id not in trajectories: 
        print(f"错误: 轨迹 {track_id} 没有轨迹数据。"); return
    vel_df = pd.DataFrame(cleaned_velocity_data, columns=['Frame', 'Time_s', 'Velocity_cms'])
    accel_df = pd.DataFrame(cleaned_acceleration_data, columns=['Frame', 'Time_s', 'Acceleration_cmss'])
    traj_df = pd.DataFrame(trajectories[track_id], columns=['X_px', 'Y_px', 'Frame'])
    relative_traj_df = pd.DataFrame(relative_trajectories.get(track_id, []), columns=['Relative_X_px', 'Relative_Y_px', 'Frame'])
    final_df = pd.merge(traj_df, relative_traj_df, on='Frame', how='left')
    final_df = pd.merge(final_df, vel_df, on='Frame', how='left')
    final_df = pd.merge(final_df, accel_df, on='Frame', how='left')
    final_df = final_df.sort_values(by='Frame').reset_index(drop=True)
    if pixels_per_cm > 0:
        final_df['Relative_X_mm'] = (final_df['Relative_X_px'] / pixels_per_cm) * 10
        final_df['Relative_Y_mm'] = (final_df['Relative_Y_px'] / pixels_per_cm) * 10
    final_df.interpolate(method='linear', inplace=True)
    output_columns = ['Frame', 'Time_s', 'X_px', 'Y_px', 'Relative_X_mm', 'Relative_Y_mm', 'Velocity_cms', 'Acceleration_cmss']
    for col in output_columns:
        if col not in final_df.columns:
            final_df[col] = np.nan
    final_df = final_df[output_columns]
    final_df.to_csv(output_path, sep='\t', index=False, float_format='%.4f')
    print(f"--- 数据已成功导出至: {output_path} ---")

class KalmanTracker:
    # ... (无变化)
    def __init__(self, track_id, initial_pos, dt):
        self.id = track_id
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1
        self.kf.statePost = np.array([initial_pos[0], initial_pos[1], 0, 0], dtype=np.float32).reshape(-1, 1)
        self.time_since_update = 0
    def predict(self):
        self.time_since_update += 1; return self.kf.predict().flatten()[:2]
    def update(self, measurement):
        self.kf.correct(np.array(measurement, dtype=np.float32).reshape(-1, 1)); self.time_since_update = 0
    def get_state(self): return self.kf.statePost.flatten()

# ==============================================================================
# --- 1. 用户控制面板 (所有可调参数均在此处) ---
# ==============================================================================
video_path = '1.8v-400mv1.7hz.mp4'
KNOWN_DISTANCE_CM = 1.25
FRAME_INTERVAL = 1
TIME_INTERVAL_S = 0.00416
CALIBRATION_WINDOW_WIDTH = 640
lower_orange_color = np.array([0, 87, 19])
upper_orange_color = np.array([15, 255, 255])
min_ball_area = 150; max_ball_area = 1500; min_ball_circularity = 0.4
min_bar_area = 1500; min_bar_aspect_ratio = 5.4; max_bar_angle_deviation = 91.0
EXPECTED_BALL_COUNT = 5
KF_TRACKING_MAX_DISTANCE = 400; TRACK_LOST_THRESHOLD = 5
output_folder = 'kalman_analysis_final_1.7hz'; PREVIEW_MAX_WIDTH = 512
VELOCITY_THRESHOLD_CM_S = 250.0 
ACCELERATION_THRESHOLD_CM_SS = 13000.0

# ==============================================================================
# --- 3. 主处理流程 ---
# ==============================================================================
def main():
    # ... (main函数前半部分无变化，直到 cap.release())
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): print(f"错误：无法打开视频文件 {video_path}"); return
    ret, first_frame = cap.read();
    if not ret: print("无法读取视频第一帧。"); return
    pixels_per_cm = get_pixel_to_cm_ratio(first_frame, KNOWN_DISTANCE_CM, CALIBRATION_WINDOW_WIDTH)
    if pixels_per_cm is None: return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频原始分辨率: {video_width}x{video_height}。"); print("\n--- 开始全自动数据处理，请稍候... ---")
    trackers = {}; next_track_id = 0
    trajectories = {}; velocity_log = {}; acceleration_log = {}
    relative_trajectories = {} 
    previous_velocities = {}
    previous_bar_box = None
    is_initialized = False; main_track_ids = []
    frame_index = 0
    while True:
        ret, frame = cap.read();
        if not ret: break
        frame_index += 1
        if frame_index % 100 == 0:
            print(f"  正在处理: {frame_index} / {total_frames} 帧...")
        if frame_index % FRAME_INTERVAL == 0:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask_orange = cv2.inRange(hsv, lower_orange_color, upper_orange_color)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)); mask_orange = cv2.erode(mask_orange, None, iterations=1); mask_orange = cv2.dilate(mask_orange, kernel, iterations=2)
            all_contours, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            current_ball_centers = []; current_bar_box = None
            if all_contours:
                 for contour in all_contours:
                    area = cv2.contourArea(contour)
                    if min_ball_area < area < max_ball_area:
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0 and (4 * np.pi * area / (perimeter**2)) > min_ball_circularity:
                            (cx, cy), _ = cv2.minEnclosingCircle(contour); current_ball_centers.append((int(cx), int(cy))); continue
                    if area > min_bar_area and current_bar_box is None:
                        rect = cv2.minAreaRect(contour); (x,y), (w,h), angle = rect;
                        if w < h: w,h = h,w; angle += 90
                        aspect_ratio = w/h if h > 0 else 0; angle_dev = min(abs(angle), abs(abs(angle)-90))
                        if aspect_ratio > min_bar_aspect_ratio and angle_dev < max_bar_angle_deviation:
                            current_bar_box = np.intp(cv2.boxPoints(rect))
            predicted_positions = {tid: tracker.predict() for tid, tracker in trackers.items()}
            matched_tracker_ids = set(); matched_detection_indices = set()
            if predicted_positions and current_ball_centers:
                pred_ids=list(predicted_positions.keys()); pred_locs=list(predicted_positions.values())
                if len(pred_locs) > 0:
                    dist_matrix = sp_dist.cdist(np.array(current_ball_centers), np.array(pred_locs))
                    for _ in range(min(len(current_ball_centers), len(pred_locs))):
                        min_val=dist_matrix.min();
                        if min_val>KF_TRACKING_MAX_DISTANCE: break
                        det_idx, pred_idx = np.unravel_index(dist_matrix.argmin(), dist_matrix.shape)
                        track_id = pred_ids[pred_idx]
                        trackers[track_id].update(current_ball_centers[det_idx])
                        matched_tracker_ids.add(track_id); matched_detection_indices.add(det_idx)
                        dist_matrix[det_idx, :] = np.inf; dist_matrix[:, pred_idx] = np.inf
            unmatched_balls = [ball for i, ball in enumerate(current_ball_centers) if i not in matched_detection_indices]
            for center in unmatched_balls:
                trackers[next_track_id] = KalmanTracker(next_track_id, center, TIME_INTERVAL_S); next_track_id += 1
            lost_ids = [tid for tid, tracker in trackers.items() if tracker.time_since_update > TRACK_LOST_THRESHOLD]
            for tid in lost_ids: del trackers[tid]; previous_velocities.pop(tid, None)
            if not is_initialized and len(trackers) >= EXPECTED_BALL_COUNT:
                main_track_ids = list(trackers.keys()); is_initialized = True
                print("\n" + "*"*50); print(f"--- 系统初始化成功! 已锁定 {len(main_track_ids)} 个主要轨迹: {main_track_ids} ---"); print("*"*50 + "\n")
            current_time_s = frame_index * TIME_INTERVAL_S / FRAME_INTERVAL
            if previous_bar_box is not None and current_bar_box is not None:
                prev_rect=cv2.minAreaRect(previous_bar_box); curr_rect=cv2.minAreaRect(current_bar_box)
                (bar_cx, bar_cy), (bar_w, bar_h), bar_angle = curr_rect
                if bar_w < bar_h: bar_angle += 90 
                bar_angle_rad = np.deg2rad(bar_angle); cos_bar = np.cos(-bar_angle_rad); sin_bar = np.sin(-bar_angle_rad)
                prev_cx,prev_cy=prev_rect[0]; curr_cx,curr_cy=curr_rect[0]
                prev_angle=prev_rect[2]
                if prev_rect[1][0]<prev_rect[1][1]: prev_angle+=90
                rotation_angle_rad=np.deg2rad(prev_angle-bar_angle); cos_a,sin_a=np.cos(rotation_angle_rad),np.sin(rotation_angle_rad)
                for track_id, tracker in trackers.items():
                    if track_id in main_track_ids and track_id in trajectories:
                        pos_x, pos_y, _, _ = tracker.get_state()
                        translated_x = pos_x - bar_cx; translated_y = pos_y - bar_cy
                        relative_x = translated_x * cos_bar - translated_y * sin_bar
                        relative_y = translated_x * sin_bar + translated_y * cos_bar
                        relative_trajectories.setdefault(track_id, []).append((relative_x, relative_y, frame_index))
                        if track_id in previous_velocities:
                            prev_pos_x, prev_pos_y, _ = trajectories[track_id][-1]
                            bx_centered = pos_x - curr_cx; by_centered = pos_y - curr_cy
                            bx_rotated = bx_centered*cos_a - by_centered*sin_a; by_rotated = bx_centered*sin_a + by_centered*cos_a
                            transformed_pos = (bx_rotated + prev_cx, by_rotated + prev_cy)
                            pixel_dist = sp_dist.euclidean((prev_pos_x, prev_pos_y), transformed_pos)
                            speed_cm_s = (pixel_dist / pixels_per_cm) / TIME_INTERVAL_S
                            velocity_log.setdefault(track_id, []).append((frame_index, current_time_s, speed_cm_s))
                            prev_speed_cm_s = previous_velocities[track_id]
                            acceleration = (speed_cm_s - prev_speed_cm_s) / TIME_INTERVAL_S
                            acceleration_log.setdefault(track_id, []).append((frame_index, current_time_s, acceleration))
                            previous_velocities[track_id] = speed_cm_s
            for track_id, tracker in trackers.items():
                pos_x, pos_y, _, _ = tracker.get_state()
                trajectories.setdefault(track_id, []).append((int(pos_x), int(pos_y), frame_index))
                if track_id not in previous_velocities:
                    previous_velocities[track_id] = 0
            previous_bar_box = current_bar_box
            for track_id, tracker in trackers.items():
                pos_x, pos_y = tracker.get_state()[:2]
                color = (0, 255, 0) if track_id in main_track_ids else (255, 0, 255)
                cv2.circle(frame, (int(pos_x), int(pos_y)), 10, color, 2)
                cv2.putText(frame, str(track_id), (int(pos_x)+10, int(pos_y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if frame.shape[1] > PREVIEW_MAX_WIDTH:
            scale_ratio = PREVIEW_MAX_WIDTH / frame.shape[1]; preview_height = int(frame.shape[0] * scale_ratio)
            display_frame = cv2.resize(frame, (PREVIEW_MAX_WIDTH, preview_height))
        else: display_frame = frame
        cv2.imshow("Stabilized Kalman Tracker V35 Enhanced", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    
    print("\n--- 全自动数据处理完成 ---")
    cap.release(); cv2.destroyAllWindows()
    if not main_track_ids: print("在整个视频中未能成功初始化并锁定主要轨迹，程序结束。"); return
    
    while True:
        # ... (菜单打印部分无变化) ...
        print("\n" + "="*50)
        print("--- 步骤2: 交互式后处理分析 ---")
        print("已成功锁定并记录了以下主要轨迹:")
        for track_id in main_track_ids:
            if track_id in trajectories:
                print(f"  - 轨迹ID (Track ID): {track_id} (总共捕捉到 {len(trajectories[track_id])} 个数据点)")
        print("\n请选择您要进行的操作:")
        print("  - 输入 'analyze [ID]' ... : 为该轨迹进行【分段】分析 (生成合并图表)")
        print("  - 输入 'plot [ID]'    ... : 为该轨迹生成【完整】图表 (生成合并图表)")
        print("  - 输入 'export [ID]'   ... : 将该轨迹的数据导出为 .txt 文件")
        print("  - 输入 'exit'            : 退出程序")
        user_input = input("> ")

        if user_input.lower() == 'exit':
            print("程序退出。感谢您的一路相伴！"); break
        parts = user_input.lower().split()
        if len(parts) != 2:
            print("错误: 命令格式不正确。请输入 '命令 ID' (例如 'plot 0')"); continue
        command, target_id_str = parts
        try:
            selected_id = int(target_id_str)
        except ValueError:
            print(f"错误: 轨迹ID '{target_id_str}' 必须是一个数字。")
            continue
        if selected_id not in main_track_ids:
            print(f"错误: 轨迹ID {selected_id} 不是一个有效的主轨迹ID。"); continue

        print(f"\n--- 正在为轨迹ID: {selected_id} 清洗和准备数据 ---")
        v_log_raw = velocity_log.get(selected_id, [])
        a_log_raw = acceleration_log.get(selected_id, [])
        v_log_physically_valid = reject_outliers_by_threshold(v_log_raw, VELOCITY_THRESHOLD_CM_S, data_name="速度")
        a_log_physically_valid = reject_outliers_by_threshold(a_log_raw, ACCELERATION_THRESHOLD_CM_SS, data_name="加速度")
        v_log_cleaned = apply_hampel_filter(v_log_physically_valid)
        a_log_cleaned = apply_hampel_filter(a_log_physically_valid)
        print("--- 数据清洗完成 ---")

        try:
            if command == 'analyze':
                try:
                    segment_len_str = input(f"\n您已选择对轨迹 {selected_id} 进行分段分析。请输入每个片段的长度（帧数）: ")
                    segment_length_frames = int(segment_len_str)
                except ValueError:
                    print("错误: 片段长度必须是一个整数。")
                    continue
                if segment_length_frames <= 0:
                    print("错误：片段长度必须是正整数。"); continue
                
                print(f"\n正在为轨迹 {selected_id} 每隔 {segment_length_frames} 帧生成一张运动学图表...")
                points_per_segment = int(segment_length_frames / FRAME_INTERVAL)
                if points_per_segment == 0: points_per_segment = 1
                segment_count = 0
                for i in range(0, len(v_log_cleaned), points_per_segment):
                    v_slice = v_log_cleaned[i : i + points_per_segment]
                    if not v_slice: break
                    start_f = v_slice[0][0]
                    end_f = v_slice[-1][0]
                    a_slice = [item for item in a_log_cleaned if start_f <= item[0] <= end_f]
                    output_path = os.path.join(output_folder, f"kinematics_track_{selected_id}_frames_{start_f}-{end_f}.png")
                    generate_kinematic_plot(selected_id, v_slice, a_slice, output_path, start_f, end_f)
                    segment_count += 1
                print(f"\n分析完成！已为轨迹 {selected_id} 生成了 {segment_count} 张分段图表。")

            elif command == 'plot':
                if v_log_cleaned:
                    start_frame = trajectories[selected_id][0][2]
                    end_frame = trajectories[selected_id][-1][2]
                    output_path = os.path.join(output_folder, f"kinematics_track_ID_{selected_id}_full.png")
                    generate_kinematic_plot(selected_id, v_log_cleaned, a_log_cleaned, output_path, start_frame, end_frame)

            elif command == 'export':
                output_path = os.path.join(output_folder, f"track_data_ID_{selected_id}.txt")
                export_track_data_to_txt(selected_id, trajectories, relative_trajectories, v_log_cleaned, a_log_cleaned, pixels_per_cm, output_path)
            
            else:
                print(f"错误: 未知的命令 '{command}'。")

        except Exception as e:
            print(f"处理时发生意外错误: {e}")

if __name__ == '__main__':
    main()