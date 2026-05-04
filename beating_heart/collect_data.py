"""
手势数据采集脚本
使用 MediaPipe Hands 检测手部关键点，按类别录制训练数据。

操作说明：
  h - 切换到"单手比心"类别（一只手，拇指和食指交叉形成心形）
  d - 切换到"双手比心"类别（两只手，共同构成爱心轮廓）
  f - 切换到"拳头"类别
  空格 - 录制当前帧的关键点
  q - 退出

注意：
  - 单手比心(h)和拳头(f)：录制时只需一只手在画面中
  - 双手比心(d)：录制时需要两只手同时在画面中
"""

import cv2
import numpy as np
import mediapipe as mp
import json
import os
import time

# ============================================================
# 配置
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "gesture_data")
# 单手类别：录制单只手的 21 个关键点
SINGLE_HAND_LABELS = {"h": "heart", "f": "fist"}
# 双手类别：录制两只手的 42 个关键点
DOUBLE_HAND_LABELS = {"d": "double_heart"}
ALL_LABELS = {**SINGLE_HAND_LABELS, **DOUBLE_HAND_LABELS}
CAMERA_INDEX = 0


# ============================================================
# 主函数
# ============================================================
def main():
    # 创建数据目录
    for label in ALL_LABELS.values():
        os.makedirs(os.path.join(DATA_DIR, label), exist_ok=True)

    # 初始化 MediaPipe
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # 打开摄像头
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    current_label = "heart"
    sample_counts = {}
    for label in ALL_LABELS.values():
        sample_counts[label] = len([f for f in os.listdir(os.path.join(DATA_DIR, label))
                                    if f.endswith(".json")])

    print("=" * 55)
    print("手势数据采集脚本")
    print("=" * 55)
    print("按键说明：")
    print("  h - 单手比心（一只手，拇指和食指交叉）")
    print("  d - 双手比心（两只手，共同构成爱心）")
    print("  f - 拳头")
    print("  空格 - 录制当前帧")
    print("  q - 退出")
    print("=" * 55)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

        # ---- 绘制状态信息 ----
        # 当前类别
        is_double = current_label in DOUBLE_HAND_LABELS.values()
        label_color = (0, 0, 255) if is_double else (0, 255, 0)
        cv2.putText(frame, f"Label: {current_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, label_color, 2)

        # 提示需要几只手
        need_hands = 2 if is_double else 1
        hands_ok = num_hands >= need_hands
        hands_color = (0, 255, 0) if hands_ok else (0, 0, 255)
        cv2.putText(frame, f"Hands: {num_hands}/{need_hands}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, hands_color, 2)

        # 各类别样本数
        y_offset = 100
        for label, count in sample_counts.items():
            color = (0, 255, 255) if label == current_label else (200, 200, 200)
            cv2.putText(frame, f"{label}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 35

        # 操作提示
        cv2.putText(frame, "SPACE=Record  h/d/f=Switch  q=Quit",
                    (w // 2 - 220, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # 绘制手部关键点
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Gesture Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('h'):
            current_label = "heart"
            print(f"切换到: {current_label} (单手)")
        elif key == ord('d'):
            current_label = "double_heart"
            print(f"切换到: {current_label} (双手)")
        elif key == ord('f'):
            current_label = "fist"
            print(f"切换到: {current_label} (单手)")
        elif key == ord(' '):
            can_record = False

            if current_label in SINGLE_HAND_LABELS.values():
                # 单手类别：需要检测到至少一只手
                if num_hands >= 1:
                    can_record = True
                    # 只取第一只手
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks_data = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                else:
                    print("未检测到手，请将手放到摄像头前")
            elif current_label in DOUBLE_HAND_LABELS.values():
                # 双手类别：需要检测到两只手
                if num_hands >= 2:
                    can_record = True
                    # 按 x 坐标排序，确保左手在前右手在后
                    hands_sorted = sorted(
                        results.multi_hand_landmarks,
                        key=lambda hl: hl.landmark[0].x
                    )
                    landmarks_data = []
                    for hl in hands_sorted:
                        for lm in hl.landmark:
                            landmarks_data.append([lm.x, lm.y, lm.z])
                else:
                    print("需要两只手，请双手同时做出比心手势")

            if can_record:
                timestamp = int(time.time() * 1000)
                filename = f"{current_label}_{timestamp}.json"
                filepath = os.path.join(DATA_DIR, current_label, filename)

                data = {"landmarks": landmarks_data, "label": current_label}
                with open(filepath, "w") as f:
                    json.dump(data, f)

                sample_counts[current_label] += 1
                print(f"已录制: {current_label} (总计: {sample_counts[current_label]})")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    print("\n录制完成！各类别样本数：")
    for label, count in sample_counts.items():
        print(f"  {label}: {count}")


if __name__ == "__main__":
    main()
