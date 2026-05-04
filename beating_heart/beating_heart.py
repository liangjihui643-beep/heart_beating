"""
跳动爱心手势检测器
通过摄像头检测"比心"手势，检测到后在屏幕上显示大量跳动的爱心动画。
支持 TFLite 神经网络分类器，缺失时自动回退到硬编码规则。
"""

import cv2
import numpy as np
import mediapipe as mp
import math
import time
import random
import os
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont

# ============================================================
# 常量定义
# ============================================================
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# 手势检测阈值
GESTURE_DISTANCE_THRESHOLD = 0.08   # 单手比心：拇指尖与食指尖的最大归一化距离
DOUBLE_HEART_THUMB_DIST = 0.12     # 双手比心：两拇指尖之间的最大距离
DOUBLE_HEART_FINGER_CURLED = 0.03  # 双手比心：手指弯曲判断阈值（指尖需低于PIP关节此值）

# 爱心动画参数
MAX_HEARTS = 80          # 最大同时显示的爱心数量
HEART_SPAWN_RATE = 5     # 每帧生成的爱心数量
HEART_LIFETIME = 4.0     # 爱心生命周期（秒）
HEART_BASE_SIZE = 25     # 爱心基础大小（像素）

# 粉色/红色系调色板 (BGR格式)
COLOR_PALETTE = [
    (147, 20, 255),   # 粉红
    (0, 0, 255),      # 红色
    (80, 127, 255),   # 浅粉
    (99, 99, 255),    # 玫瑰红
    (180, 105, 255),  # 紫粉
    (0, 140, 255),    # 橙红
    (128, 0, 255),    # 深粉
    (0, 80, 255),     # 深红
]

# TFLite 模型配置
GESTURE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_classifier.tflite")
GESTURE_LABELS = {0: "heart", 1: "double_heart", 2: "fist"}  # 模型输出


# ============================================================
# GestureClassifier - TFLite 手势分类器
# ============================================================
class GestureClassifier:
    """基于 TFLite 的手势分类器，输入 21 个手部关键点，输出手势类别。"""

    def __init__(self, model_path=GESTURE_MODEL_PATH):
        self.interpreter = None
        self.input_details = None
        self.output_details = None

        if not os.path.exists(model_path):
            print(f"[模型] 文件不存在: {model_path}")
            print(f"[模型] 将使用规则回退模式。请先运行 collect_data.py 和 train_model.py")
            return

        try:
            from ai_edge_litert.interpreter import Interpreter
            self.interpreter = Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"[模型] 加载成功: {model_path}")
        except ImportError:
            print(f"[模型] 未安装 ai-edge-litert，请运行: pip install ai-edge-litert")
            print(f"[模型] 将使用规则回退模式")
        except Exception as e:
            print(f"[模型] 加载失败: {e}，将使用规则回退模式")

    def predict(self, landmarks):
        """
        预测单手手势类别。

        Args:
            landmarks: MediaPipe 的 21 个 landmark 对象列表

        Returns:
            (label_id, confidence): 类别ID和置信度
            如果模型未加载，返回 (-1, 0.0)
        """
        if self.interpreter is None:
            return -1, 0.0

        vector = self._extract_landmark_vector(landmarks)
        input_data = np.expand_dims(vector, axis=0).astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        label_id = int(np.argmax(output))
        confidence = float(output[label_id])
        return label_id, confidence

    def predict_double_hand(self, landmarks_list):
        """
        预测双手手势类别。

        Args:
            landmarks_list: 两只手的 MediaPipe landmark 列表

        Returns:
            (label_id, confidence): 类别ID和置信度
        """
        if self.interpreter is None or len(landmarks_list) < 2:
            return -1, 0.0

        vector = self._extract_double_hand_vector(landmarks_list)
        input_data = np.expand_dims(vector, axis=0).astype(np.float32)

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        label_id = int(np.argmax(output))
        confidence = float(output[label_id])
        return label_id, confidence

    @staticmethod
    def _extract_landmark_vector(landmarks):
        """
        将 21 个关键点转换为归一化的 126 维向量。
        前63维：归一化后的手部数据
        后63维：全零（无第二只手）
        """
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        wrist = points[0]
        centered = points - wrist
        palm_size = np.linalg.norm(centered[9])
        if palm_size < 1e-6:
            palm_size = 1.0
        normalized = (centered / palm_size).flatten()  # (63,)
        return np.concatenate([normalized, np.zeros(63, dtype=np.float32)])

    @staticmethod
    def _extract_double_hand_vector(landmarks_list):
        """
        将两只手的 42 个关键点转换为归一化的 126 维向量。
        前63维：左手归一化数据
        后63维：右手归一化数据
        """
        def normalize_hand(landmarks):
            points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            wrist = points[0]
            centered = points - wrist
            palm_size = np.linalg.norm(centered[9])
            if palm_size < 1e-6:
                palm_size = 1.0
            return (centered / palm_size).flatten()

        # 按 x 坐标排序，左手在前
        sorted_hands = sorted(landmarks_list, key=lambda hl: hl.landmark[0].x)
        left_vec = normalize_hand(sorted_hands[0].landmark)
        right_vec = normalize_hand(sorted_hands[1].landmark)
        return np.concatenate([left_vec, right_vec])


# ============================================================
# HeartLandmarkDetector - 手势检测器
# ============================================================
class HeartLandmarkDetector:
    """封装摄像头捕获和 MediaPipe 手部关键点检测。"""

    def __init__(self, camera_index=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT, classifier=None):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.hands = None
        self.classifier = classifier
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils

    def start(self):
        """打开摄像头并初始化 MediaPipe Hands。"""
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 (index={self.camera_index})")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        print("摄像头已启动，按 'q' 退出")
        print("请做出比心手势（握拳，拇指和食指交错形成心形）")

    def stop(self):
        """释放摄像头和 MediaPipe 资源。"""
        if self.cap:
            self.cap.release()
        if self.hands:
            self.hands.close()

    def read_frame(self):
        """从摄像头读取一帧画面。返回 (success, frame)。"""
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def detect_gesture(self, frame):
        """
        检测画面中手势。
        返回 (gesture_type, hand_landmarks_list)。
        gesture_type: 0=无手势, 1=单手比心, 2=双手比心, 3=拳头
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if not results.multi_hand_landmarks:
            return 0, []

        landmarks_list = results.multi_hand_landmarks

        # 如果分类器可用，使用神经网络
        if self.classifier and self.classifier.interpreter is not None:
            return self._detect_with_classifier(landmarks_list)
        else:
            # 回退到硬编码规则
            return self._detect_with_rules(landmarks_list)

    def _detect_with_classifier(self, landmarks_list):
        """使用 TFLite 分类器检测手势。"""
        # 优先检测双手比心
        if len(landmarks_list) >= 2:
            label_id, confidence = self.classifier.predict_double_hand(landmarks_list)
            if label_id == 1 and confidence > 0.6:  # 1=double_heart
                return 2, landmarks_list

        # 检测单手手势（比心或拳头）
        for hand_landmarks in landmarks_list:
            label_id, confidence = self.classifier.predict(hand_landmarks.landmark)
            if confidence > 0.6:
                if label_id == 0:  # heart
                    return 1, landmarks_list
                elif label_id == 2:  # fist
                    return 3, landmarks_list

        return 0, landmarks_list

    def _detect_with_rules(self, landmarks_list):
        """回退模式：使用硬编码规则检测手势。"""
        # 优先检测双手比心
        if len(landmarks_list) >= 2:
            if self._fallback_double_heart(landmarks_list):
                return 2, landmarks_list

        # 检测单手比心
        for hand_landmarks in landmarks_list:
            if self._fallback_heart(hand_landmarks.landmark):
                return 1, landmarks_list

        # 检测拳头
        for hand_landmarks in landmarks_list:
            if self._fallback_fist(hand_landmarks.landmark):
                return 3, landmarks_list

        return 0, landmarks_list

    def draw_landmarks(self, frame, hand_landmarks_list):
        """在画面上绘制手部关键点（调试用）。"""
        for hand_landmarks in hand_landmarks_list:
            self.mp_draw.draw_landmarks(
                frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )

    def _fallback_heart(self, landmarks):
        """
        单手比心手势识别。

        标准单手比心特征：
        1. 拇指尖(4)和食指尖(8)相互靠近、距离极近，贴合围成心形上半部分
        2. 拇指根部(2)和食指根部(5)自然张开，形成爱心轮廓
        3. 中指(12)、无名指(16)、小指(20)完全自然伸直，不弯曲
        4. 手掌侧立或正对镜头都可以
        """
        # ---- 条件1: 拇指尖与食指尖极近 ----
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        dx = thumb_tip.x - index_tip.x
        dy = thumb_tip.y - index_tip.y
        tip_dist = math.sqrt(dx * dx + dy * dy)

        if tip_dist > GESTURE_DISTANCE_THRESHOLD:
            return False

        # ---- 条件2: 拇指根部与食指根部自然张开 ----
        # 拇指MCP(2)与食指MCP(5)之间应有一定距离
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        dx_base = thumb_mcp.x - index_mcp.x
        dy_base = thumb_mcp.y - index_mcp.y
        base_dist = math.sqrt(dx_base * dx_base + dy_base * dy_base)

        # 根部距离应大于指尖距离（形成张开的爱心轮廓）
        if base_dist < tip_dist * 1.2:
            return False

        # ---- 条件3: 中指、无名指、小指完全伸直 ----
        # 伸直判断：指尖(8)在PIP关节上方（y更小）
        for tip_idx, pip_idx in [(12, 10), (16, 14), (20, 18)]:
            if landmarks[tip_idx].y > landmarks[pip_idx].y + 0.02:
                return False

        return True

    def _fallback_double_heart(self, landmarks_list):
        """
        双手比心检测逻辑。

        条件：
        1. 必须同时检测到两只手
        2. 左右手拇指指尖相互靠近贴合
        3. 左右手食指分别弯曲，共同构成爱心轮廓
        4. 两只手的中指、无名指、小指全部自然弯曲收拢
        """
        if len(landmarks_list) < 2:
            return False

        hand1 = landmarks_list[0].landmark
        hand2 = landmarks_list[1].landmark

        # ---- 条件2: 两拇指尖靠近 ----
        thumb1 = hand1[4]  # 手1拇指尖
        thumb2 = hand2[4]  # 手2拇指尖
        dx = thumb1.x - thumb2.x
        dy = thumb1.y - thumb2.y
        thumb_dist = math.sqrt(dx * dx + dy * dy)

        if thumb_dist > DOUBLE_HEART_THUMB_DIST:
            return False

        # ---- 条件3: 两手食指弯曲（指尖在DIP关节下方） ----
        # 食指尖(8) 应该在 食指DIP(7) 下方（y更大表示更低）
        index1_curved = hand1[8].y > hand1[7].y - DOUBLE_HEART_FINGER_CURLED
        index2_curved = hand2[8].y > hand2[7].y - DOUBLE_HEART_FINGER_CURLED

        if not (index1_curved and index2_curved):
            return False

        # ---- 条件4: 两手的中指、无名指、小指全部弯曲 ----
        finger_pairs = [(12, 10), (16, 14), (20, 18)]  # (tip, pip)

        for hand in [hand1, hand2]:
            for tip_idx, pip_idx in finger_pairs:
                # 指尖(8)应该在PIP关节下方（弯曲状态）
                if hand[tip_idx].y < hand[pip_idx].y - DOUBLE_HEART_FINGER_CURLED:
                    return False

        return True

    def _fallback_fist(self, landmarks):
        """
        拳头手势识别。

        标准拳头特征：
        1. 拇指弯曲，压住其他四指外侧或贴合拳面
        2. 食指、中指、无名指、小指全部完全弯曲，指尖向手心收拢
        3. 所有手指中间关节、末端关节都呈弯曲状态
        4. 五根手指互相靠拢、并拢，无明显张开缝隙
        5. 手掌呈收拢包裹状态
        """
        # ---- 条件1: 四指全部弯曲 ----
        # 指尖在PIP关节下方（弯曲状态）
        finger_pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]  # (tip, pip)
        fingers_curled = 0
        for tip_idx, pip_idx in finger_pairs:
            if landmarks[tip_idx].y > landmarks[pip_idx].y - 0.02:
                fingers_curled += 1

        # 四根手指必须全部弯曲
        if fingers_curled < 4:
            return False

        # ---- 条件2: 拇指弯曲 ----
        # 拇指尖(4)在拇指IP(3)下方或接近（弯曲压住其他手指）
        thumb_curved = landmarks[4].y > landmarks[3].y - 0.03
        if not thumb_curved:
            return False

        # ---- 条件3: 手指并拢 ----
        # 中指MCP(9)与无名指MCP(13)之间距离小
        dx = landmarks[9].x - landmarks[13].x
        dy = landmarks[9].y - landmarks[13].y
        mid_ring_dist = math.sqrt(dx * dx + dy * dy)

        # 无名指MCP(13)与小指MCP(17)之间距离小
        dx2 = landmarks[13].x - landmarks[17].x
        dy2 = landmarks[13].y - landmarks[17].y
        ring_pinky_dist = math.sqrt(dx2 * dx2 + dy2 * dy2)

        # 手指并拢阈值（归一化距离）
        if mid_ring_dist > 0.12 or ring_pinky_dist > 0.12:
            return False

        # ---- 条件4: 不是比心手势 ----
        # 拇指尖与食指尖不应太近（否则是比心）
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        dx_ti = thumb_tip.x - index_tip.x
        dy_ti = thumb_tip.y - index_tip.y
        ti_dist = math.sqrt(dx_ti * dx_ti + dy_ti * dy_ti)

        # 如果拇指食指尖太近，可能是比心而不是拳头
        if ti_dist < 0.06:
            return False

        return True


# ============================================================
# HeartParticle - 爱心粒子数据类
# ============================================================
@dataclass
class HeartParticle:
    """表示屏幕上的一个爱心动画粒子。"""
    x: float            # 中心 x 坐标（像素）
    y: float            # 中心 y 坐标（像素）
    size: float         # 基础大小（像素）
    color: tuple        # BGR 颜色
    birth_time: float   # 创建时间
    velocity_y: float   # 垂直速度（像素/秒，负值=向上）
    velocity_x: float   # 水平漂移速度（像素/秒）
    phase_offset: float # 跳动动画相位偏移
    rotation: float     # 旋转角度（弧度）
    trail: list = None  # 拖尾位置历史 [(x, y, size, time), ...]
    trail_max: int = 8  # 最大拖尾帧数

    def __post_init__(self):
        if self.trail is None:
            self.trail = []


# ============================================================
# HeartAnimationEngine - 爱心动画引擎
# ============================================================
class HeartAnimationEngine:
    """管理爱心粒子的生命周期、物理更新和绘制。"""

    def __init__(self, max_hearts=MAX_HEARTS, spawn_rate=HEART_SPAWN_RATE):
        self.max_hearts = max_hearts
        self.spawn_rate = spawn_rate
        self.hearts = []
        self._prev_particle_frame = None

    def spawn_hearts(self, count, frame_width, frame_height):
        """在画面底部区域生成 count 个新爱心粒子。"""
        current_time = time.time()
        for _ in range(count):
            if len(self.hearts) >= self.max_hearts:
                break
            heart = HeartParticle(
                x=random.uniform(0, frame_width),
                y=random.uniform(frame_height * 0.5, frame_height),
                size=random.uniform(HEART_BASE_SIZE * 0.5, HEART_BASE_SIZE * 1.5),
                color=random.choice(COLOR_PALETTE),
                birth_time=current_time,
                velocity_y=random.uniform(50, 150),
                velocity_x=random.uniform(-20, 20),
                phase_offset=random.uniform(0, 2 * math.pi),
                rotation=random.uniform(-0.3, 0.3),
            )
            self.hearts.append(heart)

    def update(self, dt):
        """更新所有粒子的物理状态，移除过期粒子。"""
        current_time = time.time()
        alive = []
        for heart in self.hearts:
            age = current_time - heart.birth_time
            if age > HEART_LIFETIME:
                continue
            # 记录拖尾位置
            heart.trail.append((heart.x, heart.y, heart.size, current_time))
            if len(heart.trail) > heart.trail_max:
                heart.trail.pop(0)
            heart.y -= heart.velocity_y * dt
            heart.x += heart.velocity_x * dt
            alive.append(heart)
        self.hearts = alive

    def draw(self, frame):
        """将所有爱心绘制到画面上。返回修改后的画面。"""
        current_time = time.time()
        overlay = np.zeros_like(frame)
        glow_overlay = np.zeros_like(frame)
        trail_overlay = np.zeros_like(frame)

        for heart in self.hearts:
            age = current_time - heart.birth_time
            alpha = 1.0 - (age / HEART_LIFETIME)
            if alpha <= 0:
                continue

            # 绘制拖尾残影
            for ti, (tx, ty, ts, tt) in enumerate(heart.trail):
                trail_age = current_time - tt
                trail_alpha = (ti / max(1, len(heart.trail))) * 0.4 * alpha
                if trail_alpha < 0.02:
                    continue
                trail_size = ts * (0.3 + 0.7 * ti / max(1, len(heart.trail)))
                trail_pts = self._generate_heart_points(tx, ty, trail_size, heart.rotation)
                trail_color = tuple(int(c * trail_alpha) for c in heart.color)
                cv2.fillPoly(trail_overlay, [trail_pts], trail_color)

            # 跳动效果：更明显的缩放 + 颜色亮度变化
            beat_phase = 2 * math.pi * 2.5 * (age + heart.phase_offset)
            beat_scale = heart.size * (1.0 + 0.3 * math.sin(beat_phase))

            # 跳动时颜色亮度变化
            brightness_boost = 1.0 + 0.2 * max(0, math.sin(beat_phase))
            beat_color = tuple(
                min(255, int(c * brightness_boost)) for c in heart.color
            )

            # 绘制发光光晕（更大的半透明爱心）
            glow_scale = beat_scale * 1.4
            glow_points = self._generate_heart_points(
                heart.x, heart.y, glow_scale, heart.rotation
            )
            glow_color = tuple(c // 4 for c in heart.color)
            cv2.fillPoly(glow_overlay, [glow_points], glow_color)

            # 绘制主爱心
            points = self._generate_heart_points(
                heart.x, heart.y, beat_scale, heart.rotation
            )
            cv2.fillPoly(overlay, [points], beat_color)

            # 绘制高光（更小的白色偏移爱心）
            highlight_scale = beat_scale * 0.4
            hx = heart.x - beat_scale * 0.15
            hy = heart.y - beat_scale * 0.15
            highlight_points = self._generate_heart_points(
                hx, hy, highlight_scale, heart.rotation
            )
            highlight_color = (
                min(255, heart.color[0] + 80),
                min(255, heart.color[1] + 80),
                min(255, heart.color[2] + 80),
            )
            cv2.fillPoly(overlay, [highlight_points], highlight_color)

        # 混合拖尾层
        trail_mask = np.any(trail_overlay > 0, axis=2)
        if np.any(trail_mask):
            frame[trail_mask] = cv2.addWeighted(
                trail_overlay, 0.5, frame, 0.5, 0
            )[trail_mask]

        # 混合光晕层
        glow_mask = np.any(glow_overlay > 0, axis=2)
        if np.any(glow_mask):
            frame[glow_mask] = cv2.addWeighted(
                glow_overlay, 0.4, frame, 0.6, 0
            )[glow_mask]

        # 整体残影：将粒子层保留一帧产生运动模糊
        if (hasattr(self, '_prev_particle_frame') and self._prev_particle_frame is not None
                and self._prev_particle_frame.shape == frame.shape):
            frame = cv2.addWeighted(frame, 0.85, self._prev_particle_frame, 0.15, 0)
        self._prev_particle_frame = frame.copy()

        # 混合主爱心层
        mask = np.any(overlay > 0, axis=2)
        if np.any(mask):
            frame[mask] = cv2.addWeighted(
                overlay, 0.85, frame, 0.15, 0
            )[mask]

        return frame

    @staticmethod
    def _generate_heart_points(center_x, center_y, scale, rotation=0.0, num_points=100):
        """
        使用参数方程生成爱心形状的点坐标。

        x(t) = 16 * sin(t)^3
        y(t) = 13*cos(t) - 5*cos(2t) - 2*cos(3t) - cos(4t)
        """
        t = np.linspace(0, 2 * np.pi, num_points)
        x = 16 * np.sin(t) ** 3
        y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))

        # 缩放和平移
        x = center_x + x * scale / 16.0
        y = center_y + y * scale / 16.0

        # 旋转
        if rotation != 0.0:
            cos_r = math.cos(rotation)
            sin_r = math.sin(rotation)
            dx = x - center_x
            dy = y - center_y
            x = center_x + dx * cos_r - dy * sin_r
            y = center_y + dx * sin_r + dy * cos_r

        points = np.column_stack([x, y]).astype(np.int32)
        return points


# ============================================================
# Sparkle - 光芒粒子
# ============================================================
@dataclass
class Sparkle:
    """漂浮的光芒粒子。"""
    x: float
    y: float
    size: float
    color: tuple
    birth_time: float
    velocity_y: float
    velocity_x: float
    twinkle_speed: float
    twinkle_offset: float


# ============================================================
# DoubleHeartEffect - 双手比心特效
# ============================================================
class DoubleHeartEffect:
    """
    双手比心时的全屏华丽特效。
    包含：全屏红色渐变、中央大爱心跳动、光芒粒子、光晕脉冲。
    """

    def __init__(self):
        self.active = False
        self.activate_time = 0.0
        self.fade_in_duration = 0.8   # 淡入时间
        self.sparkles = []
        self.max_sparkles = 120
        self.transition_alpha = 0.0   # 用于平滑过渡

    def activate(self):
        """激活双手比心特效。"""
        if not self.active:
            self.active = True
            self.activate_time = time.time()
            self.sparkles.clear()

    def deactivate(self):
        """停用双手比心特效。"""
        self.active = False
        self.sparkles.clear()

    def update(self, dt, frame_width, frame_height):
        """更新特效状态和粒子。"""
        if not self.active:
            self.transition_alpha = max(0, self.transition_alpha - dt * 2.0)
            return

        current_time = time.time()
        age = current_time - self.activate_time

        # 淡入过渡
        self.transition_alpha = min(1.0, age / self.fade_in_duration)

        # 持续生成光芒粒子
        if len(self.sparkles) < self.max_sparkles:
            for _ in range(3):
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(50, min(frame_width, frame_height) * 0.4)
                cx, cy = frame_width // 2, frame_height // 2
                self.sparkles.append(Sparkle(
                    x=cx + math.cos(angle) * dist,
                    y=cy + math.sin(angle) * dist,
                    size=random.uniform(2, 6),
                    color=random.choice([
                        (255, 200, 200), (255, 150, 150), (255, 255, 200),
                        (200, 200, 255), (255, 180, 220),
                    ]),
                    birth_time=current_time,
                    velocity_y=random.uniform(-30, -80),
                    velocity_x=random.uniform(-15, 15),
                    twinkle_speed=random.uniform(3, 8),
                    twinkle_offset=random.uniform(0, 2 * math.pi),
                ))

        # 更新粒子
        alive = []
        for s in self.sparkles:
            s_age = current_time - s.birth_time
            if s_age > 2.5:
                continue
            s.y += s.velocity_y * dt
            s.x += s.velocity_x * dt
            alive.append(s)
        self.sparkles = alive

    def draw(self, frame):
        """绘制双手比心特效。返回修改后的画面。"""
        if self.transition_alpha <= 0.01:
            return frame

        current_time = time.time()
        age = current_time - self.activate_time if self.active else 0
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        alpha = self.transition_alpha

        # ---- 1. 全屏红色渐变叠加 ----
        red_overlay = np.zeros_like(frame)
        # 径向渐变：中心最红，边缘较淡
        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_dist = math.sqrt(cx ** 2 + cy ** 2)
        radial_fade = 1.0 - (dist_from_center / max_dist)
        radial_fade = np.clip(radial_fade, 0.15, 1.0)

        # 脉冲效果
        pulse = 0.6 + 0.15 * math.sin(2 * math.pi * 1.5 * age)

        red_overlay[:, :, 2] = (radial_fade * 80 * pulse * alpha).astype(np.uint8)  # R通道
        red_overlay[:, :, 1] = (radial_fade * 10 * pulse * alpha).astype(np.uint8)  # G通道
        red_overlay[:, :, 0] = (radial_fade * 20 * pulse * alpha).astype(np.uint8)  # B通道

        frame = cv2.add(frame, red_overlay)

        # ---- 2. 中央大爱心（多层） ----
        base_size = min(w, h) * 0.25

        # 弹性跳动：快速膨胀 + 缓慢恢复（"砰"的重击感）
        beat_phase = 2 * math.pi * 2.0 * age
        raw_beat = math.sin(beat_phase)
        # 用缓动函数：快速膨胀(sin正半周取平方根)，缓慢恢复(sin负半周取平方)
        if raw_beat >= 0:
            eased_beat = math.sqrt(raw_beat) * 0.25
        else:
            eased_beat = -(raw_beat ** 2) * 0.12
        beat_scale = base_size * (1.0 + eased_beat)

        # 外层光晕（最大、最模糊）- 更多层更柔和
        for i in range(4):
            glow_size = beat_scale * (1.8 - i * 0.15)
            glow_alpha = alpha * (0.10 - i * 0.02)
            glow_color = (30 - i * 5, 10 - i * 2, 100 - i * 20)
            glow_pts = HeartAnimationEngine._generate_heart_points(
                cx, cy, glow_size
            )
            glow_layer = np.zeros_like(frame)
            cv2.fillPoly(glow_layer, [glow_pts], glow_color)
            frame = cv2.addWeighted(glow_layer, glow_alpha, frame, 1.0, 0)

        # 主爱心（渐变填充效果：多层叠加）
        main_pts = HeartAnimationEngine._generate_heart_points(
            cx, cy, beat_scale
        )
        brightness = 1.0 + 0.15 * max(0, math.sin(beat_phase))
        main_color = (
            int(80 * brightness),
            int(20 * brightness),
            int(220 * brightness),
        )
        main_layer = np.zeros_like(frame)
        cv2.fillPoly(main_layer, [main_pts], main_color)
        frame = cv2.addWeighted(main_layer, 0.9 * alpha, frame, 1.0, 0)

        # 白色轮廓线描边
        outline_layer = np.zeros_like(frame)
        cv2.polylines(outline_layer, [main_pts], True, (220, 200, 255), 3, cv2.LINE_AA)
        frame = cv2.addWeighted(outline_layer, 0.7 * alpha, frame, 1.0, 0)

        # 内部条纹高光（多层偏移的半透明白色爱心）
        for si, (s_scale, sx, sy, s_alpha) in enumerate([
            (0.65, -0.08, -0.10, 0.35),
            (0.45, -0.12, -0.14, 0.25),
            (0.30, -0.05, -0.08, 0.20),
        ]):
            stripe_size = beat_scale * s_scale
            stripe_x = cx + beat_scale * sx
            stripe_y = cy + beat_scale * sy
            stripe_pts = HeartAnimationEngine._generate_heart_points(
                stripe_x, stripe_y, stripe_size
            )
            stripe_layer = np.zeros_like(frame)
            cv2.fillPoly(stripe_layer, [stripe_pts], (255, 240, 255))
            frame = cv2.addWeighted(stripe_layer, s_alpha * alpha, frame, 1.0, 0)

        # 最亮中心点（玻璃高光）
        dot_size = beat_scale * 0.18
        dot_pts = HeartAnimationEngine._generate_heart_points(
            cx - beat_scale * 0.06, cy - beat_scale * 0.12, dot_size
        )
        dot_layer = np.zeros_like(frame)
        cv2.fillPoly(dot_layer, [dot_pts], (250, 220, 255))
        frame = cv2.addWeighted(dot_layer, 0.65 * alpha, frame, 1.0, 0)

        # 十字光芒（从爱心中心向外）
        ray_layer = np.zeros_like(frame)
        for ri in range(4):
            angle = math.pi / 4 * ri + age * 0.5
            ray_len = beat_scale * 0.4
            end_x = int(cx + math.cos(angle) * ray_len)
            end_y = int(cy + math.sin(angle) * ray_len)
            cv2.line(ray_layer, (cx, cy), (end_x, end_y), (150, 100, 220), 2, cv2.LINE_AA)
        frame = cv2.addWeighted(ray_layer, 0.3 * alpha, frame, 1.0, 0)

        # ---- 3. 光芒粒子 ----
        sparkle_layer = np.zeros_like(frame)
        for s in self.sparkles:
            s_age = current_time - s.birth_time
            s_alpha = 1.0 - (s_age / 2.5)
            if s_alpha <= 0:
                continue
            twinkle = 0.5 + 0.5 * math.sin(
                s.twinkle_speed * (s_age + s.twinkle_offset)
            )
            r = int(s.size * twinkle * s_alpha)
            if r < 1:
                continue
            color = tuple(int(c * s_alpha) for c in s.color)
            cv2.circle(sparkle_layer, (int(s.x), int(s.y)), r, color, -1)
            # 十字光芒
            if r > 2:
                line_len = int(r * 2.5 * twinkle)
                cv2.line(sparkle_layer,
                         (int(s.x) - line_len, int(s.y)),
                         (int(s.x) + line_len, int(s.y)),
                         color, max(1, r // 2))
                cv2.line(sparkle_layer,
                         (int(s.x), int(s.y) - line_len),
                         (int(s.x), int(s.y) + line_len),
                         color, max(1, r // 2))

        frame = cv2.addWeighted(sparkle_layer, 0.8 * alpha, frame, 1.0, 0)

        # ---- 4. 边缘暗角效果 ----
        vignette = np.zeros_like(frame)
        vignette_fade = (dist_from_center / max_dist)
        vignette_fade = np.clip(vignette_fade - 0.5, 0, 0.5) * 2
        vignette_val = np.nan_to_num(vignette_fade * 30 * alpha, nan=0.0, posinf=255.0, neginf=0.0)
        vignette_val = np.clip(vignette_val, 0, 255).astype(np.uint8)
        vignette[:, :, 0] = vignette_val
        vignette[:, :, 1] = vignette_val
        vignette[:, :, 2] = vignette_val
        frame = cv2.subtract(frame, vignette)

        return frame


# ============================================================
# ScrollingText3D - 滚动3D旋转文字特效（支持中文）
# ============================================================

# 查找系统中文字体
def _find_chinese_font():
    """查找可用的中文字体路径。"""
    import os
    candidates = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None

_CHINESE_FONT_PATH = _find_chinese_font()


class ScrollingText3D:
    """
    在屏幕上1/4区域滚动播放中文文字，文字带3D旋转效果。
    使用 PIL 渲染中文字符。
    """

    def __init__(self, text="My lover", font_size=72):
        self.text = text
        self.font_size = font_size
        self.scroll_speed = 150          # 像素/秒
        self.rotation_speed = 1.2        # 旋转频率 (Hz)
        self.char_spacing = 20           # 字符间距（像素）
        self.active = False
        self.activate_time = 0.0
        self.transition_alpha = 0.0
        self.fade_duration = 0.5

        # 加载 PIL 字体
        if _CHINESE_FONT_PATH:
            self.pil_font = ImageFont.truetype(_CHINESE_FONT_PATH, font_size)
            # 用稍小的字体做阴影和发光
            self.pil_font_small = ImageFont.truetype(_CHINESE_FONT_PATH, font_size - 4)
        else:
            print("警告：未找到中文字体，文字可能无法正常显示")
            self.pil_font = ImageFont.load_default()
            self.pil_font_small = self.pil_font

        # 预计算每个字符的尺寸
        self.char_sizes = []
        total_w = 0
        max_h = 0
        for ch in self.text:
            bbox = self.pil_font.getbbox(ch)
            cw = bbox[2] - bbox[0]
            ch_h = bbox[3] - bbox[1]
            self.char_sizes.append((cw, ch_h))
            total_w += cw + self.char_spacing
            max_h = max(max_h, ch_h)

        self.text_width = total_w
        self.text_height = max_h
        self.group_width = total_w

        # 预渲染每个字符的 PIL 图像（原始大小）
        self.char_images = []
        for ci, ch in enumerate(self.text):
            cw, ch_h = self.char_sizes[ci]
            # 创建带透明通道的字符图像
            pad = 20
            img = Image.new("RGBA", (cw + pad * 2, ch_h + pad * 2), (0, 0, 0, 0))
            draw = ImageDraw.Draw(img)
            draw.text((pad, pad), ch, font=self.pil_font, fill=(255, 255, 255, 255))
            self.char_images.append((img, pad))

    def activate(self):
        if not self.active:
            self.active = True
            self.activate_time = time.time()

    def deactivate(self):
        self.active = False

    def draw(self, frame, current_time):
        """绘制滚动3D旋转文字。"""
        if self.active:
            age = current_time - self.activate_time
            self.transition_alpha = min(1.0, age / self.fade_duration)
        else:
            self.transition_alpha = max(0.0, self.transition_alpha - 0.05)

        if self.transition_alpha <= 0.01:
            return frame

        h, w = frame.shape[:2]
        band_top = 0
        band_height = h // 4
        band_center_y = band_top + band_height // 2

        alpha = self.transition_alpha

        # 半透明背景条
        bg_bar = np.zeros_like(frame)
        bg_bar[band_top:band_top + band_height, :, :] = (20, 5, 10)
        frame = cv2.addWeighted(bg_bar, 0.35 * alpha, frame, 1.0, 0)

        # 计算滚动偏移
        elapsed = current_time - self.activate_time if self.active else 0
        scroll_offset = (self.scroll_speed * elapsed) % self.group_width

        # 3D旋转角度
        base_angle = math.sin(2 * math.pi * self.rotation_speed * elapsed) * 0.5

        # 需要绘制的组数
        num_groups = (w // self.group_width) + 2

        for i in range(num_groups):
            group_x = -scroll_offset + i * self.group_width
            x_cursor = group_x

            for ci, ch in enumerate(self.text):
                cw, ch_h = self.char_sizes[ci]

                char_phase = base_angle + ci * 0.35
                scale_x = abs(math.cos(char_phase))
                effective_width = max(cw * scale_x, cw * 0.3) + self.char_spacing

                char_center_x = x_cursor + cw // 2
                if -cw * 2 < char_center_x < w + cw * 2:
                    self._draw_3d_char_pil(
                        frame, ci, int(x_cursor), band_center_y,
                        cw, ch_h, char_phase, alpha
                    )

                x_cursor += effective_width

        return frame

    def _draw_3d_char_pil(self, frame, char_idx, x, y, cw, ch_h, angle, alpha):
        """用 PIL 渲染带3D旋转效果的单个中文字符。"""
        cos_a = math.cos(angle)
        scale_x = abs(cos_a)

        if scale_x < 0.08:
            return

        char_img, pad = self.char_images[char_idx]

        # 根据3D缩放调整图像大小
        new_w = max(1, int(char_img.width * scale_x))
        new_h = char_img.height

        # 使用高质量缩放
        resized = char_img.resize((new_w, new_h), Image.LANCZOS)

        # 颜色（彩虹渐变）
        hue = (angle + math.pi) / (2 * math.pi)
        r = int(255 * (0.5 + 0.5 * math.sin(2 * math.pi * hue)))
        g = int(255 * (0.5 + 0.5 * math.sin(2 * math.pi * hue + 2.094)))
        b = int(255 * (0.5 + 0.5 * math.sin(2 * math.pi * hue + 4.189)))

        # 给字符上色
        colored = self._colorize_char(resized, (r, g, b))

        # 计算放置位置（居中）
        paste_x = x + (cw - new_w) // 2
        paste_y = y - new_h // 2

        # 边界检查
        fh, fw = frame.shape[:2]
        if paste_x + new_w < 0 or paste_x >= fw:
            return

        # 将 PIL 图像混合到 OpenCV 帧
        self._overlay_pil_rgba(frame, colored, paste_x, paste_y, alpha)

    def _colorize_char(self, char_img, color):
        """给白色字符图像上色。"""
        r, g, b = color
        # 分离通道
        channels = list(char_img.split())
        if len(channels) < 4:
            return char_img
        # R, G, B, A
        # 将白色区域替换为目标颜色
        data = np.array(char_img)
        mask = data[:, :, 3] > 0
        result = np.zeros_like(data)
        result[mask, 0] = b  # OpenCV BGR
        result[mask, 1] = g
        result[mask, 2] = r
        result[mask, 3] = data[mask, 3]
        return Image.fromarray(result, "RGBA")

    def _overlay_pil_rgba(self, frame, pil_img, x, y, alpha):
        """将 PIL RGBA 图像叠加到 OpenCV BGR 帧上。"""
        fh, fw = frame.shape[:2]
        img_w, img_h = pil_img.size

        # 裁剪到帧范围内
        src_x1 = max(0, -x)
        src_y1 = max(0, -y)
        src_x2 = min(img_w, fw - x)
        src_y2 = min(img_h, fh - y)

        if src_x1 >= src_x2 or src_y1 >= src_y2:
            return

        dst_x1 = max(0, x)
        dst_y1 = max(0, y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        # 获取 PIL 图像的 numpy 数组
        img_data = np.array(pil_img)
        roi = img_data[src_y1:src_y2, src_x1:src_x2]

        # Alpha 通道
        a = roi[:, :, 3].astype(float) / 255.0 * alpha
        a = a[:, :, np.newaxis]

        # BGR 通道（PIL 是 RGBA，OpenCV 是 BGR）
        char_bgr = roi[:, :, [2, 1, 0]].astype(float)
        frame_roi = frame[dst_y1:dst_y2, dst_x1:dst_x2].astype(float)

        # Alpha 混合
        blended = char_bgr * a + frame_roi * (1 - a)
        frame[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)


# ============================================================
# FlameParticle - 火焰粒子（增强版）
# ============================================================
@dataclass
class FlameParticle:
    """火焰粒子，支持拖尾轨迹。"""
    x: float
    y: float
    size: float
    color: tuple
    birth_time: float
    velocity_y: float
    velocity_x: float
    lifetime: float
    flicker_speed: float
    flicker_offset: float
    trail: list = None
    gravity: float = 0.0       # 重力加速度（火星飞溅用）

    def __post_init__(self):
        if self.trail is None:
            self.trail = []


# ============================================================
# SmokeParticle - 烟雾粒子
# ============================================================
@dataclass
class SmokeParticle:
    """烟雾粒子，在火焰消失位置产生。"""
    x: float
    y: float
    size: float
    birth_time: float
    velocity_y: float
    velocity_x: float
    lifetime: float
    alpha_base: float = 0.3


# ============================================================
# FistFlameEffect - 拳头格斗火焰特效（增强版）
# ============================================================
class FistFlameEffect:
    """
    拳头检测时的格斗气场特效。
    包含：真实火焰粒子、烟雾、火星飞溅、气场光环、能量辐射、热浪畸变。
    """

    def __init__(self):
        self.active = False
        self.activate_time = 0.0
        self.fade_duration = 0.4
        self.transition_alpha = 0.0
        self.flames = []
        self.max_flames = 120
        self.smoke = []
        self.max_smoke = 50
        self.sparks = []
        self.max_sparks = 40
        self.aura_particles = []
        self.max_aura = 60

    def activate(self):
        if not self.active:
            self.active = True
            self.activate_time = time.time()
            self.flames.clear()
            self.smoke.clear()
            self.sparks.clear()
            self.aura_particles.clear()

    def deactivate(self):
        self.active = False

    def update(self, dt, frame_width, frame_height, hand_landmarks_list=None):
        """更新所有粒子。"""
        if self.active:
            age = time.time() - self.activate_time
            self.transition_alpha = min(1.0, age / self.fade_duration)
        else:
            self.transition_alpha = max(0.0, self.transition_alpha - dt * 3.0)

        if self.transition_alpha <= 0.01:
            self.flames.clear()
            self.smoke.clear()
            self.sparks.clear()
            self.aura_particles.clear()
            return

        current_time = time.time()

        # 获取手掌中心
        palm_x, palm_y = frame_width // 2, frame_height // 2
        if hand_landmarks_list:
            for hand in hand_landmarks_list:
                lm = hand.landmark
                palm_x = int((lm[0].x + lm[9].x) / 2 * frame_width)
                palm_y = int((lm[0].y + lm[9].y) / 2 * frame_height)
                break

        # ---- 生成火焰粒子（真实火焰：多层椭圆，亮黄→暗红） ----
        if len(self.flames) < self.max_flames:
            for _ in range(5):
                spread = 25
                self.flames.append(FlameParticle(
                    x=palm_x + random.uniform(-spread, spread),
                    y=palm_y + random.uniform(-5, 5),
                    size=random.uniform(5, 14),
                    color=random.choice([
                        (0, 60, 255),    # 深红
                        (0, 120, 255),   # 橙红
                        (0, 200, 255),   # 橙色
                        (0, 255, 255),   # 亮黄
                        (30, 180, 255),  # 浅橙
                    ]),
                    birth_time=current_time,
                    velocity_y=random.uniform(-150, -300),
                    velocity_x=random.uniform(-30, 30),
                    lifetime=random.uniform(0.4, 1.0),
                    flicker_speed=random.uniform(8, 18),
                    flicker_offset=random.uniform(0, 2 * math.pi),
                ))

        # ---- 生成火星飞溅 ----
        if len(self.sparks) < self.max_sparks and random.random() < 0.6:
            for _ in range(2):
                angle = random.uniform(-math.pi * 0.8, -math.pi * 0.2)
                speed = random.uniform(200, 400)
                self.sparks.append(FlameParticle(
                    x=palm_x + random.uniform(-20, 20),
                    y=palm_y + random.uniform(-10, 10),
                    size=random.uniform(1.5, 3.5),
                    color=random.choice([
                        (0, 255, 255),   # 亮黄
                        (200, 255, 255), # 白黄
                        (0, 200, 255),   # 橙色
                    ]),
                    birth_time=current_time,
                    velocity_x=math.cos(angle) * speed,
                    velocity_y=math.sin(angle) * speed,
                    lifetime=random.uniform(0.3, 0.7),
                    flicker_speed=random.uniform(15, 25),
                    flicker_offset=random.uniform(0, 2 * math.pi),
                    gravity=400,  # 重力
                ))

        # ---- 生成气场环绕粒子 ----
        if len(self.aura_particles) < self.max_aura:
            for _ in range(2):
                angle = random.uniform(0, 2 * math.pi)
                radius = random.uniform(60, 130)
                self.aura_particles.append(FlameParticle(
                    x=palm_x + math.cos(angle) * radius,
                    y=palm_y + math.sin(angle) * radius,
                    size=random.uniform(3, 9),
                    color=random.choice([
                        (0, 40, 180),
                        (0, 80, 220),
                        (20, 20, 180),
                    ]),
                    birth_time=current_time,
                    velocity_y=random.uniform(-60, -130),
                    velocity_x=random.uniform(-25, 25),
                    lifetime=random.uniform(0.5, 1.3),
                    flicker_speed=random.uniform(5, 12),
                    flicker_offset=random.uniform(0, 2 * math.pi),
                ))

        # ---- 更新火焰粒子 ----
        alive_flames = []
        dead_positions = []
        for f in self.flames:
            f_age = current_time - f.birth_time
            if f_age > f.lifetime:
                dead_positions.append((f.x, f.y))
                continue
            # 记录拖尾
            f.trail.append((f.x, f.y, current_time))
            if len(f.trail) > 5:
                f.trail.pop(0)
            f.y += f.velocity_y * dt
            f.x += f.velocity_x * dt
            f.x += math.sin(f_age * f.flicker_speed + f.flicker_offset) * 2
            alive_flames.append(f)
        self.flames = alive_flames

        # ---- 在火焰消失位置生成烟雾 ----
        for px, py in dead_positions:
            if len(self.smoke) < self.max_smoke:
                self.smoke.append(SmokeParticle(
                    x=px + random.uniform(-10, 10),
                    y=py,
                    size=random.uniform(8, 18),
                    birth_time=current_time,
                    velocity_y=random.uniform(-40, -80),
                    velocity_x=random.uniform(-15, 15),
                    lifetime=random.uniform(0.8, 1.5),
                    alpha_base=random.uniform(0.15, 0.30),
                ))

        # ---- 更新烟雾粒子 ----
        alive_smoke = []
        for s in self.smoke:
            s_age = current_time - s.birth_time
            if s_age > s.lifetime:
                continue
            s.y += s.velocity_y * dt
            s.x += s.velocity_x * dt
            s.size += dt * 6  # 烟雾扩散
            alive_smoke.append(s)
        self.smoke = alive_smoke

        # ---- 更新火星飞溅（抛物线轨迹） ----
        alive_sparks = []
        for s in self.sparks:
            s_age = current_time - s.birth_time
            if s_age > s.lifetime:
                continue
            s.trail.append((s.x, s.y, current_time))
            if len(s.trail) > 6:
                s.trail.pop(0)
            s.x += s.velocity_x * dt
            s.y += s.velocity_y * dt
            s.velocity_y += s.gravity * dt  # 重力
            alive_sparks.append(s)
        self.sparks = alive_sparks

        # ---- 更新气场粒子 ----
        alive_aura = []
        for a in self.aura_particles:
            a_age = current_time - a.birth_time
            if a_age > a.lifetime:
                continue
            a.y += a.velocity_y * dt
            a.x += a.velocity_x * dt
            alive_aura.append(a)
        self.aura_particles = alive_aura

    def draw(self, frame, hand_landmarks_list=None):
        """绘制增强火焰特效。"""
        if self.transition_alpha <= 0.01:
            return frame

        current_time = time.time()
        alpha = self.transition_alpha
        h, w = frame.shape[:2]

        # 获取手掌中心
        palm_x, palm_y = w // 2, h // 2
        if hand_landmarks_list:
            for hand in hand_landmarks_list:
                lm = hand.landmark
                palm_x = int((lm[0].x + lm[9].x) / 2 * w)
                palm_y = int((lm[0].y + lm[9].y) / 2 * h)
                break

        overlay = np.zeros_like(frame)

        # ---- 1. 红色气场光环（脉冲呼吸） ----
        aura_radius = int(110 + 25 * math.sin(2 * math.pi * 3 * current_time))
        for i in range(4):
            r = aura_radius - i * 18
            if r > 0:
                thickness = 4 - i
                color = (10 - i * 2, 35 - i * 8, 130 - i * 25)
                cv2.circle(overlay, (palm_x, palm_y), r, color, max(1, thickness))
        # 外层柔光
        glow_r = aura_radius + 40
        glow_layer = np.zeros_like(frame)
        cv2.circle(glow_layer, (palm_x, palm_y), glow_r, (5, 15, 60), -1)
        frame = cv2.addWeighted(glow_layer, 0.15 * alpha, frame, 1.0, 0)

        # ---- 2. 真实火焰粒子（多层绘制） ----
        for f in self.flames:
            f_age = current_time - f.birth_time
            f_alpha = 1.0 - (f_age / f.lifetime)
            if f_alpha <= 0:
                continue

            flicker = 0.6 + 0.4 * math.sin(f.flicker_speed * f_age + f.flicker_offset)
            r = max(1, int(f.size * flicker * f_alpha))
            center = (int(f.x), int(f.y))

            # 绘制拖尾
            for ti, (tx, ty, tt) in enumerate(f.trail):
                t_alpha = (ti / max(1, len(f.trail))) * 0.3 * f_alpha
                if t_alpha < 0.02:
                    continue
                tr = max(1, int(r * 0.5 * ti / max(1, len(f.trail))))
                t_color = tuple(int(c * t_alpha) for c in f.color)
                cv2.ellipse(overlay, (int(tx), int(ty)), (tr, int(tr * 1.4)),
                            0, 0, 360, t_color, -1)

            # 外层火焰（暗色，较大）
            outer_color = tuple(int(c * f_alpha * 0.5) for c in f.color)
            cv2.ellipse(overlay, center, (r, int(r * 1.6)), 0, 0, 360, outer_color, -1)

            # 主火焰层
            color = tuple(int(c * f_alpha * flicker) for c in f.color)
            cv2.ellipse(overlay, center, (r, int(r * 1.5)), 0, 0, 360, color, -1)

            # 内层火焰核心（更亮更小）
            if r > 2:
                inner_r = max(1, r // 2)
                inner_color = (
                    min(255, color[0] + 50),
                    min(255, color[1] + 50),
                    min(255, color[2] + 50),
                )
                cv2.ellipse(overlay, center, (inner_r, int(inner_r * 1.3)),
                            0, 0, 360, inner_color, -1)

            # 最亮中心白点
            if r > 4:
                white_r = max(1, r // 3)
                cv2.ellipse(overlay, center, (white_r, int(white_r * 1.2)),
                            0, 0, 360, (200, 255, 255), -1)

        # ---- 3. 烟雾粒子 ----
        smoke_layer = np.zeros_like(frame)
        for s in self.smoke:
            s_age = current_time - s.birth_time
            s_alpha = (1.0 - s_age / s.lifetime) * s.alpha_base
            if s_alpha <= 0:
                continue
            r = int(s.size)
            if r < 1:
                continue
            gray = int(80 + 40 * (1.0 - s_age / s.lifetime))
            color = (gray, gray, gray)
            cv2.circle(smoke_layer, (int(s.x), int(s.y)), r, color, -1)
        # 高斯模糊让烟雾更柔和
        if np.any(smoke_layer > 0):
            smoke_layer = cv2.GaussianBlur(smoke_layer, (15, 15), 5)
            frame = cv2.addWeighted(smoke_layer, 0.4 * alpha, frame, 1.0, 0)

        # ---- 4. 火星飞溅（短拖尾） ----
        for s in self.sparks:
            s_age = current_time - s.birth_time
            s_alpha = 1.0 - (s_age / s.lifetime)
            if s_alpha <= 0:
                continue
            r = max(1, int(s.size * s_alpha))

            # 绘制拖尾线
            if len(s.trail) >= 2:
                for ti in range(1, len(s.trail)):
                    t_alpha = (ti / len(s.trail)) * s_alpha
                    t_color = tuple(int(c * t_alpha) for c in s.color)
                    pt1 = (int(s.trail[ti - 1][0]), int(s.trail[ti - 1][1]))
                    pt2 = (int(s.trail[ti][0]), int(s.trail[ti][1]))
                    cv2.line(overlay, pt1, pt2, t_color, max(1, r))

            # 火星本体
            color = tuple(int(c * s_alpha) for c in s.color)
            cv2.circle(overlay, (int(s.x), int(s.y)), r, color, -1)
            # 发光效果
            cv2.circle(overlay, (int(s.x), int(s.y)), r + 2,
                        tuple(c // 3 for c in s.color), -1)

        # ---- 5. 气场环绕粒子 ----
        for a in self.aura_particles:
            a_age = current_time - a.birth_time
            a_alpha = 1.0 - (a_age / a.lifetime)
            if a_alpha <= 0:
                continue
            flicker = 0.5 + 0.5 * math.sin(a.flicker_speed * a_age + a.flicker_offset)
            r = max(1, int(a.size * flicker * a_alpha))
            color = tuple(int(c * a_alpha) for c in a.color)
            cv2.circle(overlay, (int(a.x), int(a.y)), r, color, -1)

        # ---- 6. 能量辐射线条 ----
        num_rays = 10
        for i in range(num_rays):
            angle = (2 * math.pi / num_rays) * i + current_time * 2.5
            ray_len = 90 + 35 * math.sin(3 * current_time + i * 0.7)
            end_x = int(palm_x + math.cos(angle) * ray_len)
            end_y = int(palm_y + math.sin(angle) * ray_len)
            # 渐变线：根部亮，末端暗
            color = (0, int(70 + 50 * math.sin(angle + current_time)), 200)
            cv2.line(overlay, (palm_x, palm_y), (end_x, end_y), color, 2, cv2.LINE_AA)
            # 末端光点
            cv2.circle(overlay, (end_x, end_y), 3,
                        (100, 200, 255), -1)

        # 混合到画面
        mask = np.any(overlay > 0, axis=2)
        if np.any(mask):
            frame = cv2.addWeighted(overlay, 0.85 * alpha, frame, 1.0, 0)

        return frame


# ============================================================
# BackgroundEffect - 动态背景响应
# ============================================================
class BackgroundEffect:
    """
    手势触发时的背景氛围效果。
    包含：色相偏移、暗角渐变、镜头呼吸缩放。
    """

    def __init__(self):
        pass

    def apply(self, frame, gesture_type, dt):
        """
        对画面应用背景效果。
        gesture_type: 0=无, 1=单手比心, 2=双手比心, 3=拳头
        """
        h, w = frame.shape[:2]

        # 1. 色相/色调偏移
        if gesture_type == 3:
            # 拳头模式：暗红色滤镜，降低亮度
            frame = self._apply_color_tint(frame, b_shift=-15, g_shift=-25, r_shift=15, brightness=0.85)
        elif gesture_type == 2:
            # 双手比心：暖色偏移
            frame = self._apply_color_tint(frame, b_shift=-10, g_shift=5, r_shift=15, brightness=1.0)
        elif gesture_type == 1:
            # 单手比心：轻微暖色
            frame = self._apply_color_tint(frame, b_shift=-5, g_shift=3, r_shift=8, brightness=1.0)

        # 2. 暗角效果
        if gesture_type in (1, 2, 3):
            frame = self._apply_vignette(frame, strength=0.5 if gesture_type == 2 else 0.3)

        return frame

    @staticmethod
    def _apply_color_tint(frame, b_shift, g_shift, r_shift, brightness):
        """应用色调偏移和亮度调整。"""
        result = frame.astype(np.float32)
        result[:, :, 0] = np.clip(result[:, :, 0] + b_shift, 0, 255)
        result[:, :, 1] = np.clip(result[:, :, 1] + g_shift, 0, 255)
        result[:, :, 2] = np.clip(result[:, :, 2] + r_shift, 0, 255)
        result *= brightness
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def _apply_vignette(frame, strength=0.4):
        """应用暗角渐变效果。"""
        h, w = frame.shape[:2]
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_dist = math.sqrt(cx ** 2 + cy ** 2)
        if max_dist < 1:
            return frame
        vignette = (dist / max_dist) ** 2
        vignette = np.clip(vignette - 0.3, 0, 1.0)
        vignette = np.clip(vignette * strength * 255, 0, 255).astype(np.uint8)
        vignette_3ch = np.stack([vignette] * 3, axis=2)
        return cv2.subtract(frame, vignette_3ch)


# ============================================================
# 主函数
# ============================================================
def main():
    # 初始化手势分类器
    classifier = GestureClassifier(GESTURE_MODEL_PATH)
    detector = HeartLandmarkDetector(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, classifier)
    engine = HeartAnimationEngine(MAX_HEARTS, HEART_SPAWN_RATE)
    double_effect = DoubleHeartEffect()
    fist_effect = FistFlameEffect()
    text_effect = ScrollingText3D("My lover")
    bg_effect = BackgroundEffect()

    try:
        detector.start()
    except RuntimeError as e:
        print(f"错误: {e}")
        return

    # 创建全屏窗口
    window_name = "Heart Gesture Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    prev_time = time.time()

    while True:
        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        # 1. 读取摄像头画面
        success, frame = detector.read_frame()
        if not success:
            print("无法读取摄像头画面")
            break

        # 2. 水平翻转（镜像效果）
        frame = cv2.flip(frame, 1)

        # 3. 检测手势（返回手势类型）
        gesture_type, landmarks_list = detector.detect_gesture(frame)

        # 3.5 动态背景效果（色相偏移、暗角、呼吸缩放）
        frame = bg_effect.apply(frame, gesture_type, dt)

        # 4. 根据手势类型处理不同模式
        if gesture_type == 2:
            # 双手比心模式
            double_effect.activate()
            text_effect.activate()
            fist_effect.deactivate()
            engine.spawn_hearts(HEART_SPAWN_RATE, FRAME_WIDTH, FRAME_HEIGHT)
        elif gesture_type == 1:
            # 单手比心模式
            double_effect.deactivate()
            text_effect.activate()
            fist_effect.deactivate()
            engine.spawn_hearts(HEART_SPAWN_RATE, FRAME_WIDTH, FRAME_HEIGHT)
        elif gesture_type == 3:
            # 拳头模式
            double_effect.deactivate()
            text_effect.deactivate()
            fist_effect.activate()
        else:
            double_effect.deactivate()
            text_effect.deactivate()
            fist_effect.deactivate()

        # 5. 更新特效和粒子
        double_effect.update(dt, FRAME_WIDTH, FRAME_HEIGHT)
        fist_effect.update(dt, FRAME_WIDTH, FRAME_HEIGHT, landmarks_list)
        engine.update(dt)

        # 6. 绘制特效到画面
        # 绘制火焰特效
        if fist_effect.active or fist_effect.transition_alpha > 0.01:
            frame = fist_effect.draw(frame, landmarks_list)

        # 双手比心全屏特效
        if double_effect.active or double_effect.transition_alpha > 0.01:
            frame = double_effect.draw(frame)

        # 绘制小爱心粒子
        frame = engine.draw(frame)

        # 绘制滚动3D文字（在顶部1/4区域）
        frame = text_effect.draw(frame, current_time)

        # 7. 显示状态文字（根据画面大小调整字号）
        font_scale = FRAME_WIDTH / 800.0
        thickness = max(2, int(font_scale * 2))
        if gesture_type == 2:
            status = "Double Heart! <3"
            color = (0, 0, 255)
        elif gesture_type == 1:
            status = "Heart gesture detected!"
            color = (0, 255, 0)
        elif gesture_type == 3:
            status = "FIST! Power Up!"
            color = (0, 100, 255)
        else:
            status = "Show me a heart gesture or fist..."
            color = (200, 200, 200)
        cv2.putText(frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness)

        # 8. 全屏显示画面
        cv2.imshow(window_name, frame)

        # 9. 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q 或 ESC 退出
            break
        elif key == ord('s'):  # s 保存截图
            screenshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
            os.makedirs(screenshot_dir, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            gesture_name = {0: "none", 1: "heart", 2: "double_heart", 3: "fist"}.get(gesture_type, "unknown")
            filename = f"{ts}_{gesture_name}.png"
            filepath = os.path.join(screenshot_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"截图已保存: {filepath}")

    detector.stop()
    cv2.destroyAllWindows()
    print("程序已退出")


if __name__ == "__main__":
    main()
