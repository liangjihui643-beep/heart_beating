"""
手势分类模型训练脚本

读取 collect_data.py 录制的数据，训练轻量神经网络，导出为 TFLite 模型。
支持三种手势类别：单手比心(heart)、双手比心(double_heart)、拳头(fist)

用法：
  .venv_train\Scripts\python.exe train_model.py

输出：
  gesture_classifier.tflite — 可被 beating_heart.py 加载的模型文件
"""

import json
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ============================================================
# 配置
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), "gesture_data")
MODEL_OUTPUT = os.path.join(os.path.dirname(__file__), "gesture_classifier.tflite")

# 三类别：单手比心、双手比心、拳头
LABEL_MAP = {"heart": 0, "double_heart": 1, "fist": 2}
NUM_CLASSES = 3
INPUT_DIM = 126  # 42 landmarks * 3 (双手，单手数据填充零向量)

EPOCHS = 50
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2


# ============================================================
# 数据加载
# ============================================================
def load_data():
    """加载 gesture_data/ 目录下的所有类别数据。"""
    features = []
    labels = []

    for label_name, label_id in LABEL_MAP.items():
        label_dir = os.path.join(DATA_DIR, label_name)
        if not os.path.exists(label_dir):
            print(f"警告：目录不存在 {label_dir}，跳过")
            continue

        files = [f for f in os.listdir(label_dir) if f.endswith(".json")]
        print(f"  {label_name} (label={label_id}): {len(files)} 个样本")

        for filename in files:
            filepath = os.path.join(label_dir, filename)
            with open(filepath, "r") as f:
                data = json.load(f)

            landmarks = np.array(data["landmarks"])

            if label_name == "double_heart":
                # 双手数据：42个关键点 = 126维
                if landmarks.shape[0] != 42:
                    print(f"  跳过非双手数据: {filename} (关键点数: {landmarks.shape[0]})")
                    continue
                vector = normalize_double_hand(landmarks)
            else:
                # 单手数据：21个关键点 = 63维，填充到126维
                if landmarks.shape[0] != 21:
                    print(f"  跳过非单手数据: {filename} (关键点数: {landmarks.shape[0]})")
                    continue
                vector = normalize_single_hand(landmarks)

            features.append(vector)
            labels.append(label_id)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


def normalize_single_hand(landmarks):
    """
    归一化单手关键点，填充到126维。
    前63维：归一化后的左手数据
    后63维：全零（无第二只手）
    """
    wrist = landmarks[0]
    centered = landmarks - wrist
    palm_size = np.linalg.norm(centered[9])
    if palm_size < 1e-6:
        palm_size = 1.0
    normalized = (centered / palm_size).flatten()  # (63,)
    # 填充到126维
    return np.concatenate([normalized, np.zeros(63, dtype=np.float32)])


def normalize_double_hand(landmarks):
    """
    归一化双手关键点。
    landmarks: (42, 3) — 前21点为左手，后21点为右手
    分别对每只手做归一化，拼接为126维。
    """
    left_hand = landmarks[:21]
    right_hand = landmarks[21:]

    left_vector = normalize_hand_vector(left_hand)
    right_vector = normalize_hand_vector(right_hand)

    return np.concatenate([left_vector, right_vector])


def normalize_hand_vector(landmarks):
    """归一化单只手的关键点为63维向量。"""
    wrist = landmarks[0]
    centered = landmarks - wrist
    palm_size = np.linalg.norm(centered[9])
    if palm_size < 1e-6:
        palm_size = 1.0
    return (centered / palm_size).flatten()


# ============================================================
# 数据增强
# ============================================================
def augment_data(features, labels, augment_ratio=3):
    """数据增强：随机抖动、缩放。"""
    augmented_features = [features]
    augmented_labels = [labels]

    for _ in range(augment_ratio):
        # 随机抖动
        noise = np.random.normal(0, 0.02, features.shape).astype(np.float32)
        augmented_features.append(features + noise)
        augmented_labels.append(labels)

        # 随机缩放
        scale = np.random.uniform(0.9, 1.1)
        augmented_features.append(features * scale)
        augmented_labels.append(labels)

    return np.concatenate(augmented_features), np.concatenate(augmented_labels)


# ============================================================
# 模型构建
# ============================================================
def build_model():
    """构建轻量全连接网络。"""
    model = keras.Sequential([
        keras.layers.Input(shape=(INPUT_DIM,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ============================================================
# 导出 TFLite
# ============================================================
def export_tflite(model, output_path):
    """导出为 TFLite 模型。"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    print(f"\n模型已导出: {output_path}")
    print(f"模型大小: {len(tflite_model) / 1024:.1f} KB")


# ============================================================
# 主函数
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("手势分类模型训练")
    print("=" * 50)
    print(f"类别映射: {LABEL_MAP}")
    print(f"输入维度: {INPUT_DIM}")
    print()

    # 加载数据
    print("[1/4] 加载数据...")
    X, y = load_data()
    print(f"总样本数: {len(X)}")

    if len(X) < 10:
        print("\n错误：样本数太少，请先用 collect_data.py 录制更多数据")
        print("建议每个类别至少 50 个样本")
        exit(1)

    # 检查类别平衡
    for label_name, label_id in LABEL_MAP.items():
        count = int(np.sum(y == label_id))
        print(f"  {label_name}: {count} 个样本")
        if count < 5:
            print(f"  警告：{label_name} 样本太少，建议至少 50 个")

    # 数据增强
    print("\n[2/4] 数据增强...")
    X_train, y_train = augment_data(X, y, augment_ratio=3)
    print(f"增强后样本数: {len(X_train)}")

    # 构建模型
    print("\n[3/4] 构建并训练模型...")
    model = build_model()
    model.summary()

    model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        verbose=1,
    )

    # 评估
    loss, acc = model.evaluate(X, y, verbose=0)
    print(f"\n训练集准确率: {acc:.4f}")

    # 导出
    print("\n[4/4] 导出 TFLite 模型...")
    export_tflite(model, MODEL_OUTPUT)

    print("\n完成！")
    print("现在可以运行 python beating_heart.py 来使用模型")
