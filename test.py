"""
DL-SCA: Deep Learning Side-Channel Attack against AES (CNN + Kubota Mixed Dataset)

論文參考:
    "Deep Learning Side-Channel Attack against Hardware Implementations of AES"
    Takaya Kubota, Kota Yoshida, Mitsuru Shiozaki, Takeshi Fujino

架構說明:
    - 使用端到端 CNN 攻擊 ASCADv2 資料集 (STM32 with Masking)
    - 採用 Kubota Mixed Dataset 技巧：同一 Column 的 4 個 Byte 混合訓練，
      讓資料量擴充 4 倍，有效打破 Masking 保護
    - 攻擊目標：AES S-Box 輸出 (Identity Leakage Model)
    - 評估指標：Guessing Entropy (GE)、Success Rate (SR)
"""

import gc
import json
import os
import random

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.utils import Sequence, to_categorical

# =============================================================================
# 1. 常數
# =============================================================================

AES_SBOX = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16,
], dtype=np.uint8)

# =============================================================================
# 2. 環境初始化
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """固定所有亂數種子，確保實驗可重現。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def enable_gpu_memory_growth() -> None:
    """啟用 GPU 動態記憶體分配，避免 TensorFlow 一次佔用全部 VRAM。"""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# =============================================================================
# 3. 資料載入與前處理
# =============================================================================

def _remove_null_traces(
    traces: np.ndarray,
    plaintext: np.ndarray,
    key: np.ndarray,
    part_name: str = "unknown",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """移除全為零的無效 power trace，並印出移除數量。"""
    null_mask = np.all(traces == 0, axis=1)
    num_removed = int(np.sum(null_mask))
    if num_removed > 0:
        print(f"[{part_name}] 移除 null traces: {num_removed} 筆")
    keep_mask = ~null_mask
    return traces[keep_mask], plaintext[keep_mask], key[keep_mask]


def _check_metadata_alignment(
    traces: np.ndarray,
    plaintext: np.ndarray,
    key: np.ndarray,
    part_name: str = "unknown",
) -> None:
    """確認 traces、plaintext、key 三者長度一致，不一致時直接拋出例外。"""
    if not (len(traces) == len(plaintext) == len(key)):
        raise ValueError(
            f"[{part_name}] metadata 對齊失敗: "
            f"traces={len(traces)}, plaintext={len(plaintext)}, key={len(key)}"
        )
    print(f"[{part_name}] metadata 對齊檢查通過: {len(traces)} 筆")


def load_ascadv2_data(
    file_path: str,
    profiling_limit: int = 10000,
    attack_limit: int = 2000,
    val_ratio: float = 0.1,
    split_seed: int = 42,
) -> dict:
    """
    從 ASCADv2 HDF5 檔案載入 profiling 與 attack 資料。

    流程:
        1. 讀取 HDF5 → 清理 null traces → 對齊檢查
        2. 將 profiling 切成 train / val
        3. 以 train 集的統計量做 Z-score 正規化
        4. 增加 channel 維度（供 Conv1D 使用）

    Returns:
        包含 X_train, X_val, X_test 及對應 plaintext / key 的 dict。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到資料集檔案: {file_path}")

    with h5py.File(file_path, "r") as f:
        X_profiling = np.array(f["Profiling_traces/traces"][:profiling_limit], dtype=np.float32)
        pt_profiling = np.array(f["Profiling_traces/metadata"]["plaintext"][:profiling_limit], dtype=np.uint8)
        key_profiling = np.array(f["Profiling_traces/metadata"]["key"][:profiling_limit], dtype=np.uint8)

        X_attack = np.array(f["Attack_traces/traces"][:attack_limit], dtype=np.float32)
        pt_attack = np.array(f["Attack_traces/metadata"]["plaintext"][:attack_limit], dtype=np.uint8)
        key_attack = np.array(f["Attack_traces/metadata"]["key"][:attack_limit], dtype=np.uint8)

    # 清理與對齊檢查
    X_profiling, pt_profiling, key_profiling = _remove_null_traces(X_profiling, pt_profiling, key_profiling, "profiling")
    X_attack, pt_attack, key_attack = _remove_null_traces(X_attack, pt_attack, key_attack, "attack")
    _check_metadata_alignment(X_profiling, pt_profiling, key_profiling, "profiling")
    _check_metadata_alignment(X_attack, pt_attack, key_attack, "attack")

    # 切分 train / val
    train_idx, val_idx = train_test_split(
        np.arange(len(X_profiling)), test_size=val_ratio, random_state=split_seed, shuffle=True
    )
    X_train, pt_train, key_train = X_profiling[train_idx], pt_profiling[train_idx], key_profiling[train_idx]
    X_val, pt_val, key_val = X_profiling[val_idx], pt_profiling[val_idx], key_profiling[val_idx]

    # Z-score 正規化（以 train 集統計量為基準，避免 data leakage）
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_attack = (X_attack - mean) / std

    # 增加 channel 維度供 Conv1D 使用：(N, T) → (N, T, 1)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_attack = X_attack[..., np.newaxis]

    print(f"資料載入完成 | train: {len(X_train)}, val: {len(X_val)}, test: {len(X_attack)}")

    return {
        "X_train": X_train, "pt_train": pt_train, "key_train": key_train,
        "X_val": X_val,   "pt_val": pt_val,   "key_val": key_val,
        "X_test": X_attack, "pt_test": pt_attack, "key_test": key_attack,
    }

# =============================================================================
# 4. 標籤產生
# =============================================================================

def generate_sbox_labels(
    plaintext: np.ndarray,
    key: np.ndarray,
    target_byte: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    計算 AES S-Box 輸出作為攻擊標籤（Identity Leakage Model）。

    label = S-Box( plaintext[target_byte] XOR key[target_byte] )

    Returns:
        sbox_values: shape (N,)，整數類別 0–255
        one_hot:     shape (N, 256)，one-hot 編碼
    """
    sbox_values = AES_SBOX[plaintext[:, target_byte] ^ key[:, target_byte]]
    one_hot = to_categorical(sbox_values, num_classes=256)
    return sbox_values, one_hot

# =============================================================================
# 5. Mixed Dataset Generator（防 OOM，資料擴充 4 倍）
# =============================================================================

class MixedDataGenerator(Sequence):
    """
    Kubota Mixed Dataset 資料生成器。

    核心思路：
        AES 的 SubBytes 操作以 Column 為單位同時計算 4 個 Byte，
        這 4 個 Byte 的側信道洩漏高度相關。
        因此，將同一 Column 的 4 個 Byte 混合成訓練資料，
        相當於把資料量擴充 4 倍，同時幫助 CNN 學習更泛化的特徵。

    論文來源:
        Kubota et al., "Deep Learning Side-Channel Attack
        against Hardware Implementations of AES"
    """

    def __init__(
        self,
        traces: np.ndarray,
        plaintext: np.ndarray,
        key: np.ndarray,
        target_byte: int,
        batch_size: int = 128,
    ) -> None:
        self.traces = traces
        self.plaintext = plaintext
        self.key = key
        self.batch_size = batch_size

        # 找出與 target_byte 同屬一個 Column 的 4 個 Byte index
        column = target_byte // 4
        self.mixed_bytes = [column * 4 + i for i in range(4)]

        # 總樣本數 = 原始筆數 × 4（每筆 trace 對應 4 種 byte 標籤）
        self.total_samples = len(traces) * len(self.mixed_bytes)
        self.indices = np.arange(self.total_samples)
        np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(self.total_samples / self.batch_size))

    def __getitem__(self, batch_idx: int) -> tuple[np.ndarray, np.ndarray]:
        batch_indices = self.indices[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]

        # 將 virtual index 解碼為 trace index 與 byte selector
        trace_idx = batch_indices % len(self.traces)
        byte_selector = batch_indices // len(self.traces)   # 值域：0, 1, 2, 3

        batch_traces = self.traces[trace_idx]
        batch_pt = self.plaintext[trace_idx]
        batch_key = self.key[trace_idx]

        # 向量化計算每筆樣本對應的 byte index（避免 Python for loop）
        selected_bytes = np.array(self.mixed_bytes)[byte_selector]          # shape: (B,)
        pt_bytes  = batch_pt[np.arange(len(batch_indices)), selected_bytes] # shape: (B,)
        key_bytes = batch_key[np.arange(len(batch_indices)), selected_bytes] # shape: (B,)
        sbox_vals = AES_SBOX[pt_bytes ^ key_bytes]                           # shape: (B,)

        # 建立 one-hot 標籤
        batch_labels = np.zeros((len(batch_indices), 256), dtype=np.float32)
        batch_labels[np.arange(len(batch_indices)), sbox_vals] = 1.0

        return batch_traces, batch_labels

    def on_epoch_end(self) -> None:
        """每個 Epoch 結束後重新打亂，提升訓練多樣性。"""
        np.random.shuffle(self.indices)

# =============================================================================
# 6. CNN 模型建構
# =============================================================================

def _conv_block(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    pool_size: int = 4,
) -> tf.Tensor:
    """
    標準 CNN 特徵萃取 Block：Conv1D → BatchNorm → ReLU → MaxPooling。

    Args:
        x:           輸入 tensor
        filters:     卷積核數量
        kernel_size: 卷積核大小
        pool_size:   MaxPooling 縮減比例
    """
    x = layers.Conv1D(filters, kernel_size=kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=pool_size)(x)
    return x


# =============================================================================
# 3. 建立端到端深層網路 (移植自 ANSSI 官方 ASCADv2 ResNet 架構)
# =============================================================================

def resnet_layer(inputs, num_filters=16, kernel_size=11, strides=1, 
                 activation='relu', batch_normalization=True, conv_first=True):
    """ 從官方腳本完美移植的 ResNet 基礎區塊 """
    conv = layers.Conv1D(num_filters,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding='same',
                         kernel_initializer='he_normal')
    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
    else:
        if batch_normalization:
            x = layers.BatchNormalization()(x)
        if activation is not None:
            x = layers.Activation(activation)(x)
        x = conv(x)
    return x

# def build_resnet_model(input_shape, depth=19, num_classes=256):
def build_resnet_model(input_shape, depth=19, num_classes=256, learning_rate=2e-4, dropout_rate=0.0):
    """ 
    改編自官方的 resnet_v1
    保留了對抗掩碼的超深層殘差結構，但將尾端收束為適應我們專案的單一 256 分類輸出
    """
    if (depth - 1) % 18 != 0:
        raise ValueError('depth 必須是 18n+1 (例如 19, 37, 55...)')
    
    num_filters = 16
    num_res_blocks = int((depth - 1) / 18)
    inputs = layers.Input(shape=input_shape)
    
    # 第一層卷積
    x = resnet_layer(inputs=inputs)
    
    # 堆疊殘差區塊 (Residual Blocks)
    for stack in range(9):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:
                strides = 2  # 透過 stride 進行降維，取代 MaxPooling
            
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)
            
            if stack > 0 and res_block == 0:
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, 
                                 strides=strides, activation=None, batch_normalization=False)
            
            # 殘差網路的靈魂：跳躍連接 (Skip Connection)
            x = layers.add([x, y])
            x = layers.Activation('relu')(x)
            
        if num_filters < 256:
            num_filters *= 2
            
    # 全域特徵壓縮
    x = layers.AveragePooling1D(pool_size=4)(x)
    x = layers.Flatten()(x)

    # 加上 Dropout 參數
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    
    # 單任務輸出層 (取代官方的多任務分支)
    x = layers.Dense(1024, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    
    model = models.Model(inputs, outputs, name='ASCADv2_Official_ResNet_Adapted')
    
    # 使用 Adam 優化器
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top_5_acc")]
    )
    return model

# =============================================================================
# 7. 攻擊評估：Key Recovery、GE、SR
# =============================================================================

def _rank_of_true_key(log_scores: np.ndarray, true_key: int) -> int:
    """計算 true key 在所有 256 個候選 key 中的排名（0 = 最高分，表示攻擊成功）。"""
    sorted_desc = np.argsort(log_scores)[::-1]
    return int(np.where(sorted_desc == true_key)[0][0])


def recover_key_log_rank(
    pred_probs: np.ndarray,
    plaintexts: np.ndarray,
    true_key: int,
    target_byte: int = 0,
) -> tuple[int, np.ndarray, np.ndarray]:
    """
    使用 log-likelihood 累加法從預測機率中恢復 AES key。

    對每筆 trace，計算 256 個 key 假設對應的 log-likelihood，
    累加後 argmax 即為最可能的 key byte。

    Returns:
        recovered_key:  預測的 key byte（0–255）
        log_key_scores: 每個 key 假設的累積 log-likelihood，shape (256,)
        rank_evolution: 每新增一筆 trace 後 true key 的排名，shape (N,)
    """
    eps = 1e-36  # 防止 log(0)
    log_key_scores = np.zeros(256, dtype=np.float64)
    rank_evolution = []

    key_hypotheses = np.arange(256, dtype=np.uint8)

    for i in range(len(pred_probs)):
        pt_byte = plaintexts[i, target_byte]
        sbox_vals = AES_SBOX[key_hypotheses ^ pt_byte]
        log_key_scores += np.log(pred_probs[i, sbox_vals] + eps)
        rank_evolution.append(_rank_of_true_key(log_key_scores, true_key))

    recovered_key = int(np.argmax(log_key_scores))
    return recovered_key, log_key_scores, np.array(rank_evolution, dtype=np.int32)


def compute_ge_sr(
    pred_probs: np.ndarray,
    plaintexts: np.ndarray,
    true_key: int,
    target_byte: int = 0,
    num_attacks: int = 20,
    max_traces: int = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    以 Monte Carlo 方式計算 Guessing Entropy (GE) 與 Success Rate (SR)。

    重複 num_attacks 次隨機排列攻擊 trace，取排名的平均值 (GE) 與
    排名為 0 的比例 (SR)，以得到穩健的統計估計。

    Returns:
        ge_curve: shape (max_traces,)，每條 trace 的平均排名
        sr_curve: shape (max_traces,)，每條 trace 的攻擊成功率
    """
    n = len(pred_probs)
    if max_traces is None or max_traces > n:
        max_traces = n

    rng = np.random.default_rng(seed)
    ge_curve = np.zeros(max_traces, dtype=np.float64)
    sr_curve = np.zeros(max_traces, dtype=np.float64)

    for _ in range(num_attacks):
        indices = rng.permutation(n)[:max_traces]
        _, _, rank_evolution = recover_key_log_rank(
            pred_probs[indices], plaintexts[indices], true_key, target_byte
        )
        ge_curve += rank_evolution
        sr_curve += (rank_evolution == 0).astype(np.float64)

    return ge_curve / num_attacks, sr_curve / num_attacks

# =============================================================================
# 8. 繪圖
# =============================================================================

def _plot_two_panels(
    left_data: tuple[object, np.ndarray, dict],
    right_data: tuple[object, np.ndarray, dict],
    output_path: str,
) -> None:
    """
    通用雙面板（1×2 subplot）繪圖函式。

    Args:
        left_data:  (x_axis, y_values_dict, plot_kwargs_dict)
        right_data: 同上，對應右圖
        output_path: 輸出圖片路徑
    """
    plt.figure(figsize=(12, 5))
    for panel_idx, (x, y_dict, kwargs) in enumerate([left_data, right_data], start=1):
        plt.subplot(1, 2, panel_idx)
        for label, y in y_dict.items():
            plt.plot(x, y, label=label, **kwargs.get(label, {}))
        plt.title(kwargs.get("title", ""))
        plt.xlabel(kwargs.get("xlabel", ""))
        plt.ylabel(kwargs.get("ylabel", ""))
        if "ylim" in kwargs:
            plt.ylim(kwargs["ylim"])
        if any(y_dict):
            plt.legend()
        plt.axhline(**kwargs["axhline"]) if "axhline" in kwargs else None
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_training_history(history, output_dir: str, target_byte: int) -> None:
    """儲存訓練過程的 Loss 與 Accuracy 雙圖表。"""
    epochs = range(1, len(history.history["loss"]) + 1)
    output_path = os.path.join(output_dir, f"training_history_byte_{target_byte}.png")

    _plot_two_panels(
        left_data=(
            epochs,
            {"Training Loss": history.history["loss"], "Validation Loss": history.history["val_loss"]},
            {"title": f"Loss (Byte {target_byte})", "xlabel": "Epochs", "ylabel": "Loss",
             "Training Loss": {"color": "b", "linewidth": 2},
             "Validation Loss": {"color": "r", "linestyle": "--", "linewidth": 2}},
        ),
        right_data=(
            epochs,
            {"Training Acc": history.history["accuracy"], "Validation Acc": history.history["val_accuracy"]},
            {"title": f"Accuracy (Byte {target_byte})", "xlabel": "Epochs", "ylabel": "Accuracy",
             "Training Acc": {"color": "b", "linewidth": 2},
             "Validation Acc": {"color": "r", "linestyle": "--", "linewidth": 2}},
        ),
        output_path=output_path,
    )


def plot_sca_metrics(ge_curve: np.ndarray, sr_curve: np.ndarray, target_byte: int, output_dir: str) -> None:
    """儲存側信道攻擊的 Guessing Entropy 與 Success Rate 雙圖表。"""
    traces = range(1, len(ge_curve) + 1)
    output_path = os.path.join(output_dir, f"sca_metrics_byte_{target_byte}.png")

    _plot_two_panels(
        left_data=(
            traces,
            {"GE": ge_curve},
            {"title": f"Guessing Entropy (Byte {target_byte})",
             "xlabel": "Number of Attack Traces", "ylabel": "Average Rank of Correct Key",
             "GE": {"color": "g", "linewidth": 2},
             "axhline": {"y": 0, "color": "r", "linestyle": ":", "alpha": 0.5}},
        ),
        right_data=(
            traces,
            {"SR": sr_curve},
            {"title": f"Success Rate (Byte {target_byte})",
             "xlabel": "Number of Attack Traces", "ylabel": "Probability of Rank == 0",
             "SR": {"color": "m", "linewidth": 2},
             "ylim": (-0.05, 1.05)},
        ),
        output_path=output_path,
    )

# =============================================================================
# 9. 訓練與攻擊（單一 Byte）
# =============================================================================

def _run_training(
    model: models.Model,
    train_gen: MixedDataGenerator,
    X_val: np.ndarray,
    y_val_onehot: np.ndarray,
    model_path: str,
    epochs: int,
    early_stop_patience: int,
    reduce_lr_patience: int,
) -> object:
    """執行模型訓練，回傳 history 物件。"""
    training_callbacks = [
        callbacks.EarlyStopping(monitor="val_loss", patience=early_stop_patience, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=reduce_lr_patience, verbose=1),
        callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=0),
    ]

    history = model.fit(
        train_gen,
        validation_data=(X_val, y_val_onehot),
        epochs=epochs,
        callbacks=training_callbacks,
        verbose=1,
        # 加上下面這兩行，讓 CPU 用多核心幫忙搬運資料
        workers=4,                  
        use_multiprocessing=True
    )
    return history


def _run_evaluation(
    model: models.Model,
    X_test: np.ndarray,
    y_test_scalar: np.ndarray,
    y_test_onehot: np.ndarray,
    pt_test: np.ndarray,
    true_key_byte: int,
    target_byte: int,
    config: dict,
) -> dict:
    """
    執行模型評估與 key recovery，回傳包含所有指標的結果 dict。
    """
    pred_probs = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(pred_probs, axis=1)

    sklearn_acc = accuracy_score(y_test_scalar, pred_classes)
    _, keras_acc, keras_top5 = model.evaluate(X_test, y_test_onehot, verbose=0)

    recovered_key, _, rank_evolution = recover_key_log_rank(
        pred_probs, pt_test, true_key=true_key_byte, target_byte=target_byte
    )

    ge_curve, sr_curve = compute_ge_sr(
        pred_probs, pt_test, true_key=true_key_byte, target_byte=target_byte,
        num_attacks=config["num_attacks"], max_traces=config["attack_limit"],
    )

    final_rank = int(rank_evolution[-1])
    first_rank0_indices = np.where(rank_evolution == 0)[0]
    first_rank0_trace = int(first_rank0_indices[0] + 1) if len(first_rank0_indices) > 0 else -1

    return {
        "byte": target_byte,
        "true_key": true_key_byte,
        "recovered_key": recovered_key,
        "final_rank": final_rank,
        "first_rank0_trace": first_rank0_trace,
        "sklearn_accuracy": float(sklearn_acc),
        "keras_accuracy": float(keras_acc),
        "top5_accuracy": float(keras_top5),
        "_ge_curve": ge_curve,   # 供繪圖使用，不會寫入 JSON
        "_sr_curve": sr_curve,
    }


def train_and_attack_byte(
    target_byte: int,
    data: dict,
    config: dict,
    output_dir: str,
) -> dict:
    """
    針對單一 AES key byte 執行完整的訓練 → 攻擊 → 評估流程。

    Args:
        target_byte: 目標 byte index（0–15）
        data:        load_ascadv2_data() 的回傳值
        config:      超參數設定 dict（見 main()）
        output_dir:  輸出圖表與模型的目錄

    Returns:
        包含攻擊結果指標的 dict
    """
    print("\n" + "=" * 70)
    print(f"  開始處理 Target Byte {target_byte} (CNN + Kubota Mixed Dataset)")
    print("=" * 70)

    # 準備標籤
    y_val_scalar, y_val_onehot = generate_sbox_labels(data["pt_val"], data["key_val"], target_byte)
    y_test_scalar, y_test_onehot = generate_sbox_labels(data["pt_test"], data["key_test"], target_byte)

    # 建立 Mixed Dataset Generator（訓練集擴充 4 倍）
    train_gen = MixedDataGenerator(
        data["X_train"], data["pt_train"], data["key_train"],
        target_byte, batch_size=config["batch_size"]
    )
    print(f"  訓練波形: {len(data['X_train'])} 筆 → 擴充後: {train_gen.total_samples} 筆/epoch")

    # 建立與訓練模型
    input_shape = data["X_train"].shape[1:]   # (T, 1)
    model = build_resnet_model(
        input_shape=input_shape,
        depth=19,                              
        dropout_rate=config["dropout_rate"],   
        learning_rate=config["learning_rate"], 
    )

    model_path = os.path.join(output_dir, f"best_cnn_model_byte_{target_byte}.keras")

    history = _run_training(
        model, train_gen, data["X_val"], y_val_onehot, model_path,
        epochs=config["epochs"],
        early_stop_patience=config["early_stop_patience"],
        reduce_lr_patience=config["reduce_lr_patience"],
    )
    plot_training_history(history, output_dir, target_byte)

    # 載入最佳 checkpoint
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)

    # 評估與 Key Recovery
    true_key_byte = int(data["key_test"][0, target_byte])
    result = _run_evaluation(
        model, data["X_test"], y_test_scalar, y_test_onehot,
        data["pt_test"], true_key_byte, target_byte, config
    )

    # 繪製 GE / SR 圖表
    plot_sca_metrics(result.pop("_ge_curve"), result.pop("_sr_curve"), target_byte, output_dir)

    # 印出摘要
    print(f"  True Key:      0x{true_key_byte:02X}")
    print(f"  Recovered Key: 0x{result['recovered_key']:02X}")
    print(f"  Final Rank:    {result['final_rank']}")
    print(f"  Accuracy:      {result['sklearn_accuracy'] * 100:.2f}%")
    print(f"  Top-5 Acc:     {result['top5_accuracy'] * 100:.2f}%")
    first_str = str(result["first_rank0_trace"]) if result["first_rank0_trace"] != -1 else "未達成"
    print(f"  First Rank-0 @ trace {first_str}")

    # 釋放記憶體
    del model
    K.clear_session()
    gc.collect()

    return result

# =============================================================================
# 10. 主程式
# =============================================================================

def main() -> None:
    enable_gpu_memory_growth()
    set_seed(42)

    # --- 路徑設定 ---
    FILE_PATH  = r"ascadv2-extracted.h5"
    OUTPUT_DIR = "sca_cnn_mixed_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 超參數（集中管理，方便調整）---
    config = {
        # 資料
        "profiling_limit": 30000,   # 建議: GPU 夠強可拉高至 30000 提升破譯率
        "attack_limit": 2000,
        "val_ratio": 0.1,
        # 訓練
        "batch_size": 128, # 樣本數
        "epochs": 30,
        "learning_rate": 1e-4, # 訓練因子
        "dropout_rate": 0.4,
        "early_stop_patience": 6,
        "reduce_lr_patience": 3,
        # 攻擊評估
        "num_attacks": 10,          # GE/SR 的 Monte Carlo 重複次數
        "target_bytes": list(range(0, 4)),
    }

    # 載入資料
    data = load_ascadv2_data(
        FILE_PATH,
        profiling_limit=config["profiling_limit"],
        attack_limit=config["attack_limit"],
        val_ratio=config["val_ratio"],
    )

    # 逐 byte 訓練與攻擊
    all_results = []
    for byte in config["target_bytes"]:
        result = train_and_attack_byte(byte, data, config, OUTPUT_DIR)
        all_results.append(result)

    # 儲存所有結果到 JSON（修正原版 bug：結果從未存檔）
    results_path = os.path.join(OUTPUT_DIR, "all_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n所有結果已儲存至: {results_path}")

    # 印出總結摘要
    print("\n" + "=" * 70)
    print("攻擊結果摘要")
    print("=" * 70)
    header = f"{'Byte':>5} | {'True':>6} | {'Recovered':>10} | {'Rank':>5} | {'Acc':>8} | {'Rank0@':>8}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        rank0_str = str(r["first_rank0_trace"]) if r["first_rank0_trace"] != -1 else "N/A"
        print(
            f"{r['byte']:>5} | 0x{r['true_key']:02X}   | "
            f"0x{r['recovered_key']:02X}       | {r['final_rank']:>5} | "
            f"{r['sklearn_accuracy']*100:>7.2f}% | {rank0_str:>8}"
        )


if __name__ == "__main__":
    main()