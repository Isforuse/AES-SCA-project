"""
DL-SCA: ASCADv2 攻擊模型 v2（輕量穩定版）
=========================================

論文參考:
    "Deep Learning Side-Channel Attack against Hardware Implementations of AES"
    Kubota et al.

本版本修正根本問題：
    1. 正確讀取 ASCADv2 的 metadata（含 masks、permutation）
    2. 攻擊目標改為「加法遮罩 (beta_mask)」—— 這是 ASCADv2 中最容易洩漏的中間值
    3. 改用輕量 CNN（針對 4-6GB VRAM 筆電 GPU 設計，不 OOM）
    4. 使用梯度累積技巧，讓小 batch 也能穩定收斂

為什麼攻擊 beta_mask 而不是直接攻擊 key？
    ASCADv2 使用 Shuffling + Masking 雙重保護。
    Shuffling 會打亂 byte 順序，使 plaintext[byte_idx] 不對應固定的 trace 特徵。
    但 beta_mask 是加法遮罩，它的側信道洩漏「不受 Shuffling 影響」，
    因為它在 ShiftRows 之前就已固定。
    一旦恢復 mask，就可以用已知的 plaintext 還原真正的 key。
"""

import gc
import json
import os
import random
import datetime

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
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def setup_gpu() -> None:
    """
    GPU 初始化：
    - 啟用動態記憶體分配（不一次佔滿 VRAM）
    - 關閉 XLA JIT（防止筆電 GPU 因激進融合造成崩潰）
    """
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.optimizer.set_jit(False)  # 關閉 XLA，防止 OOM/崩潰
    print(f"偵測到 {len(gpus)} 個 GPU，已設定動態記憶體 + 關閉 XLA")

# =============================================================================
# 3. 資料載入（正確讀取 ASCADv2 metadata）
# =============================================================================

def _inspect_metadata_keys(file_path: str) -> None:
    """
    印出 HDF5 檔案中所有 metadata 欄位名稱。
    第一次使用時建議先執行這個函式，確認你的 .h5 檔案有哪些欄位。
    """
    with h5py.File(file_path, "r") as f:
        print("=== Profiling metadata 欄位 ===")
        for k in f["Profiling_traces/metadata"].dtype.names:
            print(f"  {k}")
        print("=== Attack metadata 欄位 ===")
        for k in f["Attack_traces/metadata"].dtype.names:
            print(f"  {k}")


def _load_split(hf: h5py.File, group: str, limit: int) -> dict:
    """
    從 HDF5 的指定 group 讀取 traces 與所有 metadata 欄位。

    ASCADv2 的 metadata 除了 plaintext / key 之外，還包含：
        - masks:       加法遮罩陣列（beta_mask 在裡面）
        - permutation: Shuffling 的洗牌順序，shape (N, 16)

    Returns:
        dict，包含 traces、plaintext、key，以及其他可用的 metadata 欄位
    """
    meta = hf[f"{group}/metadata"]
    data = {
        "traces":    np.array(hf[f"{group}/traces"][:limit], dtype=np.float32),
        "plaintext": np.array(meta["plaintext"][:limit], dtype=np.uint8),
        "key":       np.array(meta["key"][:limit], dtype=np.uint8),
    }

    # 選擇性讀取 masks（若欄位存在）
    if "masks" in meta.dtype.names:
        data["masks"] = np.array(meta["masks"][:limit], dtype=np.uint8)
        print(f"  [{group}] 找到 masks，shape: {data['masks'].shape}")
    else:
        data["masks"] = None
        print(f"  [{group}] 無 masks 欄位（可能是舊版格式）")

    # 選擇性讀取 permutation（若欄位存在）
    if "permutation" in meta.dtype.names:
        data["permutation"] = np.array(meta["permutation"][:limit], dtype=np.uint8)
        print(f"  [{group}] 找到 permutation，shape: {data['permutation'].shape}")
    else:
        data["permutation"] = None
        print(f"  [{group}] 無 permutation 欄位")

    return data


def _remove_null_traces(data: dict, part_name: str) -> dict:
    """移除全為零的無效 trace，同步過濾所有對應的 metadata。"""
    null_mask = np.all(data["traces"] == 0, axis=1)
    n_removed = int(np.sum(null_mask))
    if n_removed > 0:
        print(f"[{part_name}] 移除 null traces: {n_removed} 筆")
    keep = ~null_mask
    return {k: (v[keep] if v is not None and hasattr(v, '__len__') and len(v) == len(data["traces"]) else v)
            for k, v in data.items()}


def _normalize(X_train, X_val, X_test):
    """以訓練集統計量做 Z-score 正規化，避免 data leakage。"""
    mean = np.mean(X_train, axis=0, keepdims=True)
    std  = np.std(X_train,  axis=0, keepdims=True) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std


def load_ascadv2_data(
    file_path: str,
    profiling_limit: int = 10000,
    attack_limit: int = 2000,
    val_ratio: float = 0.1,
    split_seed: int = 42,
) -> dict:
    """
    完整載入 ASCADv2 資料，包含所有 metadata 欄位。

    第一次使用時，建議先呼叫 _inspect_metadata_keys(file_path) 確認欄位名稱。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到資料集: {file_path}")

    print(f"載入資料集: {file_path}")
    with h5py.File(file_path, "r") as f:
        prof = _load_split(f, "Profiling_traces", profiling_limit)
        atk  = _load_split(f, "Attack_traces",    attack_limit)

    prof = _remove_null_traces(prof, "profiling")
    atk  = _remove_null_traces(atk,  "attack")

    # 切分 train / val
    idx = np.arange(len(prof["traces"]))
    train_idx, val_idx = train_test_split(idx, test_size=val_ratio, random_state=split_seed)

    def _split(arr, t_idx, v_idx):
        if arr is None:
            return None, None
        return arr[t_idx], arr[v_idx]

    X_train_raw, X_val_raw = prof["traces"][train_idx], prof["traces"][val_idx]
    X_train, X_val, X_test = _normalize(X_train_raw, X_val_raw, atk["traces"])

    # 增加 channel 維度：(N, T) → (N, T, 1)
    X_train = X_train[..., np.newaxis]
    X_val   = X_val[..., np.newaxis]
    X_test  = X_test[..., np.newaxis]

    masks_train, masks_val = _split(prof["masks"], train_idx, val_idx)
    perm_train,  perm_val  = _split(prof["permutation"], train_idx, val_idx)

    print(f"資料載入完成 | train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    return {
        # Traces
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        # Profiling metadata
        "pt_train":   prof["plaintext"][train_idx],
        "key_train":  prof["key"][train_idx],
        "masks_train": masks_train,
        "perm_train":  perm_train,
        # Validation metadata
        "pt_val":    prof["plaintext"][val_idx],
        "key_val":   prof["key"][val_idx],
        "masks_val": masks_val,
        "perm_val":  perm_val,
        # Attack metadata
        "pt_test":   atk["plaintext"],
        "key_test":  atk["key"],
        "masks_test": atk["masks"],
        "perm_test":  atk["permutation"],
    }

# =============================================================================
# 4. 標籤產生（三種策略，依資料集實際欄位選用）
# =============================================================================

def make_label_direct_sbox(
    plaintext: np.ndarray,
    key: np.ndarray,
    target_byte: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    策略 A：直接預測 S-Box 輸出（Identity Leakage Model）。

    適用於：沒有 masks / permutation 的情況，或作為 baseline 使用。
    注意：在 ASCADv2 上效果差，因為 Shuffling 使 byte 對應關係錯亂。

    label = S-Box( plaintext[target_byte] XOR key[target_byte] )
    """
    vals = AES_SBOX[plaintext[:, target_byte] ^ key[:, target_byte]]
    return vals, to_categorical(vals, num_classes=256)


def make_label_masked_sbox(
    plaintext: np.ndarray,
    key: np.ndarray,
    masks: np.ndarray,
    target_byte: int,
    mask_col: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    策略 B：預測帶遮罩的 S-Box 輸出（Masked Identity Leakage Model）。

    適用於：有 masks 欄位，且已知哪個 column 對應 target_byte。

    label = S-Box( pt[target_byte] XOR key[target_byte] ) XOR beta_mask

    Args:
        masks:    shape (N, M)，M 為遮罩數量（依版本而異）
        mask_col: 使用 masks 的第幾個欄位作為 beta_mask
    """
    sbox_out = AES_SBOX[plaintext[:, target_byte] ^ key[:, target_byte]]
    beta_mask = masks[:, mask_col]
    vals = sbox_out ^ beta_mask
    return vals, to_categorical(vals, num_classes=256)


def make_label_with_permutation(
    plaintext: np.ndarray,
    key: np.ndarray,
    permutation: np.ndarray,
    masks: np.ndarray,
    target_byte: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    策略 C：使用 permutation 還原真實 byte 位置後計算標籤（最完整）。

    適用於：同時有 masks 和 permutation 欄位的情況。

    流程：
        1. 用 permutation[i] 找出第 i 筆 trace 中，target_byte 實際對應的位置
        2. 取出正確的 plaintext byte
        3. XOR key 後過 S-Box，再 XOR 遮罩

    注意：permutation 的語義可能是「位置 j 的 byte 原本在哪裡」，
    需配合你的資料集版本確認方向。
    """
    n = len(plaintext)
    vals = np.zeros(n, dtype=np.uint8)

    # 向量化：找每筆 trace 中 target_byte 的實際位置
    # permutation[i] 的值代表洗牌後第 i 個位置對應原始的哪個 byte
    actual_positions = permutation[:, target_byte]  # shape (N,)
    pt_bytes  = plaintext[np.arange(n), actual_positions]
    key_bytes = key[:, target_byte]
    sbox_out  = AES_SBOX[pt_bytes ^ key_bytes]

    if masks is not None:
        beta_mask = masks[:, target_byte % masks.shape[1]]
        vals = sbox_out ^ beta_mask
    else:
        vals = sbox_out

    return vals, to_categorical(vals, num_classes=256)


# def auto_select_label_strategy(
#     data: dict,
#     target_byte: int,
#     split: str = "train",
# ) -> tuple[np.ndarray, np.ndarray]:
#     """
#     自動根據資料集實際有哪些欄位，選擇最合適的 label 策略。

#     優先順序：C（permutation + masks） > B（masks only） > A（baseline）
#     """
#     pt   = data[f"pt_{split}"]
#     key  = data[f"key_{split}"]
#     masks = data.get(f"masks_{split}")
#     perm  = data.get(f"perm_{split}")

#     if perm is not None and masks is not None:
#         print(f"  [{split}] 使用策略 C：permutation + masks")
#         return make_label_with_permutation(pt, key, perm, masks, target_byte)
#     elif masks is not None:
#         print(f"  [{split}] 使用策略 B：masked S-Box")
#         return make_label_masked_sbox(pt, key, masks, target_byte)
#     else:
#         print(f"  [{split}] 使用策略 A：直接 S-Box（baseline）")
#         return make_label_direct_sbox(pt, key, target_byte)

def auto_select_label_strategy(data: dict, target_byte: int, split: str = "train"):
    # 強制使用策略 A (Identity Leakage)
    # 讓 CNN 自己在內部去對抗 Mask
    pt   = data[f"pt_{split}"]
    key  = data[f"key_{split}"]
    return make_label_direct_sbox(pt, key, target_byte)

# =============================================================================
# 5. 輕量 CNN（針對 4-6 GB VRAM 設計）
# =============================================================================

def _conv_block(x, filters: int, kernel_size: int, pool_size: int = 2):
    """Conv1D → BatchNorm → SiLU(Swish) → MaxPool。"""
    x = layers.Conv1D(filters, kernel_size=kernel_size, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    # SiLU / Swish 在 SCA 問題上比 ReLU 更平滑，有助於學習微弱的洩漏特徵
    x = layers.Activation("swish")(x)
    x = layers.MaxPooling1D(pool_size=pool_size)(x)
    return x


# def build_lightweight_cnn(
#     input_shape: tuple,
#     num_classes: int = 256,
#     dropout_rate: float = 0.3,
#     learning_rate: float = 1e-4,
# ) -> models.Model:
#     """
#     輕量 CNN，針對 4-6 GB VRAM 的筆電 GPU 設計。

#     設計原則：
#     - 只有 3 個 Conv Block（vs 原來 4 個），大幅降低參數量
#     - pool_size 從 4 改為 2，每層只壓縮一半，梯度更穩定
#     - filters 上限壓在 128（原來到 256），節省 VRAM
#     - 加入 GlobalAveragePooling（比 Flatten 少 10x 參數）
#     - 加入 L2 正則化防止 overfitting（小資料集容易過擬合）

#     參數量估計：~200K（原版約 2M，ResNet 約 5M）

#     記憶體估計（batch=64, trace長度=250000點）：
#         Warning: 如果 trace 很長（> 50000 點），建議先用前處理做 POI 選取，
#         只取有洩漏的時間窗口，否則即使輕量模型也會 OOM。
#     """
#     l2 = tf.keras.regularizers.l2(1e-4)
#     inputs = layers.Input(shape=input_shape)

#     # Block 1：大 kernel 捕捉長距離特徵
#     x = _conv_block(inputs, filters=32, kernel_size=11, pool_size=4)

#     # Block 2：中 kernel 捕捉中距離特徵
#     x = _conv_block(x, filters=64, kernel_size=7, pool_size=4)

#     # Block 3：小 kernel 精細特徵萃取
#     x = _conv_block(x, filters=128, kernel_size=3, pool_size=4)

#     # GlobalAveragePooling 取代 Flatten，大幅減少參數量並改善泛化
#     x = layers.GlobalAveragePooling1D()(x)

#     # 分類頭
#     x = layers.Dense(128, activation="swish", kernel_regularizer=l2)(x)
#     x = layers.Dropout(dropout_rate)(x)
#     outputs = layers.Dense(num_classes, activation="softmax")(x)

#     model = models.Model(inputs=inputs, outputs=outputs, name="LightweightCNN_SCA")
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#         loss="categorical_crossentropy",
#         metrics=[
#             "accuracy",
#             tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc"),
#         ],
#     )
#     return model

def build_lightweight_cnn(
        input_shape: tuple, 
        num_classes: int = 256, 
        dropout_rate: float = 0.3, 
        learning_rate: float = 1e-4):
    l2 = tf.keras.regularizers.l2(1e-4)
    inputs = layers.Input(shape=input_shape)

    # 加大 Pool Size，強烈壓縮時間軸，這樣後面 Flatten 就不會爆炸
    x = _conv_block(inputs, filters=32, kernel_size=11, pool_size=4)
    x = _conv_block(x, filters=64, kernel_size=7, pool_size=4)
    x = _conv_block(x, filters=128, kernel_size=3, pool_size=4)
    x = _conv_block(x, filters=256, kernel_size=3, pool_size=4) # 多加一層，增強解碼能力

    x = layers.Flatten()(x) # 絕對要用 Flatten
    
    x = layers.Dense(256, activation="swish", kernel_regularizer=l2)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5_acc")]
    )
    return model

# =============================================================================
# 6. 輕量版 Mixed Data Generator
# =============================================================================

class MixedDataGenerator(Sequence):
    """
    Kubota Mixed Dataset Generator（輕量版）。

    相較原版的改進：
    - __getitem__ 完全向量化，移除 Python for loop
    - 支援動態注入 labels（配合策略 B / C 的 masks）
    """

    def __init__(
        self,
        traces: np.ndarray,
        labels_per_byte: list[np.ndarray],  # 長度 4，每個 shape (N,)
        batch_size: int = 64,
    ) -> None:
        """
        Args:
            traces:         shape (N, T, 1)
            labels_per_byte: 4 個 scalar label array，分別對應同 column 的 4 個 byte
            batch_size:     建議筆電 GPU 使用 64（原版 128 容易 OOM）
        """
        self.traces = traces
        self.labels_per_byte = [np.array(l, dtype=np.int32) for l in labels_per_byte]
        self.batch_size = batch_size

        n = len(traces)
        self.total_samples = n * 4  # 4 個 byte 各產生一份標籤
        self.indices = np.arange(self.total_samples)
        np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(self.total_samples / self.batch_size))

    def __getitem__(self, batch_idx: int) -> tuple[np.ndarray, np.ndarray]:
        batch_indices = self.indices[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]

        n = len(self.traces)
        trace_idx    = batch_indices % n       # 對應哪筆 trace
        byte_selector = batch_indices // n     # 對應哪個 byte（0, 1, 2, 3）

        batch_traces = self.traces[trace_idx]  # shape (B, T, 1)

        # 向量化取出每筆樣本對應的 scalar label
        label_arrays = np.stack(self.labels_per_byte, axis=1)  # shape (N, 4)
        scalar_labels = label_arrays[trace_idx, byte_selector]  # shape (B,)

        # 轉為 one-hot
        batch_labels = to_categorical(scalar_labels, num_classes=256).astype(np.float32)
        return batch_traces, batch_labels

    def on_epoch_end(self) -> None:
        np.random.shuffle(self.indices)

# =============================================================================
# 7. 攻擊評估
# =============================================================================

def recover_key_log_rank(
    pred_probs: np.ndarray,
    plaintexts: np.ndarray,
    true_key: int,
    target_byte: int = 0,
) -> tuple[int, np.ndarray, np.ndarray]:
    """log-likelihood 累加法恢復 key，追蹤每步的排名變化。"""
    eps = 1e-36
    log_scores = np.zeros(256, dtype=np.float64)
    rank_evolution = []
    key_hypotheses = np.arange(256, dtype=np.uint8)

    for i in range(len(pred_probs)):
        pt_byte = plaintexts[i, target_byte]
        sbox_vals = AES_SBOX[key_hypotheses ^ pt_byte]
        log_scores += np.log(pred_probs[i, sbox_vals] + eps)

        sorted_desc = np.argsort(log_scores)[::-1]
        rank = int(np.where(sorted_desc == true_key)[0][0])
        rank_evolution.append(rank)

    return int(np.argmax(log_scores)), log_scores, np.array(rank_evolution, dtype=np.int32)


def compute_ge_sr(
    pred_probs: np.ndarray,
    plaintexts: np.ndarray,
    true_key: int,
    target_byte: int = 0,
    num_attacks: int = 10,
    max_traces: int = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Monte Carlo 計算 Guessing Entropy 與 Success Rate。"""
    n = len(pred_probs)
    max_traces = min(max_traces or n, n)
    rng = np.random.default_rng(seed)

    ge_curve = np.zeros(max_traces, dtype=np.float64)
    sr_curve = np.zeros(max_traces, dtype=np.float64)

    for _ in range(num_attacks):
        idx = rng.permutation(n)[:max_traces]
        _, _, ranks = recover_key_log_rank(pred_probs[idx], plaintexts[idx], true_key, target_byte)
        ge_curve += ranks
        sr_curve += (ranks == 0).astype(np.float64)

    return ge_curve / num_attacks, sr_curve / num_attacks

# =============================================================================
# 8. 繪圖
# =============================================================================

def plot_training_history(history, output_dir: str, target_byte: int) -> None:
    epochs = range(1, len(history.history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history.history["loss"],     "b-",  label="Train Loss", linewidth=2)
    ax1.plot(epochs, history.history["val_loss"], "r--", label="Val Loss",   linewidth=2)
    ax1.set_title(f"Loss (Byte {target_byte})")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, history.history["accuracy"],     "b-",  label="Train Acc", linewidth=2)
    ax2.plot(epochs, history.history["val_accuracy"], "r--", label="Val Acc",   linewidth=2)
    ax2.set_title(f"Accuracy (Byte {target_byte})")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_history_byte_{target_byte}.png"), dpi=150)
    plt.close()


def plot_sca_metrics(ge_curve: np.ndarray, sr_curve: np.ndarray, target_byte: int, output_dir: str) -> None:
    traces = range(1, len(ge_curve) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(traces, ge_curve, "g-", linewidth=2)
    ax1.axhline(y=0, color="r", linestyle=":", alpha=0.5)
    ax1.set_title(f"Guessing Entropy (Byte {target_byte})")
    ax1.set_xlabel("Attack Traces"); ax1.set_ylabel("Rank"); ax1.grid(True)

    ax2.plot(traces, sr_curve, "m-", linewidth=2)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title(f"Success Rate (Byte {target_byte})")
    ax2.set_xlabel("Attack Traces"); ax2.set_ylabel("P(rank=0)"); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sca_metrics_byte_{target_byte}.png"), dpi=150)
    plt.close()

# =============================================================================
# 9. 主訓練與攻擊流程
# =============================================================================

def _build_mixed_generator(data: dict, target_byte: int, batch_size: int) -> MixedDataGenerator:
    """
    為 target_byte 所在的 column 建立 Mixed Generator。
    對每個 byte 分別用 auto_select_label_strategy 計算正確標籤。
    """
    column = target_byte // 4
    column_bytes = [column * 4 + i for i in range(4)]

    labels_per_byte = []
    for b in column_bytes:
        scalars, _ = auto_select_label_strategy(data, b, split="train")
        labels_per_byte.append(scalars)

    return MixedDataGenerator(data["X_train"], labels_per_byte, batch_size=batch_size)


def train_and_attack_byte(
    target_byte: int,
    data: dict,
    config: dict,
    output_dir: str,
) -> dict:
    """
    針對單一 AES key byte 執行完整的訓練 → 攻擊 → 評估流程。
    """
    print("\n" + "=" * 70)
    print(f"  Target Byte {target_byte} | 輕量 CNN + Kubota Mixed Dataset + 正確 Label")
    print("=" * 70)

    # 生成驗證集 / 測試集標籤
    y_val_scalar, y_val_onehot   = auto_select_label_strategy(data, target_byte, "val")
    y_test_scalar, y_test_onehot = auto_select_label_strategy(data, target_byte, "test")

    # 建立 Mixed Generator（訓練集）
    train_gen = _build_mixed_generator(data, target_byte, config["batch_size"])
    print(f"  訓練樣本數（擴充後）: {train_gen.total_samples}")

    # 建模
    input_shape = data["X_train"].shape[1:]
    model = build_lightweight_cnn(
        input_shape,
        dropout_rate=config["dropout_rate"],
        learning_rate=config["learning_rate"],
    )
    model.summary()

    model_path = os.path.join(output_dir, f"best_model_byte_{target_byte}.keras")
    training_callbacks = [
        callbacks.EarlyStopping(monitor="val_loss", patience=config["early_stop_patience"], restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=config["reduce_lr_patience"], min_lr=1e-6, verbose=1),
        callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=0),
    ]

    history = model.fit(
        train_gen,
        validation_data=(data["X_val"], y_val_onehot),
        epochs=config["epochs"],
        callbacks=training_callbacks,
        verbose=1,
    )
    plot_training_history(history, output_dir, target_byte)

    # 載入最佳 checkpoint
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)

    # 預測與評估
    pred_probs = model.predict(data["X_test"], verbose=0)
    pred_classes = np.argmax(pred_probs, axis=1)
    sklearn_acc = accuracy_score(y_test_scalar, pred_classes)
    _, keras_acc, keras_top5 = model.evaluate(data["X_test"], y_test_onehot, verbose=0)

    true_key_byte = int(data["key_test"][0, target_byte])
    recovered_key, _, rank_evolution = recover_key_log_rank(
        pred_probs, data["pt_test"], true_key_byte, target_byte
    )
    ge_curve, sr_curve = compute_ge_sr(
        pred_probs, data["pt_test"], true_key_byte, target_byte,
        num_attacks=config["num_attacks"], max_traces=config["attack_limit"],
    )
    plot_sca_metrics(ge_curve, sr_curve, target_byte, output_dir)

    final_rank = int(rank_evolution[-1])
    first_rank0_indices = np.where(rank_evolution == 0)[0]
    first_rank0_trace = int(first_rank0_indices[0] + 1) if len(first_rank0_indices) > 0 else -1

    result = {
        "byte": target_byte,
        "true_key": true_key_byte,
        "recovered_key": recovered_key,
        "final_rank": final_rank,
        "first_rank0_trace": first_rank0_trace,
        "sklearn_accuracy": float(sklearn_acc),
        "keras_accuracy": float(keras_acc),
        "top5_accuracy": float(keras_top5),
    }

    rank0_str = str(first_rank0_trace) if first_rank0_trace != -1 else "未達成"
    print(f"  True Key:      0x{true_key_byte:02X}")
    print(f"  Recovered Key: 0x{recovered_key:02X}")
    print(f"  Final Rank:    {final_rank}")
    print(f"  Accuracy:      {sklearn_acc * 100:.2f}%  |  Top-5: {keras_top5 * 100:.2f}%")
    print(f"  First Rank-0 @ trace {rank0_str}")

    del model
    K.clear_session()
    gc.collect()
    return result

# =============================================================================
# 10. 主程式
# =============================================================================

def main() -> None:
    setup_gpu()
    set_seed(42)

    # -------------------------------------------------------------------------
    # 【建議步驟 0】第一次執行前，先執行下面這行確認你的 .h5 有哪些 metadata 欄位
    _inspect_metadata_keys("ascadv2-extracted.h5")
    # -------------------------------------------------------------------------

    FILE_PATH  = r"ascadv2-extracted.h5"
    
    # 1. 定義總資料夾
    BASE_DIR = "sca_cnn_v2_results"
    
    # 2. 取得現在的時間，格式化為 年月日_時分秒 (例如: 20260410_011830)
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 3. 組合出這次專屬的子資料夾路徑
    OUTPUT_DIR = os.path.join(BASE_DIR, f"run_{current_time}")
    
    # 4. 建立資料夾 (會自動連同外層的 BASE_DIR 一起建好)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n本次訓練結果將獨立儲存於: {OUTPUT_DIR}\n")

    # 集中管理所有超參數
    config = {
        # 資料
        "profiling_limit": 30000,   # OOM 問題解決後可嘗試拉高至 30000
        "attack_limit":    2000,
        "val_ratio":       0.1,
        # 訓練（針對 4-6 GB VRAM 調整）
        "batch_size":      64,      # 原版 128，降為 64 防 OOM
        "epochs":          50,      # 配合 EarlyStopping，設高一點讓它自己停
        "learning_rate":   1e-4,
        "dropout_rate":    0.3,
        "early_stop_patience": 8,
        "reduce_lr_patience":  4,
        # 攻擊評估
        "num_attacks":   10,
        "target_bytes":  list(range(0, 4)),
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

    # 儲存結果
    results_path = os.path.join(OUTPUT_DIR, "all_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n結果已儲存: {results_path}")

    # 總結摘要
    print("\n" + "=" * 70)
    print("攻擊結果摘要")
    print("=" * 70)
    print(f"{'Byte':>5} | {'True':>6} | {'Recovered':>10} | {'Rank':>5} | {'Acc%':>7} | {'Top5%':>7} | {'Rank0@':>8}")
    print("-" * 65)
    for r in all_results:
        r0 = str(r["first_rank0_trace"]) if r["first_rank0_trace"] != -1 else "N/A"
        print(
            f"{r['byte']:>5} | 0x{r['true_key']:02X}   | 0x{r['recovered_key']:02X}       | "
            f"{r['final_rank']:>5} | {r['sklearn_accuracy']*100:>6.2f}% | "
            f"{r['top5_accuracy']*100:>6.2f}% | {r0:>8}"
        )


if __name__ == "__main__":
    main()