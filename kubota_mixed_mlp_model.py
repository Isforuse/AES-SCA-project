import os
import random
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gc

from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# =============================================================================
# 1. 基礎設定與 AES S-Box
# =============================================================================
AES_Sbox = np.array([
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
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
], dtype=np.uint8)

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# =============================================================================
# 2. 資料載入與 SNR 特徵提取 (配合論文的精簡輸入)
# =============================================================================
def load_data_2d(file_path, profiling_limit=50000, attack_limit=2000, val_ratio=0.1):
    with h5py.File(file_path, "r") as f:
        X_prof = np.array(f["Profiling_traces/traces"][:profiling_limit], dtype=np.float32)
        pt_prof = np.array(f["Profiling_traces/metadata"]["plaintext"][:profiling_limit], dtype=np.uint8)
        key_prof = np.array(f["Profiling_traces/metadata"]["key"][:profiling_limit], dtype=np.uint8)

        X_attack = np.array(f["Attack_traces/traces"][:attack_limit], dtype=np.float32)
        pt_attack = np.array(f["Attack_traces/metadata"]["plaintext"][:attack_limit], dtype=np.uint8)
        key_attack = np.array(f["Attack_traces/metadata"]["key"][:attack_limit], dtype=np.uint8)

    # 清除 null traces (省略實作細節，直接使用標準化)
    train_idx, val_idx = train_test_split(np.arange(len(X_prof)), test_size=val_ratio, random_state=42)
    
    X_train, pt_train, key_train = X_prof[train_idx], pt_prof[train_idx], key_prof[train_idx]
    X_val, pt_val, key_val = X_prof[val_idx], pt_prof[val_idx], key_prof[val_idx]
    X_test, pt_test, key_test = X_attack, pt_attack, key_attack

    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True) + 1e-8
    
    return (X_train - mean) / std, pt_train, key_train, (X_val - mean) / std, pt_val, key_val, (X_test - mean) / std, pt_test, key_test

def extract_pois_snr(traces, pt, key, target_byte, num_pois=30):
    """ 使用 256 分類計算 SNR 以擷取特徵點 """
    labels = AES_Sbox[np.bitwise_xor(pt[:, target_byte], key[:, target_byte])]
    means = np.zeros((256, traces.shape[1]))
    vars_ = np.zeros((256, traces.shape[1]))
    
    for i in range(256):
        idx = (labels == i)
        if np.sum(idx) > 0:
            means[i] = np.mean(traces[idx], axis=0)
            vars_[i] = np.var(traces[idx], axis=0)
            
    snr = np.var(means, axis=0) / (np.mean(vars_, axis=0) + 1e-8)
    return np.sort(np.argsort(snr)[-num_pois:])

# =============================================================================
# 3. 論文核心實作：混合模型資料集 (Mixed Model Dataset)
# =============================================================================
def get_adjacent_bytes(target_byte):
    """ 根據論文設定，找尋同一個 Column 內的其他三個 Byte 作為混合標籤 """
    col = target_byte % 4
    return [i for i in range(16) if i % 4 == col and i != target_byte]

def create_mixed_dataset(X, pt, key, target_byte):
    """ 
    Kubota 等人提出的核心方法：複製波形並混入相鄰 Byte 的標籤。
    目的：解決 ShiftRows 造成的分佈偏差，並擴增 3 倍資料量。
    """
    print(f"   [Mixed Model] 正在建立混合資料集... 目標 Byte: {target_byte}")
    mixed_X = []
    mixed_Y = []
    
    adjacent_bytes = get_adjacent_bytes(target_byte)
    print(f"   [Mixed Model] 借用標籤的相鄰 Bytes: {adjacent_bytes}")
    
    for adj_byte in adjacent_bytes:
        # 計算相鄰 Byte 的真實 256 類標籤
        adj_labels = AES_Sbox[np.bitwise_xor(pt[:, adj_byte], key[:, adj_byte])]
        adj_y_onehot = to_categorical(adj_labels, num_classes=256)
        
        # 波形相同，但標籤換成相鄰 Byte 的
        mixed_X.append(X)
        mixed_Y.append(adj_y_onehot)
        
    # 垂直疊加，資料量暴增 3 倍
    X_mixed_final = np.vstack(mixed_X)
    Y_mixed_final = np.vstack(mixed_Y)
    
    # 隨機打亂資料
    shuffle_idx = np.random.permutation(len(X_mixed_final))
    return X_mixed_final[shuffle_idx], Y_mixed_final[shuffle_idx]

# =============================================================================
# 4. 論文核心架構：Kubota CNN 模型
# =============================================================================
def build_kubota_cnn(input_shape):
    """ 完全依照論文 Table II 的結構打造的 CNN """
    inputs = layers.Input(shape=input_shape)
    
    # 論文中的唯一一層 Convolution (size=10, filters=10, stride=1, Relu)
    # 這裡加入 padding="same" 確保長度不變，並適度加入 MaxPooling 降維
    x = layers.Conv1D(filters=10, kernel_size=10, strides=1, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x) 
    x = layers.Flatten()(x)
    
    # 論文的四層 Full connection layer (64->64->256->256)
    # 論文特別指定使用 Tanh 激活函數
    x = layers.Dense(64, activation="tanh")(x)
    x = layers.Dense(64, activation="tanh")(x)
    x = layers.Dense(256, activation="tanh")(x)
    
    # Output layer (256 classes, Softmax)
    outputs = layers.Dense(256, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Kubota_Mixed_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =============================================================================
# 5. 攻擊評估流程
# =============================================================================
def recover_key_log_rank_id(pred_probs, plaintexts, true_key, target_byte=0):
    eps = 1e-36
    log_key_scores = np.zeros(256, dtype=np.float64)
    rank_evolution = []

    for i in range(len(pred_probs)):
        pt_b = plaintexts[i, target_byte]
        key_hypotheses = np.arange(256, dtype=np.uint8)
        hyp_intermediates = AES_Sbox[np.bitwise_xor(key_hypotheses, pt_b)]
        probs_for_all_keys = pred_probs[i, hyp_intermediates]
        
        log_key_scores += np.log(probs_for_all_keys + eps)
        sorted_indices = np.argsort(log_key_scores)[::-1]
        rank_evolution.append(int(np.where(sorted_indices == true_key)[0][0]))

    return int(np.argmax(log_key_scores)), np.array(rank_evolution, dtype=np.int32)

def run_kubota_attack(target_byte, data_dict, num_pois=30):
    print("\n" + "=" * 60)
    print(f"執行 Kubota 混合模型攻擊 - 目標 Byte: {target_byte}")
    print("=" * 60)
    
    # 1. 抓取 POI (降維)
    poi_indices = extract_pois_snr(data_dict["X_train"], data_dict["pt_train"], data_dict["key_train"], target_byte, num_pois)
    X_tr_cut = data_dict["X_train"][:, poi_indices]
    X_v_cut = data_dict["X_val"][:, poi_indices]
    X_te_cut = data_dict["X_test"][:, poi_indices]

    # 2. 【核心】產生擴增 3 倍的混合資料集
    X_mixed, Y_mixed = create_mixed_dataset(X_tr_cut, data_dict["pt_train"], data_dict["key_train"], target_byte)
    
    # 驗證集也需要產生單一的正確標籤供評估用
    _, Y_val = to_categorical(AES_Sbox[np.bitwise_xor(data_dict["pt_val"][:, target_byte], data_dict["key_val"][:, target_byte])], num_classes=256), to_categorical(AES_Sbox[np.bitwise_xor(data_dict["pt_val"][:, target_byte], data_dict["key_val"][:, target_byte])], num_classes=256)
    
    # CNN 需要 3D 輸入
    X_mixed = X_mixed[..., np.newaxis]
    X_v_cut = X_v_cut[..., np.newaxis]
    X_te_cut = X_te_cut[..., np.newaxis]

    # 3. 建立並訓練模型
    model = build_kubota_cnn(input_shape=(num_pois, 1))
    
    cb = [callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    
    print(f"   [訓練] 使用混合資料集 (總筆數: {len(X_mixed)}) 開始訓練...")
    model.fit(X_mixed, Y_mixed, validation_data=(X_v_cut, Y_val), batch_size=256, epochs=30, callbacks=cb, verbose=1)

    # 4. 進行攻擊預測
    pred_probs = model.predict(X_te_cut, verbose=0)
    true_key = int(data_dict["key_test"][0, target_byte])
    
    recovered_key, rank_evolution = recover_key_log_rank_id(pred_probs, data_dict["pt_test"], true_key, target_byte)
    
    final_rank = rank_evolution[-1]
    print(f"   => Byte {target_byte} 最終排名 (Final Rank): {final_rank}")
    
    del model
    K.clear_session()
    gc.collect()
    return final_rank

# =============================================================================
# 6. 主程式
# =============================================================================
def main():
    set_seed(42)
    FILE_PATH = r"ascadv2-extracted.h5"
    TARGET_BYTES = [0] # 測試 Byte 0，這正是受 ShiftRows 偏差影響最深的 Byte
    
    print("正在載入資料...")
    X_tr, pt_tr, k_tr, X_v, pt_v, k_v, X_te, pt_te, k_te = load_data_2d(FILE_PATH, 50000, 2000)
    data_dict = {"X_train": X_tr, "pt_train": pt_tr, "key_train": k_tr, "X_val": X_v, "pt_val": pt_v, "key_val": k_v, "X_test": X_te, "pt_test": pt_te, "key_test": k_te}

    for byte in TARGET_BYTES:
        run_kubota_attack(byte, data_dict, num_pois=30)

if __name__ == "__main__":
    main()