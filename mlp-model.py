import os
import random
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
import datetime

from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# =============================================================================
# 1. 常數與初始化設定
# =============================================================================

# AES S-box：用於計算加密過程中的中間值（Sbox(P ⊕ K)）
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
    """ 固定所有隨機種子，確保實驗結果可重複 """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# =============================================================================
# 2. 資料載入與預處理
# =============================================================================

def is_null_trace(x):
    """ 檢查波形是否全為 0，防止異常資料進入訓練 """
    return np.all(x == 0, axis=1)

def remove_null_traces(traces, plaintext, key):
    """ 移除無效波形及其對應的明文與密鑰 """
    null_mask = is_null_trace(traces)
    keep_mask = ~null_mask
    return traces[keep_mask], plaintext[keep_mask], key[keep_mask]

def load_data_2d(file_path, profiling_limit=50000, attack_limit=2000, val_ratio=0.1, split_seed=42):
    """ 從 H5 檔案讀取資料 。此版本不進行 CNN 的 3D 重塑，保留 2D 形狀給 MLP 使用。"""
    with h5py.File(file_path, "r") as f:
        # 讀取訓練用資料 (Profiling traces) 
        X_prof = np.array(f["Profiling_traces/traces"][:profiling_limit], dtype=np.float32)
        pt_prof = np.array(f["Profiling_traces/metadata"]["plaintext"][:profiling_limit], dtype=np.uint8)
        key_prof = np.array(f["Profiling_traces/metadata"]["key"][:profiling_limit], dtype=np.uint8)

        # 讀取測試用資料 (Attack traces) 
        X_attack = np.array(f["Attack_traces/traces"][:attack_limit], dtype=np.float32)
        pt_attack = np.array(f["Attack_traces/metadata"]["plaintext"][:attack_limit], dtype=np.uint8)
        key_attack = np.array(f["Attack_traces/metadata"]["key"][:attack_limit], dtype=np.uint8)

    # 基礎清洗
    X_prof, pt_prof, key_prof = remove_null_traces(X_prof, pt_prof, key_prof)
    X_attack, pt_attack, key_attack = remove_null_traces(X_attack, pt_attack, key_attack)

    # 分割訓練集與驗證集 
    train_idx, val_idx = train_test_split(
        np.arange(len(X_prof)), test_size=val_ratio, random_state=split_seed, shuffle=True
    )

    X_train, pt_train, key_train = X_prof[train_idx], pt_prof[train_idx], key_prof[train_idx]
    X_val, pt_val, key_val = X_prof[val_idx], pt_prof[val_idx], key_prof[val_idx]
    X_test, pt_test, key_test = X_attack, pt_attack, key_attack

    # Z-score 標準化：讓波形的數值分佈在 0 附近，加速模型收斂 
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True) + 1e-8
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, pt_train, key_train, X_val, pt_val, key_val, X_test, pt_test, key_test

# =============================================================================
# 3. 標籤產生與密鑰恢復 (Identity 256 分類)
# =============================================================================

def generate_id_labels(pt, key, target_byte=0):
    """ 計算 Sbox(P ⊕ K) 並轉為 One-hot 256 分類。 Identity 模型能提供比 HW 模型更高的資訊熵。"""
    sbox_values = AES_Sbox[np.bitwise_xor(pt[:, target_byte], key[:, target_byte])]
    y = to_categorical(sbox_values, num_classes=256)
    return sbox_values, y

def rank_of_true_key(log_probs, true_key):
    """ 在 256 個候選密鑰中，找出正確密鑰目前的排名（0 代表猜中） """
    sorted_indices = np.argsort(log_probs)[::-1]
    return int(np.where(sorted_indices == true_key)[0][0])

def recover_key_log_rank_id(pred_probs, plaintexts, true_key, target_byte=0):
    """ 使用對數似然估計 (Log-likelihood) 累積多條波形的預測機率來恢復密鑰 """
    eps = 1e-36 # 防止 log(0) 報錯
    log_key_scores = np.zeros(256, dtype=np.float64)
    rank_evolution = []

    for i in range(len(pred_probs)):
        pt_b = plaintexts[i, target_byte]
        key_hypotheses = np.arange(256, dtype=np.uint8)
        # 對於 256 種假設的密鑰，分別計算對應的 S-box 輸出
        hyp_intermediates = AES_Sbox[np.bitwise_xor(key_hypotheses, pt_b)]
        # 從模型預測的機率向量中，抓取這些 S-box 輸出的機率
        probs_for_all_keys = pred_probs[i, hyp_intermediates]
        
        # 累積機率的對數值
        log_key_scores += np.log(probs_for_all_keys + eps)
        # 紀錄隨著波形增加，正確密鑰的排名變化
        rank_evolution.append(rank_of_true_key(log_key_scores, true_key))

    recovered_key = int(np.argmax(log_key_scores))
    return recovered_key, log_key_scores, np.array(rank_evolution, dtype=np.int32)

def compute_ge(pred_probs, plaintexts, true_key, target_byte=0, num_attacks=20, max_traces=None):
    """ 計算猜測熵 (Guessing Entropy, GE)。重複多次攻擊取平均值，確保結果具備統計意義。 """
    rng = np.random.default_rng(42)
    n = len(pred_probs)
    if max_traces is None or max_traces > n: max_traces = n

    ge_curve = np.zeros(max_traces, dtype=np.float64)

    for _ in range(num_attacks):
        indices = rng.permutation(n)[:max_traces] # 隨機打亂測試波形順序
        sub_preds, sub_pts = pred_probs[indices], plaintexts[indices]
        _, _, rank_evolution = recover_key_log_rank_id(sub_preds, sub_pts, true_key, target_byte)
        ge_curve += rank_evolution

    return ge_curve / num_attacks # 取平均排名

def plot_ge_curve(values, target_byte, output_dir, exp_tag=""):
    """ 繪製 GE 曲線，觀察模型在多少條波形時能完全破解密鑰 (GE=0) """
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(values) + 1), values, label="Guessing Entropy")
    plt.xlabel("Number of Attack Traces")
    plt.ylabel("Guessing Entropy")
    # 標題也印出 tag，方便截圖放入報告
    plt.title(f"GE Curve (MLP) - Byte {target_byte} [{exp_tag}]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # 檔名加上 exp_tag
    filename = f"ge_mlp_byte_{target_byte}_{exp_tag}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close()

# =============================================================================
# 4. SNR 特徵提取 (Points of Interest, POIs)
# =============================================================================

def extract_pois_snr(traces, labels, target_byte, output_dir, exp_tag="", num_classes=256, num_pois=50):
    """ 
    計算信噪比 (Signal-to-Noise Ratio)。
    目的：從 100,000 個採樣點中，找出電力洩漏最嚴重的 POIs，藉此將 MLP 的輸入維度大幅壓縮。
    """
    print(f"   [特徵工程] 正在計算 SNR，提取 Byte {target_byte} 的 {num_pois} 個黃金特徵點...")
    num_samples = traces.shape[1]
    means = np.zeros((num_classes, num_samples))
    vars_ = np.zeros((num_classes, num_samples))
    
    # 計算各類別 (0-255) 的平均電力波形與變異數
    for i in range(num_classes):
        idx = (labels == i)
        if np.sum(idx) > 0:
            means[i] = np.mean(traces[idx], axis=0)
            vars_[i] = np.var(traces[idx], axis=0)
            
    # SNR = Var(類別平均) / Mean(類別變異數)
    snr = np.var(means, axis=0) / (np.mean(vars_, axis=0) + 1e-8)
    
    # 排序並選出前 N 個最高峰值的座標
    poi_indices = np.sort(np.argsort(snr)[-num_pois:])
    
    # 將 SNR 曲線存成圖表，方便展示特徵提取成果
    filename = f"snr_pois_byte_{target_byte}_{exp_tag}.png"
    plt.title(f"SNR & Selected POIs (Byte {target_byte}) [{exp_tag}]")
    plt.figure(figsize=(10, 4))
    plt.plot(snr, color='lightblue', alpha=0.7, label='Full SNR Curve')
    plt.scatter(poi_indices, snr[poi_indices], color='red', s=15, zorder=5, label=f'Top {num_pois} Selected POIs')
    plt.title(f"SNR & Selected Points of Interest (Byte {target_byte})")
    plt.xlabel("Time Sample Point")
    plt.ylabel("SNR Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=200)
    plt.close()

    return poi_indices

# =============================================================================
# 5. MLP 模型架構與訓練流程
# =============================================================================

def build_mlp_model(input_shape):
    """ 專門處理低維度特徵的全連接神經網路 """
    inputs = layers.Input(shape=input_shape)
    
    # 第一層 Dense
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.BatchNormalization()(x) # 標準化層，防止權重偏移
    x = layers.Dropout(0.3)(x)         # 隨機捨棄 30% 節點，防止過擬合
    
    # 第二層 Dense
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # 輸出層：256 類對應 0x00-0xFF
    outputs = layers.Dense(256, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="Dedicated_SCA_MLP")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def run_mlp_attack(target_byte, data_dict, output_dir, num_pois=50, exp_tag="", epochs=40):
    """ 單一 Byte 的完整 MLP 訓練與攻擊流程 """
    print("\n" + "=" * 60)
    print(f"執行專屬 MLP 流程 - 目標 Byte: {target_byte}")
    print("=" * 60)
    
    # 1. 產生 256 分類標籤
    y_train_true, Y_train = generate_id_labels(data_dict["pt_train"], data_dict["key_train"], target_byte)
    _, Y_val = generate_id_labels(data_dict["pt_val"], data_dict["key_val"], target_byte)
    y_test_true, Y_test = generate_id_labels(data_dict["pt_test"], data_dict["key_test"], target_byte)

    # 2. 【核心特徵提取】執行 SNR 分析
    poi_indices = extract_pois_snr(
        data_dict["X_train"], y_train_true, target_byte, 
        output_dir, exp_tag=exp_tag, num_pois=num_pois
    )
    
    # 3. 執行資料切片 (Slicing)，將 100,000 點波形縮減為 num_pois 點
    X_train_mlp = data_dict["X_train"][:, poi_indices]
    X_val_mlp = data_dict["X_val"][:, poi_indices]
    X_test_mlp = data_dict["X_test"][:, poi_indices]

    print(f"   [進度報告] 輸入維度已從 {data_dict['X_train'].shape[1]} 壓縮至 {X_train_mlp.shape[1]} (僅保留物理洩漏點)。")

    # 4. 建立模型
    model = build_mlp_model(input_shape=(num_pois,))
    model_name = f"best_mlp_byte_{target_byte}_{exp_tag}.keras"
    model_path = os.path.join(output_dir, model_name)
    
    # 訓練監控
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=0)
    ]

    # 開始訓練
    model.fit(
        X_train_mlp, Y_train, 
        validation_data=(X_val_mlp, Y_val), 
        batch_size=256, 
        # epochs=EPOCHS, 
        epochs=epochs,
        callbacks=cb, 
        verbose=1
    )

    # 5. 載入最佳模型進行預測
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)

    pred_probs = model.predict(X_test_mlp, verbose=0)
    true_key_byte = int(data_dict["key_test"][0, target_byte])
    
    # 計算恢復排名與 GE 曲線
    recovered_key, _, rank_evolution = recover_key_log_rank_id(pred_probs, data_dict["pt_test"], true_key_byte, target_byte)
    ge_curve = compute_ge(pred_probs, data_dict["pt_test"], true_key_byte, target_byte, max_traces=2000)
    plot_ge_curve(ge_curve, target_byte, output_dir, exp_tag=exp_tag)
    
    final_rank = rank_evolution[-1]
    print(f"   => Byte {target_byte} 分析完畢。正解: 0x{true_key_byte:02X}, 模型預測: 0x{recovered_key:02X}")
    print(f"   => 最終排名 (Final Rank): {final_rank} (0 代表破解成功)")

    # 釋放記憶體
    del model
    K.clear_session()
    gc.collect()
    
    return {"byte": target_byte, "true": true_key_byte, "recovered": recovered_key, "final_rank": final_rank}

# =============================================================================
# 6. 主程式進入點
# =============================================================================

def main():
    # GPU 資源優化配置
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    set_seed(42)

    # 參數設定
    FILE_PATH = r"ascadv2-extracted.h5"
    OUTPUT_DIR = "sca_mlp_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    TARGET_BYTES = list(range(16)) # 初始測試 Byte 0，確認無誤後可改為 list(range(16))
    
    # ==========================================
    # 控制台面板 (參數設定區)
    # ==========================================
    NUM_POIS = 50       # 每個 Byte 鎖定 50 個電力特徵最強的採樣點
    BATCH_SIZE = 256     # 批次大小 (如果你的 run_mlp_attack 有開放這個參數)
    # global EPOCHS 
    EPOCHS = 40          # 訓練代數 (如果你的 run_mlp_attack 有開放這個參數)
    MODEL_ARCH = "512_256" # 備註你的模型架構大小，方便辨識

    # 取得當下時間，格式為 月日_時分 
    current_time = datetime.datetime.now().strftime("%m%d_%H%M")

    # 【關鍵】：自動生成這個實驗的專屬標籤
    EXPERIMENT_TAG = f"pois{NUM_POIS}_bs{BATCH_SIZE}_arch{MODEL_ARCH}_{current_time}"
    print("\n" + "*" * 50)
    print(f"目前實驗標籤設定為: {EXPERIMENT_TAG}")
    print("*" * 50 + "\n")

    print(f"正在載入 ASCADv2 資料庫 (50,000 筆) 並進行標準化預處理...")
    X_tr, pt_tr, k_tr, X_v, pt_v, k_v, X_te, pt_te, k_te = load_data_2d(
        FILE_PATH, profiling_limit=50000, attack_limit=2000 # 50000 筆訓練資料，2000 筆攻擊資料，剩下的 1000 筆留作驗證 (val_ratio=0.1)
    )
    
    # 包裝資料字典
    data_dict = {
        "X_train": X_tr, "pt_train": pt_tr, "key_train": k_tr,
        "X_val": X_v, "pt_val": pt_v, "key_val": k_v,
        "X_test": X_te, "pt_test": pt_te, "key_test": k_te
    }

    # 執行迴圈攻擊各個 Byte
    all_results = []
    for byte in TARGET_BYTES:
        # 在這裡將 exp_tag 傳遞給引擎
        res = run_mlp_attack(
            byte, 
            data_dict, 
            OUTPUT_DIR, 
            num_pois=NUM_POIS,
            exp_tag=EXPERIMENT_TAG, # <--- 加上這行
            epochs=EPOCHS # <--- 如果你有開放 epochs 參數，也可以傳入
        )
        all_results.append(res)

    # 打印最終總結報告
    print("\n" + "=" * 60)
    print("側信道攻擊 MLP + SNR 總結報告")
    print("=" * 60)
    for r in all_results:
        status = "破解成功 (Rank 0)" if r['final_rank'] == 0 else f"尚未完全收斂 (Rank {r['final_rank']})"
        # 總結報告也印出 Tag，讓你截圖時一目了然
        print(f"Byte {r['byte']:02d} | 密鑰: 0x{r['true']:02X} | 實驗: [{EXPERIMENT_TAG}] | {status}")

if __name__ == "__main__":
    main()