import os
import random
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gc
import datetime
import json

from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

# =============================================================================
# 1. 常數與初始化設定
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
# 2. 資料載入與預處理
# =============================================================================

def is_null_trace(x):
    return np.all(x == 0, axis=1)

def remove_null_traces(traces, plaintext, key):
    null_mask = is_null_trace(traces)
    keep_mask = ~null_mask
    return traces[keep_mask], plaintext[keep_mask], key[keep_mask]

def load_data_2d(file_path, profiling_limit=50000, attack_limit=2000, val_ratio=0.1, split_seed=42):
    with h5py.File(file_path, "r") as f:
        X_prof = np.array(f["Profiling_traces/traces"][:profiling_limit], dtype=np.float32)
        pt_prof = np.array(f["Profiling_traces/metadata"]["plaintext"][:profiling_limit], dtype=np.uint8)
        key_prof = np.array(f["Profiling_traces/metadata"]["key"][:profiling_limit], dtype=np.uint8)

        X_attack = np.array(f["Attack_traces/traces"][:attack_limit], dtype=np.float32)
        pt_attack = np.array(f["Attack_traces/metadata"]["plaintext"][:attack_limit], dtype=np.uint8)
        key_attack = np.array(f["Attack_traces/metadata"]["key"][:attack_limit], dtype=np.uint8)

    X_prof, pt_prof, key_prof = remove_null_traces(X_prof, pt_prof, key_prof)
    X_attack, pt_attack, key_attack = remove_null_traces(X_attack, pt_attack, key_attack)

    train_idx, val_idx = train_test_split(
        np.arange(len(X_prof)), test_size=val_ratio, random_state=split_seed, shuffle=True
    )

    X_train, pt_train, key_train = X_prof[train_idx], pt_prof[train_idx], key_prof[train_idx]
    X_val, pt_val, key_val = X_prof[val_idx], pt_prof[val_idx], key_prof[val_idx]
    X_test, pt_test, key_test = X_attack, pt_attack, key_attack

    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True) + 1e-8
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, pt_train, key_train, X_val, pt_val, key_val, X_test, pt_test, key_test

# =============================================================================
# 3. 標籤與評估指標 (GE & SR)
# =============================================================================

def generate_id_labels(pt, key, target_byte=0):
    sbox_values = AES_Sbox[np.bitwise_xor(pt[:, target_byte], key[:, target_byte])]
    y = to_categorical(sbox_values, num_classes=256)
    return sbox_values, y

def rank_of_true_key(log_probs, true_key):
    sorted_indices = np.argsort(log_probs)[::-1]
    return int(np.where(sorted_indices == true_key)[0][0])

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
        rank_evolution.append(rank_of_true_key(log_key_scores, true_key))

    recovered_key = int(np.argmax(log_key_scores))
    return recovered_key, log_key_scores, np.array(rank_evolution, dtype=np.int32)

def compute_ge_sr(pred_probs, plaintexts, true_key, target_byte=0, num_attacks=20, max_traces=None):
    """ 同時計算 Guessing Entropy (GE) 與 Success Rate (SR) """
    rng = np.random.default_rng(42)
    n = len(pred_probs)
    if max_traces is None or max_traces > n: max_traces = n

    ge_curve = np.zeros(max_traces, dtype=np.float64)
    sr_curve = np.zeros(max_traces, dtype=np.float64)

    for _ in range(num_attacks):
        indices = rng.permutation(n)[:max_traces]
        sub_preds, sub_pts = pred_probs[indices], plaintexts[indices]
        _, _, rank_evolution = recover_key_log_rank_id(sub_preds, sub_pts, true_key, target_byte)
        ge_curve += rank_evolution
        sr_curve += (rank_evolution == 0).astype(np.float64)

    return ge_curve / num_attacks, sr_curve / num_attacks

# =============================================================================
# 4. 圖表輸出 (Metrics, Loss, ACC)
# =============================================================================

def plot_training_history(history, output_dir, target_byte):
    """ 繪製 Cross Entropy Loss 與 Accuracy 訓練曲線 """
    epochs = range(1, len(history.history["loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左圖: Cross Entropy Loss
    ax1.plot(epochs, history.history["loss"], "b-", label="Train Loss", linewidth=2)
    ax1.plot(epochs, history.history["val_loss"], "r--", label="Val Loss", linewidth=2)
    ax1.set_title(f"Cross Entropy Loss (Byte {target_byte})")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    # 右圖: Accuracy
    ax2.plot(epochs, history.history["accuracy"], "b-", label="Train Acc", linewidth=2)
    ax2.plot(epochs, history.history["val_accuracy"], "r--", label="Val Acc", linewidth=2)
    ax2.set_title(f"Accuracy (Byte {target_byte})")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"training_history_byte_{target_byte}.png"), dpi=150)
    plt.close()

def plot_sca_metrics(ge_curve, sr_curve, target_byte, output_dir):
    """ 繪製 Guessing Entropy 與 Success Rate 攻擊指標圖 """
    traces = range(1, len(ge_curve) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左圖: GE
    ax1.plot(traces, ge_curve, "g-", linewidth=2)
    ax1.axhline(y=0, color="r", linestyle=":", alpha=0.5)
    ax1.set_title(f"Guessing Entropy (Byte {target_byte})")
    ax1.set_xlabel("Attack Traces")
    ax1.set_ylabel("Rank")
    ax1.grid(True)

    # 右圖: SR
    ax2.plot(traces, sr_curve, "m-", linewidth=2)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title(f"Success Rate (Byte {target_byte})")
    ax2.set_xlabel("Attack Traces")
    ax2.set_ylabel("Probability of Rank=0")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"sca_metrics_byte_{target_byte}.png"), dpi=150)
    plt.close()

def extract_pois_snr(traces, labels, target_byte, output_dir, exp_tag="", num_classes=256, num_pois=200):
    """ SNR POI 提取 (因應 desync50，提升取點數量以覆蓋偏移範圍) """
    print(f"   [特徵工程] 正在計算 SNR，提取 Byte {target_byte} 的 {num_pois} 個特徵點...")
    num_samples = traces.shape[1]
    means, vars_ = np.zeros((num_classes, num_samples)), np.zeros((num_classes, num_samples))
    
    for i in range(num_classes):
        idx = (labels == i)
        if np.sum(idx) > 0:
            means[i], vars_[i] = np.mean(traces[idx], axis=0), np.var(traces[idx], axis=0)
            
    snr = np.var(means, axis=0) / (np.mean(vars_, axis=0) + 1e-8)
    poi_indices = np.sort(np.argsort(snr)[-num_pois:])
    
    plt.figure(figsize=(10, 4))
    plt.plot(snr, color='lightblue', alpha=0.7, label='Full SNR Curve')
    plt.scatter(poi_indices, snr[poi_indices], color='red', s=15, zorder=5, label=f'Top {num_pois} POIs')
    plt.title(f"SNR & Selected POIs (Byte {target_byte})")
    plt.xlabel("Time Sample Point")
    plt.ylabel("SNR Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"snr_pois_byte_{target_byte}.png"), dpi=150)
    plt.close()

    return poi_indices

# =============================================================================
# 5. MLP 模型架構與訓練流程
# =============================================================================

def build_mlp_model(input_shape):
    from tensorflow.keras import regularizers # 引入正則化模組
    
    # 設定 L2 正則化強度 (懲罰過度死背的權重)
    l2_reg = regularizers.l2(1e-4) 

    inputs = layers.Input(shape=input_shape)
    
    # 1. 將神經元數量減半，降低死背能力
    x = layers.Dense(256, activation="relu", kernel_regularizer=l2_reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x) # 2. 提高 Dropout 到 40%
    
    x = layers.Dense(128, activation="relu", kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(256, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        # 3. 稍微調低學習率，讓它學得更謹慎
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4), 
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def run_mlp_attack(target_byte, data_dict, output_dir, num_pois=200, epochs=40):
    print("\n" + "=" * 60)
    print(f"執行專屬 MLP 流程 - 目標 Byte: {target_byte} (Desync50)")
    print("=" * 60)
    
    y_train_true, Y_train = generate_id_labels(data_dict["pt_train"], data_dict["key_train"], target_byte)
    _, Y_val = generate_id_labels(data_dict["pt_val"], data_dict["key_val"], target_byte)
    y_test_true, Y_test = generate_id_labels(data_dict["pt_test"], data_dict["key_test"], target_byte)

    # poi_indices = extract_pois_snr(data_dict["X_train"], y_train_true, target_byte, output_dir, num_pois=num_pois)
    # X_train_mlp, X_val_mlp, X_test_mlp = data_dict["X_train"][:, poi_indices], data_dict["X_val"][:, poi_indices], data_dict["X_test"][:, poi_indices]
    
    # ✅ 直接使用全部的 700 個點！
    X_train_mlp = data_dict["X_train"]
    X_val_mlp = data_dict["X_val"]
    X_test_mlp = data_dict["X_test"]

    # 動態取得輸入維度 (也就是 700)
    input_dim = X_train_mlp.shape[1] 
    print(f"   [進度報告] 直接使用全波形，輸入維度: {input_dim}")

    # model = build_mlp_model(input_shape=(num_pois,))
    model = build_mlp_model(input_shape=(input_dim,))
    model_path = os.path.join(output_dir, f"best_mlp_byte_{target_byte}.keras")
    
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=0)
    ]

    history = model.fit(
        X_train_mlp, Y_train, 
        validation_data=(X_val_mlp, Y_val), 
        batch_size=256, 
        epochs=epochs,
        callbacks=cb, 
        verbose=1
    )

    plot_training_history(history, output_dir, target_byte)

    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)

    # 取得最終評估指標 (Loss, ACC)
    val_loss, val_acc = model.evaluate(X_test_mlp, Y_test, verbose=0)
    
    pred_probs = model.predict(X_test_mlp, verbose=0)
    true_key_byte = int(data_dict["key_test"][0, target_byte])
    
    # 計算 GE 與 SR
    recovered_key, _, rank_evolution = recover_key_log_rank_id(pred_probs, data_dict["pt_test"], true_key_byte, target_byte)
    ge_curve, sr_curve = compute_ge_sr(pred_probs, data_dict["pt_test"], true_key_byte, target_byte, num_attacks=20, max_traces=2000)
    
    plot_sca_metrics(ge_curve, sr_curve, target_byte, output_dir)
    
    final_rank = rank_evolution[-1]
    final_sr = sr_curve[-1]
    
    print(f"   => Byte {target_byte} 分析完畢。正解: 0x{true_key_byte:02X}, 模型預測: 0x{recovered_key:02X}")
    print(f"   => 最終排名 (Final Rank): {final_rank}")

    del model
    K.clear_session()
    gc.collect()
    
    return {
        "byte": target_byte, 
        "true": true_key_byte, 
        "recovered": recovered_key, 
        "final_rank": int(final_rank),
        "val_loss": float(val_loss),
        "val_acc": float(val_acc),
        "final_sr": float(final_sr)
    }

# =============================================================================
# 6. 主程式進入點
# =============================================================================

def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    set_seed(42)

    # ==========================================
    # 建立動態時間戳記的輸出資料夾
    # ==========================================
    BASE_DIR = "mlp-model-result"
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(BASE_DIR, f"result_{current_time}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    FILE_PATH = r"ASCAD.h5"  # <--- 請確認這裡的路徑是否符合你的電腦環境
    
    TARGET_BYTES = [0, 1, 2, 3] # 先測前 4 個 Byte 節省時間
    NUM_POIS = 50 # Desync50 特調：放大取樣點以涵蓋時間偏移
    EPOCHS = 40

    print("\n" + "*" * 50)
    print(f"本次訓練與圖表將獨立儲存於: {OUTPUT_DIR}")
    print("*" * 50 + "\n")

    print(f"正在載入 ASCADv1 Desync50 資料庫...")
    X_tr, pt_tr, k_tr, X_v, pt_v, k_v, X_te, pt_te, k_te = load_data_2d(
        FILE_PATH, profiling_limit=50000, attack_limit=2000
    )
    
    data_dict = {
        "X_train": X_tr, "pt_train": pt_tr, "key_train": k_tr,
        "X_val": X_v, "pt_val": pt_v, "key_val": k_v,
        "X_test": X_te, "pt_test": pt_te, "key_test": k_te
    }

    all_results = []
    for byte in TARGET_BYTES:
        res = run_mlp_attack(byte, data_dict, OUTPUT_DIR, num_pois=NUM_POIS, epochs=EPOCHS)
        all_results.append(res)

    # 將所有指標儲存為 JSON 檔
    results_path = os.path.join(OUTPUT_DIR, "all_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("側信道攻擊 MLP (Desync50) 總結報告")
    print("=" * 80)
    header = f"{'Byte':>5} | {'True':>6} | {'Recovered':>10} | {'Rank':>5} | {'Val_Loss':>8} | {'Val_Acc%':>8} | {'Final_SR%':>9}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['byte']:>5} | 0x{r['true']:02X}   | "
            f"0x{r['recovered']:02X}       | {r['final_rank']:>5} | "
            f"{r['val_loss']:>8.4f} | {r['val_acc']*100:>7.2f}% | {r['final_sr']*100:>8.2f}%"
        )
    print(f"\n所有圖表與 JSON 已存入: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()