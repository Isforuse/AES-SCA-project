import os
import json
import random
from datetime import datetime

import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


# AES S-Box：用來計算中間值 SBox(PT xor K)
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
    """
    固定亂數種子，讓每次實驗結果更容易重現。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def hw_u8(arr):
    """
    計算每個 uint8 數值的 Hamming Weight（bit 中 1 的個數）。
    例如：
        0x00 -> 0
        0x01 -> 1
        0x03 -> 2
        0xFF -> 8
    """
    arr = arr.astype(np.uint8)
    return np.unpackbits(arr[:, None], axis=1).sum(axis=1).astype(np.uint8)


def load_ascad_v1(file_path, num_train=50000, num_attack=10000, val_ratio=0.1, seed=42):
    """
    載入 ASCAD v1 資料集。

    參數:
        file_path: ASCAD.h5 路徑
        num_train: 取多少 profiling traces 當訓練來源
        num_attack: 取多少 attack traces 當測試/攻擊資料
        val_ratio: 從 profiling traces 中再切出 validation 的比例
        seed: 固定切分用亂數種子

    回傳:
        X_train, pt_train, key_train
        X_val, pt_val, key_val
        X_attack, pt_attack, key_attack
    """
    with h5py.File(file_path, "r") as f:
        # Profiling traces：拿來訓練模型
        X_prof = np.array(f["Profiling_traces/traces"][:num_train], dtype=np.float32)
        pt_prof = np.array(f["Profiling_traces/metadata"]["plaintext"][:num_train], dtype=np.uint8)
        key_prof = np.array(f["Profiling_traces/metadata"]["key"][:num_train], dtype=np.uint8)

        # Attack traces：拿來做分類測試與 key recovery
        X_attack = np.array(f["Attack_traces/traces"][:num_attack], dtype=np.float32)
        pt_attack = np.array(f["Attack_traces/metadata"]["plaintext"][:num_attack], dtype=np.uint8)
        key_attack = np.array(f["Attack_traces/metadata"]["key"][:num_attack], dtype=np.uint8)

    # 將 profiling traces 固定切為 train / validation
    idx_train, idx_val = train_test_split(
        np.arange(len(X_prof)),
        test_size=val_ratio,
        random_state=seed,
        shuffle=True
    )

    X_train = X_prof[idx_train]
    X_val = X_prof[idx_val]
    pt_train = pt_prof[idx_train]
    pt_val = pt_prof[idx_val]
    key_train = key_prof[idx_train]
    key_val = key_prof[idx_val]

    # 用 training set 的平均與標準差做 z-score normalization
    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_attack = (X_attack - mean) / std

    # Conv1D 需要 channel 維度，因此最後補一維
    # 例如： (45000, 700) -> (45000, 700, 1)
    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_attack = X_attack[..., np.newaxis]

    return X_train, pt_train, key_train, X_val, pt_val, key_val, X_attack, pt_attack, key_attack


def generate_labels(pt, key, target_byte=0, label_mode="identity"):
    """
    依照 plaintext 與 key 計算分類標籤。

    label_mode:
        identity -> 預測 SBox(PT xor K)，共 256 類
        hw       -> 預測 HW(SBox(PT xor K))，共 9 類
    """
    sbox_values = AES_Sbox[np.bitwise_xor(pt[:, target_byte], key[:, target_byte])]

    if label_mode == "identity":
        y_true = sbox_values.astype(np.uint8)
        y_onehot = to_categorical(y_true, num_classes=256)
        num_classes = 256

    elif label_mode == "hw":
        y_true = hw_u8(sbox_values)
        y_onehot = to_categorical(y_true, num_classes=9)
        num_classes = 9

    else:
        raise ValueError("label_mode 必須是 'identity' 或 'hw'")

    return y_true, y_onehot, num_classes


def compute_class_weight_sqrt(y_labels, num_classes):
    """
    給 HW model 使用的較溫和 class_weight。

    原因：
    HW 類別分布不均，像 HW=4 會遠多於 HW=0 或 8，
    若完全不補償，模型可能會偏猜中間類別。
    """
    counts = np.bincount(y_labels, minlength=num_classes).astype(np.float64)
    total = counts.sum()

    raw = total / (num_classes * np.maximum(counts, 1.0))
    weights = np.sqrt(raw)

    # 限制權重範圍，避免極端類別權重過大導致訓練不穩
    weights = np.clip(weights, 0.5, 5.0)

    return {i: float(weights[i]) for i in range(num_classes)}


def build_cnn_model(input_shape, num_classes):
    """
    建立 ASCAD v1 展示用 CNN 模型。
    """
    inputs = layers.Input(shape=input_shape)

    # 卷積區塊 1：抓取較大範圍的局部波形特徵
    x = layers.Conv1D(32, kernel_size=11, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.AveragePooling1D(pool_size=2)(x)

    # 卷積區塊 2：抽取中階特徵
    x = layers.Conv1D(64, kernel_size=7, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.AveragePooling1D(pool_size=2)(x)

    # 卷積區塊 3：抽取高階特徵
    x = layers.Conv1D(128, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.AveragePooling1D(pool_size=2)(x)

    # 將整段 feature map 壓成固定長度向量
    x = layers.GlobalAveragePooling1D()(x)

    # 全連接分類前的特徵整合
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    # 輸出層：identity 為 256 類，HW 為 9 類
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ASCADv1_CNN")

    top_k = 5 if num_classes >= 256 else 3

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=top_k, name="top_k_acc")
        ]
    )

    return model


def rank_of_true_key(scores, true_key):
    """
    根據 256 個 key hypothesis 的分數，找出真實 key 的排名。
    rank=0 表示真實 key 排第一名。
    """
    sorted_idx = np.argsort(scores)[::-1]
    return int(np.where(sorted_idx == true_key)[0][0])


def recover_key_identity(pred_probs, plaintexts, true_key, target_byte=0):
    """
    Identity model 的 key ranking。

    對每條 attack trace：
        對 256 個 key hypothesis 計算 SBox(PT xor key_guess)
        再從模型輸出的 256 類機率中取對應類別機率
        最後做 log-likelihood 累加
    """
    eps = 1e-36
    log_scores = np.zeros(256, dtype=np.float64)
    rank_evolution = []

    for i in range(len(pred_probs)):
        pt_b = plaintexts[i, target_byte]
        key_hyp = np.arange(256, dtype=np.uint8)

        classes = AES_Sbox[np.bitwise_xor(key_hyp, pt_b)]
        probs = pred_probs[i, classes]

        log_scores += np.log(probs + eps)
        rank_evolution.append(rank_of_true_key(log_scores, true_key))

    recovered_key = int(np.argmax(log_scores))
    return recovered_key, log_scores, np.array(rank_evolution)


def recover_key_hw(pred_probs, plaintexts, true_key, target_byte=0):
    """
    HW model 的 key ranking。

    對每條 attack trace：
        對 256 個 key hypothesis 計算 HW(SBox(PT xor key_guess))
        再從模型輸出的 9 類機率中取對應類別機率
        最後做 log-likelihood 累加
    """
    eps = 1e-36
    log_scores = np.zeros(256, dtype=np.float64)
    rank_evolution = []

    for i in range(len(pred_probs)):
        pt_b = plaintexts[i, target_byte]
        key_hyp = np.arange(256, dtype=np.uint8)

        sbox_vals = AES_Sbox[np.bitwise_xor(key_hyp, pt_b)]
        hw_vals = hw_u8(sbox_vals)
        probs = pred_probs[i, hw_vals]

        log_scores += np.log(probs + eps)
        rank_evolution.append(rank_of_true_key(log_scores, true_key))

    recovered_key = int(np.argmax(log_scores))
    return recovered_key, log_scores, np.array(rank_evolution)


def compute_ge_sr(pred_probs, plaintexts, true_key, target_byte=0, label_mode="identity",
                  num_attacks=20, max_traces=2000, seed=42):
    """
    計算 Guessing Entropy (GE) 與 Success Rate (SR)。

    GE: 多次隨機攻擊下，真實 key 的平均 rank，越低越好
    SR: 多次隨機攻擊下，真實 key 排第一的比例，越高越好
    """
    rng = np.random.default_rng(seed)
    n = len(pred_probs)
    max_traces = min(max_traces, n)

    ge_curve = np.zeros(max_traces, dtype=np.float64)
    sr_curve = np.zeros(max_traces, dtype=np.float64)

    for _ in range(num_attacks):
        idx = rng.permutation(n)[:max_traces]
        sub_probs = pred_probs[idx]
        sub_pts = plaintexts[idx]

        if label_mode == "identity":
            _, _, rank_evo = recover_key_identity(sub_probs, sub_pts, true_key, target_byte)
        else:
            _, _, rank_evo = recover_key_hw(sub_probs, sub_pts, true_key, target_byte)

        ge_curve += rank_evo
        sr_curve += (rank_evo == 0).astype(np.float64)

    ge_curve /= num_attacks
    sr_curve /= num_attacks

    return ge_curve, sr_curve


def plot_curve(values, title, ylabel, save_path):
    """
    繪製 GE / SR 曲線。
    """
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(values) + 1), values)
    plt.xlabel("Number of Traces")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_training_history(history, save_dir):
    """
    繪製 training / validation curves。
    會輸出：
        1. loss curve
        2. accuracy curve
        3. top-k accuracy curve
    """
    hist = history.history

    # Loss 曲線
    plt.figure(figsize=(8, 5))
    plt.plot(hist["loss"], label="train_loss")
    plt.plot(hist["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_validation_loss.png"), dpi=200)
    plt.close()

    # Accuracy 曲線
    if "accuracy" in hist and "val_accuracy" in hist:
        plt.figure(figsize=(8, 5))
        plt.plot(hist["accuracy"], label="train_accuracy")
        plt.plot(hist["val_accuracy"], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training / Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_validation_accuracy.png"), dpi=200)
        plt.close()

    # Top-k Accuracy 曲線
    if "top_k_acc" in hist and "val_top_k_acc" in hist:
        plt.figure(figsize=(8, 5))
        plt.plot(hist["top_k_acc"], label="train_top_k_acc")
        plt.plot(hist["val_top_k_acc"], label="val_top_k_acc")
        plt.xlabel("Epoch")
        plt.ylabel("Top-k Accuracy")
        plt.title("Training / Validation Top-k Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_validation_topk.png"), dpi=200)
        plt.close()


if __name__ == "__main__":
    # 固定亂數種子
    set_seed(42)

    # ===================== 基本參數設定 =====================
    FILE_PATH = r"ASCAD.h5"
    TARGET_BYTE = 2
    LABEL_MODE = "hw"   # 可改成 "hw"
    NUM_TRAIN = 50000
    NUM_ATTACK = 10000
    VAL_RATIO = 0.1
    BATCH_SIZE = 128
    EPOCHS = 50

    # ===================== 建立時間資料夾 =====================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    BASE_OUTPUT_DIR = "cnn-model-ascadv1-result"
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, f"result_{timestamp}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ===================== 儲存實驗參數 =====================
    experiment_config = {
        "file_path": FILE_PATH,
        "target_byte": TARGET_BYTE,
        "label_mode": LABEL_MODE,
        "num_train": NUM_TRAIN,
        "num_attack": NUM_ATTACK,
        "val_ratio": VAL_RATIO,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "output_dir": OUTPUT_DIR,
        "model_type": "CNN",
        "dataset": "ASCAD v1"
    }

    with open(os.path.join(OUTPUT_DIR, "experiment_config.json"), "w", encoding="utf-8") as f:
        json.dump(experiment_config, f, indent=2, ensure_ascii=False)

    # ===================== 載入資料 =====================
    X_train, pt_train, key_train, X_val, pt_val, key_val, X_attack, pt_attack, key_attack = load_ascad_v1(
        FILE_PATH,
        num_train=NUM_TRAIN,
        num_attack=NUM_ATTACK,
        val_ratio=VAL_RATIO,
        seed=42
    )

    # ===================== 產生標籤 =====================
    y_train_true, Y_train, num_classes = generate_labels(pt_train, key_train, TARGET_BYTE, LABEL_MODE)
    y_val_true, Y_val, _ = generate_labels(pt_val, key_val, TARGET_BYTE, LABEL_MODE)
    y_attack_true, Y_attack, _ = generate_labels(pt_attack, key_attack, TARGET_BYTE, LABEL_MODE)

    # ===================== 建立模型 =====================
    model = build_cnn_model((X_train.shape[1], 1), num_classes)
    model.summary()

    # 將模型摘要另存成 txt
    with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    # ===================== callback 設定 =====================
    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, "best_cnn_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    # ===================== 準備 fit 參數 =====================
    fit_kwargs = dict(
        x=X_train,
        y=Y_train,
        validation_data=(X_val, Y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=cb,
        verbose=1
    )

    # HW 模式時加入 class_weight
    if LABEL_MODE == "hw":
        class_weight = compute_class_weight_sqrt(y_train_true, num_classes=9)
        print("class_weight =", class_weight)
        fit_kwargs["class_weight"] = class_weight

    # ===================== 開始訓練 =====================
    history = model.fit(**fit_kwargs)

    # 儲存 history
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2, ensure_ascii=False)

    # 繪製 training / validation curves
    plot_training_history(history, OUTPUT_DIR)

    # 載入最佳模型
    model = tf.keras.models.load_model(os.path.join(OUTPUT_DIR, "best_cnn_model.keras"))

    # ===================== 分類評估 =====================
    pred_probs = model.predict(X_attack, verbose=0)
    pred_classes = np.argmax(pred_probs, axis=1)

    acc = accuracy_score(y_attack_true, pred_classes)
    f1 = f1_score(y_attack_true, pred_classes, average="macro")
    _, keras_acc, keras_topk = model.evaluate(X_attack, Y_attack, verbose=0)

    # ===================== Key Recovery =====================
    true_key_byte = int(key_attack[0, TARGET_BYTE])

    if LABEL_MODE == "identity":
        recovered_key, log_scores, rank_evo = recover_key_identity(
            pred_probs, pt_attack, true_key_byte, TARGET_BYTE
        )
    else:
        recovered_key, log_scores, rank_evo = recover_key_hw(
            pred_probs, pt_attack, true_key_byte, TARGET_BYTE
        )

    # ===================== GE / SR =====================
    ge_curve, sr_curve = compute_ge_sr(
        pred_probs,
        pt_attack,
        true_key=true_key_byte,
        target_byte=TARGET_BYTE,
        label_mode=LABEL_MODE,
        num_attacks=20,
        max_traces=2000,
        seed=42
    )

    final_rank = int(rank_evo[-1])
    first_rank0 = np.where(rank_evo == 0)[0]
    first_rank0_trace = int(first_rank0[0] + 1) if len(first_rank0) > 0 else -1

    # 繪製 GE / SR 圖
    plot_curve(
        ge_curve,
        f"GE Curve - ASCAD v1 CNN ({LABEL_MODE})",
        "Guessing Entropy",
        os.path.join(OUTPUT_DIR, "ge_curve.png")
    )

    plot_curve(
        sr_curve,
        f"SR Curve - ASCAD v1 CNN ({LABEL_MODE})",
        "Success Rate",
        os.path.join(OUTPUT_DIR, "sr_curve.png")
    )

    # ===================== 儲存結果摘要 =====================
    result_summary = {
        "label_mode": LABEL_MODE,
        "target_byte": TARGET_BYTE,
        "true_key_byte": f"0x{true_key_byte:02X}",
        "recovered_key_byte": f"0x{recovered_key:02X}",
        "final_key_rank": int(final_rank),
        "first_rank0_trace": int(first_rank0_trace),
        "accuracy": float(acc),
        "top_k_accuracy": float(keras_topk),
        "macro_f1": float(f1)
    }

    with open(os.path.join(OUTPUT_DIR, "result_summary.json"), "w", encoding="utf-8") as f:
        json.dump(result_summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_DIR, "result_summary.txt"), "w", encoding="utf-8") as f:
        f.write("ASCAD v1 CNN 展示結果\n")
        f.write("=" * 50 + "\n")
        f.write(f"LABEL_MODE:         {LABEL_MODE}\n")
        f.write(f"Target byte:        {TARGET_BYTE}\n")
        f.write(f"True key byte:      0x{true_key_byte:02X}\n")
        f.write(f"Recovered key byte: 0x{recovered_key:02X}\n")
        f.write(f"Final key rank:     {final_rank}\n")
        f.write(f"First rank-0 trace: {first_rank0_trace if first_rank0_trace != -1 else '未達成'}\n")
        f.write(f"Accuracy:           {acc * 100:.2f}%\n")
        f.write(f"Top-k Accuracy:     {keras_topk * 100:.2f}%\n")
        f.write(f"Macro F1:           {f1:.4f}\n")

    # ===================== 終端輸出 =====================
    print("\n" + "=" * 60)
    print("ASCAD v1 CNN 展示結果")
    print("=" * 60)
    print(f"LABEL_MODE:            {LABEL_MODE}")
    print(f"Target byte:           {TARGET_BYTE}")
    print(f"True key byte:         0x{true_key_byte:02X}")
    print(f"Recovered key byte:    0x{recovered_key:02X}")
    print(f"Final key rank:        {final_rank}")
    print(f"First rank-0 trace:    {first_rank0_trace if first_rank0_trace != -1 else '未達成'}")
    print(f"Accuracy:              {acc * 100:.2f}%")
    print(f"Top-k Accuracy:        {keras_topk * 100:.2f}%")
    print(f"Macro F1:              {f1:.4f}")
    print("=" * 60)