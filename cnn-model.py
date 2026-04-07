import os
import json
import random
import h5py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gc

from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K

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


def hw_u8(arr):
    arr = arr.astype(np.uint8)
    return np.unpackbits(arr[:, None], axis=1).sum(axis=1).astype(np.uint8)


def is_null_trace(x):
    return np.all(x == 0, axis=1)


def remove_null_traces(traces, plaintext, key, part_name="unknown"):
    null_mask = is_null_trace(traces)
    removed = int(np.sum(null_mask))

    if removed > 0:
        print(f"[{part_name}] 移除 null traces: {removed} 筆")
    else:
        print(f"[{part_name}] 未發現 null traces")

    keep_mask = ~null_mask
    return traces[keep_mask], plaintext[keep_mask], key[keep_mask], removed


def check_metadata_alignment(traces, plaintext, key, part_name="unknown"):
    n_traces = len(traces)
    n_pt = len(plaintext)
    n_key = len(key)

    if not (n_traces == n_pt == n_key):
        raise ValueError(
            f"[{part_name}] metadata 對齊失敗: traces={n_traces}, plaintext={n_pt}, key={n_key}"
        )

    if plaintext.ndim != 2 or plaintext.shape[1] != 16:
        raise ValueError(f"[{part_name}] plaintext shape 異常: {plaintext.shape}")

    if key.ndim != 2 or key.shape[1] != 16:
        raise ValueError(f"[{part_name}] key shape 異常: {key.shape}")

    print(f"[{part_name}] metadata 對齊檢查通過: {n_traces} 筆")


def add_gaussian_noise(traces, sigma=0.0, seed=42):
    if sigma <= 0:
        return traces.copy()

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=traces.shape).astype(np.float32)
    return traces + noise


def load_ascadv2_data(
    file_path,
    profiling_limit=10000,
    attack_limit=2000,
    val_ratio=0.1,
    split_seed=42,
    noise_sigma_train=0.0,
    noise_sigma_val=0.0,
    noise_sigma_test=0.0,
    noise_seed=1234
):
    with h5py.File(file_path, "r") as f:
        X_prof = np.array(f["Profiling_traces/traces"][:profiling_limit], dtype=np.float32)
        pt_prof = np.array(f["Profiling_traces/metadata"]["plaintext"][:profiling_limit], dtype=np.uint8)
        key_prof = np.array(f["Profiling_traces/metadata"]["key"][:profiling_limit], dtype=np.uint8)

        X_attack = np.array(f["Attack_traces/traces"][:attack_limit], dtype=np.float32)
        pt_attack = np.array(f["Attack_traces/metadata"]["plaintext"][:attack_limit], dtype=np.uint8)
        key_attack = np.array(f["Attack_traces/metadata"]["key"][:attack_limit], dtype=np.uint8)

    check_metadata_alignment(X_prof, pt_prof, key_prof, part_name="profiling_before_clean")
    check_metadata_alignment(X_attack, pt_attack, key_attack, part_name="attack_before_clean")

    X_prof, pt_prof, key_prof, removed_prof = remove_null_traces(X_prof, pt_prof, key_prof, part_name="profiling")
    X_attack, pt_attack, key_attack, removed_attack = remove_null_traces(X_attack, pt_attack, key_attack, part_name="attack")

    check_metadata_alignment(X_prof, pt_prof, key_prof, part_name="profiling_after_clean")
    check_metadata_alignment(X_attack, pt_attack, key_attack, part_name="attack_after_clean")

    train_idx, val_idx = train_test_split(
        np.arange(len(X_prof)),
        test_size=val_ratio,
        random_state=split_seed,
        shuffle=True
    )

    X_train = X_prof[train_idx]
    pt_train = pt_prof[train_idx]
    key_train = key_prof[train_idx]

    X_val = X_prof[val_idx]
    pt_val = pt_prof[val_idx]
    key_val = key_prof[val_idx]

    X_test = X_attack
    pt_test = pt_attack
    key_test = key_attack

    mean = np.mean(X_train, axis=0, keepdims=True)
    std = np.std(X_train, axis=0, keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    X_train = add_gaussian_noise(X_train, sigma=noise_sigma_train, seed=noise_seed)
    X_val = add_gaussian_noise(X_val, sigma=noise_sigma_val, seed=noise_seed + 1)
    X_test = add_gaussian_noise(X_test, sigma=noise_sigma_test, seed=noise_seed + 2)

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    protocol = {
        "dataset": "ASCADv2 (STM32)",
        "profiling_limit_before_clean": profiling_limit,
        "attack_limit_before_clean": attack_limit,
        "removed_null_traces_profiling": removed_prof,
        "removed_null_traces_attack": removed_attack,
        "split": {
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "test_size": int(len(X_test)),
            "val_ratio": val_ratio,
            "split_seed": split_seed
        },
        "normalization": "z-score using training set mean/std",
        "noise_injection": {
            "distribution": "Gaussian",
            "noise_sigma_train": noise_sigma_train,
            "noise_sigma_val": noise_sigma_val,
            "noise_sigma_test": noise_sigma_test,
            "noise_seed": noise_seed,
            "injection_stage": "after normalization"
        }
    }

    return {
        "X_train": X_train, "pt_train": pt_train, "key_train": key_train,
        "X_val": X_val, "pt_val": pt_val, "key_val": key_val,
        "X_test": X_test, "pt_test": pt_test, "key_test": key_test,
        "protocol": protocol
    }


def generate_hw_labels(pt, key, target_byte=0):
    sbox_values = AES_Sbox[np.bitwise_xor(pt[:, target_byte], key[:, target_byte])]
    hw_values = hw_u8(sbox_values)
    y = to_categorical(hw_values, num_classes=9)
    return hw_values, y

def compute_class_weight_from_labels(y_labels, num_classes=9):
    counts = np.bincount(y_labels, minlength=num_classes).astype(np.float64)
    total = counts.sum()
    weights = np.sqrt(total / (num_classes * np.maximum(counts, 1.0)))
    return {i: float(weights[i]) for i in range(num_classes)}


def build_hw_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # 第一層
    x = layers.Conv1D(32, kernel_size=11, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=4)(x) # 增加 pool_size 大幅壓縮

    # 第二層
    x = layers.Conv1D(64, kernel_size=7, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=4)(x)

    # 第三層
    x = layers.Conv1D(128, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=4)(x)

    # 第四層 (多加一層繼續壓縮)
    x = layers.Conv1D(256, kernel_size=3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling1D(pool_size=4)(x)

    # 到這裡，時間長度大概被除了 256 倍，Flatten 就不會爆炸了
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x) # 降回 256 減輕負擔
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(9, activation="softmax")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ASCADv2_HW_CNN_Deep")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_acc")]
    )
    return model


def rank_of_true_key(log_probs, true_key):
    sorted_indices = np.argsort(log_probs)[::-1]
    return int(np.where(sorted_indices == true_key)[0][0])


def recover_key_log_rank_hw(pred_probs, plaintexts, true_key, target_byte=0):
    eps = 1e-36
    log_key_scores = np.zeros(256, dtype=np.float64)
    rank_evolution = []

    for i in range(len(pred_probs)):
        pt_b = plaintexts[i, target_byte]
        key_hypotheses = np.arange(256, dtype=np.uint8)
        sbox_vals = AES_Sbox[np.bitwise_xor(key_hypotheses, pt_b)]
        hw_vals = hw_u8(sbox_vals)
        probs_for_all_keys = pred_probs[i, hw_vals]
        log_key_scores += np.log(probs_for_all_keys + eps)
        rank_evolution.append(rank_of_true_key(log_key_scores, true_key))

    recovered_key = int(np.argmax(log_key_scores))
    return recovered_key, log_key_scores, np.array(rank_evolution, dtype=np.int32)


def compute_ge_sr_hw(pred_probs, plaintexts, true_key, target_byte=0, num_attacks=20, max_traces=None, seed=42):
    rng = np.random.default_rng(seed)
    n = len(pred_probs)

    if max_traces is None or max_traces > n:
        max_traces = n

    ge_curve = np.zeros(max_traces, dtype=np.float64)
    sr_curve = np.zeros(max_traces, dtype=np.float64)

    for _ in range(num_attacks):
        indices = rng.permutation(n)[:max_traces]
        sub_preds = pred_probs[indices]
        sub_pts = plaintexts[indices]

        _, _, rank_evolution = recover_key_log_rank_hw(
            sub_preds,
            sub_pts,
            true_key=true_key,
            target_byte=target_byte
        )

        ge_curve += rank_evolution
        sr_curve += (rank_evolution == 0).astype(np.float64)

    ge_curve /= num_attacks
    sr_curve /= num_attacks
    return ge_curve, sr_curve


def plot_curve(values, title, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(1, len(values) + 1), values)
    plt.xlabel("Number of Traces")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def train_and_attack_byte_hw(
    target_byte,
    X_train, pt_train, key_train,
    X_val, pt_val, key_val,
    X_test, pt_test, key_test,
    batch_size=128,
    epochs=30,
    output_dir="sca_hw_results"
):
    print("\n" + "=" * 70)
    print(f"開始處理 Target Byte {target_byte} (HW model)")
    print("=" * 70)

    y_train_true, Y_train = generate_hw_labels(pt_train, key_train, target_byte=target_byte)
    y_val_true, Y_val = generate_hw_labels(pt_val, key_val, target_byte=target_byte)
    y_test_true, Y_test = generate_hw_labels(pt_test, key_test, target_byte=target_byte)

    class_weight = compute_class_weight_from_labels(y_train_true, num_classes=9)
    print(f"Byte {target_byte} class_weight: {class_weight}")

    model = build_hw_model(input_shape=(X_train.shape[1], 1))
    model_path = os.path.join(output_dir, f"best_hw_model_byte_{target_byte}.keras")

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        callbacks.ModelCheckpoint(model_path, monitor="val_loss", save_best_only=True, verbose=0)
    ]

    model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=cb,
        class_weight=class_weight,
        verbose=1
    )

    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)

    pred_probs = model.predict(X_test, verbose=0)
    pred_classes = np.argmax(pred_probs, axis=1)

    acc = accuracy_score(y_test_true, pred_classes)
    f1 = f1_score(y_test_true, pred_classes, average="macro")
    _, keras_acc, keras_top3 = model.evaluate(X_test, Y_test, verbose=0)

    true_key_byte = int(key_test[0, target_byte])

    recovered_key, log_key_scores, rank_evolution = recover_key_log_rank_hw(
        pred_probs,
        pt_test,
        true_key=true_key_byte,
        target_byte=target_byte
    )

    ge_curve, sr_curve = compute_ge_sr_hw(
        pred_probs,
        pt_test,
        true_key=true_key_byte,
        target_byte=target_byte,
        num_attacks=10,
        max_traces=2000,
        seed=42
    )

    final_rank = int(rank_evolution[-1])
    first_rank0 = np.where(rank_evolution == 0)[0]
    first_rank0_trace = int(first_rank0[0] + 1) if len(first_rank0) > 0 else -1

    plot_curve(
        ge_curve,
        title=f"GE Curve (HW) - Byte {target_byte}",
        ylabel="Guessing Entropy",
        save_path=os.path.join(output_dir, f"ge_curve_hw_byte_{target_byte}.png")
    )

    plot_curve(
        sr_curve,
        title=f"SR Curve (HW) - Byte {target_byte}",
        ylabel="Success Rate",
        save_path=os.path.join(output_dir, f"sr_curve_hw_byte_{target_byte}.png")
    )

    result = {
        "byte": target_byte,
        "true_key": true_key_byte,
        "recovered_key": recovered_key,
        "final_rank": final_rank,
        "first_rank0_trace": first_rank0_trace,
        "accuracy": acc,
        "top3": keras_top3,
        "f1": f1
    }

    print(f"Byte {target_byte} true key:      0x{true_key_byte:02X}")
    print(f"Byte {target_byte} recovered key: 0x{recovered_key:02X}")
    print(f"Byte {target_byte} final rank:    {final_rank}")
    print(f"Byte {target_byte} accuracy:      {acc * 100:.2f}%")
    print(f"Byte {target_byte} top-3 acc:     {keras_top3 * 100:.2f}%")
    print(f"Byte {target_byte} macro F1:      {f1:.4f}")
    print(f"Byte {target_byte} first rank-0 trace: {first_rank0_trace if first_rank0_trace != -1 else '未達成'}")

    del model # 釋放模型資源
    K.clear_session()
    gc.collect()
    return result


def main():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU 動態顯存配置已啟用")
        except RuntimeError as e:
            print(f"GPU 配置錯誤: {e}")
    set_seed(42)

    FILE_PATH = r"ascadv2-extracted.h5"
    OUTPUT_DIR = "sca_hw_results"
    # TARGET_BYTES = list(range(16))
    TARGET_BYTES = [0]

    PROFILE_LIMIT = 10000
    ATTACK_LIMIT = 2000
    VAL_RATIO = 0.1
    SPLIT_SEED = 42

    NOISE_SIGMA_TRAIN = 0.0
    NOISE_SIGMA_VAL = 0.0
    NOISE_SIGMA_TEST = 0.0
    NOISE_SEED = 1234

    BATCH_SIZE = 128
    EPOCHS = 30

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_ascadv2_data(
        file_path=FILE_PATH,
        profiling_limit=PROFILE_LIMIT,
        attack_limit=ATTACK_LIMIT,
        val_ratio=VAL_RATIO,
        split_seed=SPLIT_SEED,
        noise_sigma_train=NOISE_SIGMA_TRAIN,
        noise_sigma_val=NOISE_SIGMA_VAL,
        noise_sigma_test=NOISE_SIGMA_TEST,
        noise_seed=NOISE_SEED
    )

    with open(os.path.join(OUTPUT_DIR, "noise_protocol.json"), "w", encoding="utf-8") as f:
        json.dump(data["protocol"], f, indent=2, ensure_ascii=False)

    all_results = []

    for target_byte in TARGET_BYTES:
        result = train_and_attack_byte_hw(
            target_byte=target_byte,
            X_train=data["X_train"], pt_train=data["pt_train"], key_train=data["key_train"],
            X_val=data["X_val"], pt_val=data["pt_val"], key_val=data["key_val"],
            X_test=data["X_test"], pt_test=data["pt_test"], key_test=data["key_test"],
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            output_dir=OUTPUT_DIR
        )
        all_results.append(result)

    true_key_hex = " ".join([f"{int(data['key_test'][0, i]):02X}" for i in range(16)])
    recovered_key_hex = " ".join([f"{r['recovered_key']:02X}" for r in all_results])

    correct_bytes = sum(int(r["true_key"] == r["recovered_key"]) for r in all_results)

    print("\n" + "=" * 80)
    print("完整 AES-128 Key Recovery 結果 (HW model)")
    print("=" * 80)
    print(f"True Key:      {true_key_hex}")
    print(f"Recovered Key: {recovered_key_hex}")
    print(f"Correct Bytes: {correct_bytes}/16")
    print("=" * 80)

    with open(os.path.join(OUTPUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("ASCADv2 HW Model Summary\n")
        f.write("=" * 60 + "\n")
        f.write(json.dumps(data["protocol"], indent=2, ensure_ascii=False))
        f.write("\n\n")
        f.write(f"True Key:      {true_key_hex}\n")
        f.write(f"Recovered Key: {recovered_key_hex}\n")
        f.write(f"Correct Bytes: {correct_bytes}/16\n\n")

        for r in all_results:
            f.write(f"Byte {r['byte']:02d}\n")
            f.write(f"  True Key:         0x{r['true_key']:02X}\n")
            f.write(f"  Recovered Key:    0x{r['recovered_key']:02X}\n")
            f.write(f"  Final Rank:       {r['final_rank']}\n")
            f.write(f"  First Rank-0:     {r['first_rank0_trace']}\n")
            f.write(f"  Accuracy:         {r['accuracy'] * 100:.2f}%\n")
            f.write(f"  Top-3 Accuracy:   {r['top3'] * 100:.2f}%\n")
            f.write(f"  Macro F1:         {r['f1']:.4f}\n")
            f.write("-" * 40 + "\n")


if __name__ == "__main__":
    main()