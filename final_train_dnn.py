"""
TRAINING SCRIPT - NETRAL + IMAGE-LEVEL AUGMENTATION BEFORE HOG
==============================================================
Tujuan:
- Satu model HOG + DNN untuk dipakai oleh:
  1) Haar Cascade -> HOG -> DNN
  2) HoG Detector -> HOG -> DNN
- Tidak memakai MediaPipe
- Augmentasi dilakukan pada level gambar sebelum ekstraksi HOG
"""

import os
import json
import random
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from utils_io import list_images_in_folders
import matplotlib.pyplot as plt

# =========================================================
# KONFIGURASI
# =========================================================
IMG_SIZE = 128
HOG_PARAMS = {
    "orientations": 9,
    "pixels_per_cell": (8, 8),
    "cells_per_block": (2, 2),
    "block_norm": "L2-Hys",
    "transform_sqrt": True
}
EPOCHS = 50
BATCH_SIZE = 32
REGULARIZATION = 0.005
EARLY_STOPPING_PATIENCE = 6
USE_DATA_AUG = True
AUGMENT_FACTOR = 2
RANDOM_STATE = 42   

TRAIN_DIR = "dataset_split/train"
VAL_DIR = "dataset_split/val"
OUT_DIR = "models_regularized_imgaug"
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# =========================================================
# PREPROCESSING & HOG
# =========================================================
def preprocess_image(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    return gray

def compute_hog(gray):
    return hog(gray, **HOG_PARAMS)

# =========================================================
# IMAGE-LEVEL AUGMENTATION
# =========================================================
def augment_image(gray):
    img = gray.copy()

    # Horizontal flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    # Rotation ringan
    angle = random.uniform(-10, 10)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )

    # Brightness/contrast ringan
    alpha = random.uniform(0.9, 1.1)   # contrast
    beta = random.uniform(-15, 15)     # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Gaussian noise ringan
    noise = np.random.normal(0, 5, img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img

# =========================================================
# LOAD DATASET
# =========================================================
def load_dataset(data_dir, augment=False):
    paths, labels, class_names = list_images_in_folders(data_dir)
    X, y = [], []
    miss = 0
    n_original_valid = 0

    for p, lbl in zip(paths, labels):
        img = cv2.imread(p)
        if img is None:
            miss += 1
            continue

        try:
            gray = preprocess_image(img)

            # Sampel asli
            feat = compute_hog(gray)
            X.append(feat)
            y.append(lbl)
            n_original_valid += 1

            # Sampel augmentasi
            if augment and USE_DATA_AUG:
                for _ in range(AUGMENT_FACTOR - 1):
                    aug_gray = augment_image(gray)
                    aug_feat = compute_hog(aug_gray)
                    X.append(aug_feat)
                    y.append(lbl)

        except Exception:
            miss += 1
            continue

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    print(f"[INFO] Data dir              = {data_dir}")
    print(f"[INFO] Total file asli      = {len(paths)}")
    print(f"[INFO] Sampel asli valid    = {n_original_valid}")
    print(f"[INFO] Sampel final dipakai = {len(X)}")
    print(f"[INFO] Gagal                = {miss}")

    return X, y, class_names

# =========================================================
# BUILD DNN
# =========================================================
def build_dnn_regularized(input_dim, num_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.BatchNormalization(),

        layers.Dense(
            128,
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(REGULARIZATION),
            bias_regularizer=tf.keras.regularizers.l2(REGULARIZATION)
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.6),

        layers.Dense(
            64,
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(REGULARIZATION),
            bias_regularizer=tf.keras.regularizers.l2(REGULARIZATION)
        ),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.5),

        layers.Dense(
            num_classes,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(REGULARIZATION)
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# =========================================================
# MAIN
# =========================================================
def main():
    print("\n" + "=" * 72)
    print("TRAINING SCRIPT - IMAGE-LEVEL AUGMENTATION BEFORE HOG")
    print("=" * 72)

    print("\n📂 Memuat dataset TRAIN...")
    X_train, y_train, class_names = load_dataset(TRAIN_DIR, augment=True)

    print("\n📂 Memuat dataset VALIDATION...")
    X_val, y_val, _ = load_dataset(VAL_DIR, augment=False)

    if len(X_train) == 0 or len(X_val) == 0:
        print("[ERROR] Dataset kosong setelah preprocessing.")
        return

    print("\n📏 Normalisasi fitur dengan StandardScaler...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    print(f" Train shape: {X_train_s.shape}")
    print(f" Val shape  : {X_val_s.shape}")
    print(f" Classes    : {class_names}")

    print("\n🏗️ Membangun model DNN...")
    model = build_dnn_regularized(X_train_s.shape[1], len(class_names))
    model.summary()

    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1
    )

    print("\n🚀 Training model...")
    history = model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    actual_epochs = len(history.history['loss'])

    print("\n📊 Membuat grafik training...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='Train Accuracy', marker='o', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', marker='s', linewidth=2)
    axes[0].set_title("Akurasi Training vs Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='Train Loss', marker='o', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    axes[1].set_title("Loss Training vs Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(OUT_DIR, "training_curve_hog_dnn_imgaug.png")
    plt.savefig(curve_path, dpi=300)
    plt.show()

    print("\n" + "=" * 72)
    print("📊 EVALUASI MODEL")
    print("=" * 72)

    train_loss, _ = model.evaluate(X_train_s, y_train, verbose=0)
    train_preds = model.predict(X_train_s, batch_size=BATCH_SIZE, verbose=0)
    train_pred_labels = np.argmax(train_preds, axis=1)
    train_acc_manual = accuracy_score(y_train, train_pred_labels)

    val_loss, _ = model.evaluate(X_val_s, y_val, verbose=0)
    val_preds = model.predict(X_val_s, batch_size=BATCH_SIZE, verbose=0)
    val_pred_labels = np.argmax(val_preds, axis=1)
    val_acc_manual = accuracy_score(y_val, val_pred_labels)

    acc_gap = (train_acc_manual - val_acc_manual) * 100

    print(f"\nTraining Accuracy   : {train_acc_manual*100:.2f}%")
    print(f"Training Loss       : {train_loss:.6f}")
    print(f"Validation Accuracy : {val_acc_manual*100:.2f}%")
    print(f"Validation Loss     : {val_loss:.6f}")
    print(f"Accuracy Gap        : {acc_gap:+.2f}%")

    print("\n📋 CLASSIFICATION REPORT - VALIDATION SET")
    print(classification_report(y_val, val_pred_labels, target_names=class_names, digits=4))

    print("\n📊 CONFUSION MATRIX - VALIDATION SET")
    cm = confusion_matrix(y_val, val_pred_labels)
    print(cm)

    print("\n💾 MENYIMPAN MODEL DAN ARTEFAK...")
    model_path = os.path.join(OUT_DIR, "hog_dnn_imgaug.keras")
    scaler_path = os.path.join(OUT_DIR, "scaler_imgaug.joblib")
    config_path = os.path.join(OUT_DIR, "hog_params_imgaug.json")

    model.save(model_path)
    dump(scaler, scaler_path)

    config = {
        "IMG_SIZE": IMG_SIZE,
        "REGULARIZATION": REGULARIZATION,
        "DROPOUT_1": 0.6,
        "DROPOUT_2": 0.5,
        "HIDDEN_LAYERS": [128, 64],
        "BATCH_NORMALIZATION": True,
        "DATA_AUGMENTATION": "image_level_before_hog",
        "AUGMENT_FACTOR": AUGMENT_FACTOR,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "ACTUAL_EPOCHS": actual_epochs,
        "TRAIN_ACCURACY": float(train_acc_manual),
        "VAL_ACCURACY": float(val_acc_manual),
        "ACCURACY_GAP": float(acc_gap),
        "TRAINING_MODE": "neutral_no_detector",
        **HOG_PARAMS,
        "classes": list(class_names)
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"✅ Model : {model_path}")
    print(f"✅ Scaler: {scaler_path}")
    print(f"✅ Config: {config_path}")
    print(f"✅ Curve : {curve_path}")

    print("\n" + "=" * 72)
    print("✅ TRAINING SELESAI!")
    print("=" * 72)

if __name__ == "__main__":
    main()