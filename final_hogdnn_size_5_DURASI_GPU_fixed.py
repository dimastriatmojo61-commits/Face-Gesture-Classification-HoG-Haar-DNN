"""
================================================================================
SISTEM DETEKSI DAN KLASIFIKASI GESTURE WAJAH REAL-TIME
Menggunakan: HOG (Histogram of Oriented Gradients) + DNN  [GPU-Accelerated]
================================================================================
Kontrol:
- Tekan S untuk mulai sesi (SESSION_DURATION detik)
- Tekan 0 = mengantuk  |  1 = senyum  (set ground-truth)
- Tekan C untuk clear ground-truth
- Tekan ESC untuk keluar dan mencetak hasil
================================================================================
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import time
import json
import csv
import glob
import numpy as np
from skimage.feature import hog
from joblib import load
from collections import deque, Counter
import dlib
import sys
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==============================
# 0) KONFIGURASI GPU
# ==============================
def setup_gpu():
    info = {"available": False, "device_name": "CPU", "device": "/CPU:0"}
    try:
        gpus = tf.config.list_physical_devices("GPU")
        print(f"[INFO] TensorFlow version : {tf.__version__}")
        print(f"[INFO] Built with CUDA    : {tf.test.is_built_with_cuda()}")
        print(f"[INFO] GPU terdeteksi TF : {len(gpus)}")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical = tf.config.list_logical_devices("GPU")
            info["available"] = True
            info["device_name"] = logical[0].name if logical else gpus[0].name
            info["device"] = "/GPU:0"
            print(f"[✓] GPU aktif: {info['device_name']}")
        else:
            print("[⚠] GPU tidak terdeteksi TensorFlow, fallback ke CPU")
    except Exception as e:
        print(f"[⚠] Gagal inisialisasi GPU TensorFlow: {e}")
        print("[INFO] Program akan lanjut memakai CPU")
    return info

GPU_INFO = setup_gpu()

# ==============================
# 1) KONFIGURASI & PATH
# ==============================
MODEL_CANDIDATES = [
    "models_regularized_imgaug/hog_dnn_imgaug.keras",
]
MODEL_PATH     = "models_regularized_imgaug/hog_dnn_imgaug.keras"
SCALER_PATH    = "models_regularized_imgaug/scaler_imgaug.joblib"
PARAMS_PATH    = "models_regularized_imgaug/hog_params_imgaug.json"
THRESHOLD_PATH = "thresholds.json"

LOG_DIR            = "log_hog_dnn_csv"
LOG_FILE_BASE_NAME = "log_hog_dnn"
LOG_FILE_BASE      = os.path.join(LOG_DIR, LOG_FILE_BASE_NAME)

FRAME_SIZE        = (640, 480)
SMOOTH_WINDOW     = 5
DEFAULT_THRESHOLD = 0.8
DETECT_EVERY_N    = 3
SHOW_FPS          = True
SESSION_DURATION  = 5

DETECTOR_NAME = "HOG"
FEATURE_NAME  = "HoG (Histogram of Oriented Gradients)"

os.makedirs(LOG_DIR, exist_ok=True)

def find_model_path():
    for path in MODEL_CANDIDATES:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "saved_model.pb")):
            return path
        if os.path.isfile(path):
            return path
    keras_like = glob.glob("models_regularized_imgaug/*.keras") + glob.glob("models_regularized_imgaug/*.h5")
    if keras_like:
        return keras_like[0]
    return None

def next_run_number(file_base: str) -> int:
    i = 1
    while os.path.exists(f"{file_base}_{i}.csv"):
        i += 1
    return i

RUN_NO       = next_run_number(LOG_FILE_BASE)
LOG_FILE_AGG = f"{LOG_FILE_BASE}.csv"
LOG_FILE     = f"{LOG_FILE_BASE}_{RUN_NO}.csv"
REPORT_FILE  = os.path.join(LOG_DIR, f"report_hog_dnn_{RUN_NO}.txt")

LOG_HEADER = [
    "Timestamp", "Frame_Index", "Det_Time_ms", "Prep_Time_ms",
    "Inf_Time_ms", "Total_Frame_ms", "FPS", "Pred_Label",
    "Confidence", "GT_Label", "Is_Correct", "Face_Detected", "Detector"
]

def ensure_csv_header(path, header):
    if not os.path.exists(path):
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)

ensure_csv_header(LOG_FILE_AGG, LOG_HEADER)
ensure_csv_header(LOG_FILE,     LOG_HEADER)

# ==============================
# 2) LOAD MODEL & ARTIFACT
# ==============================
print("[INFO] Memuat model dan artifact...")
model_path = find_model_path()
if model_path is None:
    print("[ERROR] Model tidak ditemukan. Cek folder models_regularized_imgaug/")
    sys.exit(1)

try:
    scaler = load(SCALER_PATH)
    model  = load_model(model_path)
    print(f"[✓] Model DNN berhasil dimuat dari {model_path}")
except Exception as e:
    print(f"[ERROR] Gagal memuat model/artifact: {e}")
    sys.exit(1)

try:
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        P = json.load(f)
except Exception as e:
    print(f"[ERROR] Gagal membaca file parameter HOG: {e}")
    sys.exit(1)

IMG_SIZE   = int(P["IMG_SIZE"])
HOG_PARAMS = {k: P[k] for k in
              ["orientations", "pixels_per_cell", "cells_per_block",
               "block_norm", "transform_sqrt"]}

ALL_CLASSES     = list(P["classes"])
DESIRED_CLASSES = ["mengantuk", "senyum"]
CLASS_NAMES     = [cls for cls in DESIRED_CLASSES if cls in ALL_CLASSES]
CLASS_IDX_MAP   = [ALL_CLASSES.index(cls) for cls in CLASS_NAMES]

if len(CLASS_NAMES) != len(DESIRED_CLASSES):
    print(f"[ERROR] Kelas target {DESIRED_CLASSES} tidak lengkap di artifact: {ALL_CLASSES}")
    sys.exit(1)

num_model_classes = model.output_shape[-1]
if num_model_classes <= max(CLASS_IDX_MAP):
    print(f"[ERROR] Output model hanya {num_model_classes} kelas, "
          f"namun membutuhkan indeks hingga {max(CLASS_IDX_MAP)} untuk {CLASS_NAMES}")
    sys.exit(1)

print(f"[INFO] Kelas gesture: {CLASS_NAMES}")

conf_matrix      = {t: {p: 0 for p in CLASS_NAMES} for t in CLASS_NAMES}
current_gt_label = None

KEY_TO_LABEL = {}
for i, cls_name in enumerate(CLASS_NAMES):
    if i < 10:
        KEY_TO_LABEL[ord(str(i))] = cls_name

try:
    with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
        all_thresholds = json.load(f)
    THRESHOLDS = {cls: all_thresholds.get(cls, DEFAULT_THRESHOLD) for cls in CLASS_NAMES}
    print(f"[✓] Threshold loaded: {THRESHOLDS}")
except Exception:
    THRESHOLDS = {cls: DEFAULT_THRESHOLD for cls in CLASS_NAMES}
    print(f"[⚠] Threshold file tidak ditemukan. Menggunakan default: {DEFAULT_THRESHOLD}")

# ==============================
# 3) HOG DLIB & HELPER
# ==============================
print(f"[INFO] Inisialisasi detektor: {FEATURE_NAME}")
hog_face_detector = dlib.get_frontal_face_detector()
print("[✓] Detektor HOG (Dlib) siap")

def detect_faces(image_bgr):
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    rects = hog_face_detector(gray, 0)
    boxes = []
    h, w  = gray.shape[:2]
    for r in rects:
        x1, y1, x2, y2 = r.left(), r.top(), r.right(), r.bottom()
        pad = int(0.10 * max(x2 - x1, y2 - y1))
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
        boxes.append((x1, y1, x2, y2))
    return boxes

def largest_box(boxes):
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

def compute_hog_features(gray_roi):
    return hog(gray_roi, **HOG_PARAMS)

def majority_vote(label_history):
    if not label_history:
        return None
    return Counter(label_history).most_common(1)[0][0]

def safe_div(a, b):
    return a / b if b != 0 else 0.0

# ==============================
# 4) LOOP REAL-TIME
# ==============================
print("[INFO] Memulai capture kamera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_SIZE[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])

if not cap.isOpened():
    print("[ERROR] Gagal membuka kamera!")
    sys.exit(1)

cv2.namedWindow(f"[{DETECTOR_NAME}-DNN] Gesture Classification", cv2.WINDOW_NORMAL)

print("[✓] Kamera siap")
print(f"[INFO] Tekan S untuk mulai sesi ({SESSION_DURATION} detik)")
print(f"[INFO] Tekan tombol angka untuk set ground-truth gesture:")
for i, cls in enumerate(CLASS_NAMES):
    print(f"       {i} = {cls}")
print("[INFO] Tekan C untuk clear ground-truth")
print(f"[INFO] Device inferensi: {GPU_INFO['device_name']}")
print("[INFO] Tekan ESC untuk keluar\n")

session_active   = False
session_start    = None
label_hist       = deque(maxlen=SMOOTH_WINDOW)
last_boxes       = []
frame_count      = 0
det_calls        = 0
inf_calls        = 0
t_all0           = time.perf_counter()
sum_det_ms       = 0.0
sum_prep_ms      = 0.0
sum_inf_ms       = 0.0
sum_total_ms     = 0.0
predictions      = []

while True:
    t_start_frame = time.perf_counter()
    ok, frame = cap.read()
    if not ok:
        break

    frame       = cv2.resize(frame, FRAME_SIZE)
    frame_count += 1

    det_time          = 0.0
    prep_time         = 0.0
    inf_time          = 0.0
    stable_label      = None
    pred_label        = None
    confidence        = 0.0
    is_correct        = ""
    used_box          = None
    pred_face_present = len(last_boxes) > 0

    if (frame_count % DETECT_EVERY_N == 0) or (not last_boxes):
        t_det0      = time.perf_counter()
        last_boxes  = detect_faces(frame)
        t_det1      = time.perf_counter()
        det_time    = (t_det1 - t_det0) * 1000.0
        sum_det_ms += det_time
        det_calls  += 1

    pred_face_present = len(last_boxes) > 0
    used_box          = largest_box(last_boxes)

    if used_box is not None:
        x1, y1, x2, y2 = used_box
        roi = frame[y1:y2, x1:x2]

        if roi.size > 0:
            t_prep0     = time.perf_counter()
            gray_roi    = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_roi    = cv2.resize(gray_roi, (IMG_SIZE, IMG_SIZE))
            feat        = compute_hog_features(gray_roi).reshape(1, -1).astype(np.float32)
            feat        = scaler.transform(feat)
            t_prep1     = time.perf_counter()
            prep_time   = (t_prep1 - t_prep0) * 1000.0
            sum_prep_ms += prep_time

            t_inf0 = time.perf_counter()
            with tf.device(GPU_INFO["device"]):
                prob_full = model.predict(feat, verbose=0)[0]
            t_inf1    = time.perf_counter()
            inf_time  = (t_inf1 - t_inf0) * 1000.0
            sum_inf_ms += inf_time
            inf_calls += 1

            prob         = prob_full[CLASS_IDX_MAP]
            idx          = int(np.argmax(prob))
            pred_label   = CLASS_NAMES[idx]
            confidence   = float(prob[idx])

            label_hist.append(pred_label)
            voted_label  = majority_vote(label_hist)
            stable_label = voted_label if voted_label is not None else pred_label

            if current_gt_label is not None and current_gt_label in conf_matrix:
                conf_matrix[current_gt_label][stable_label] += 1
                is_correct = int(current_gt_label == stable_label)
                predictions.append({
                    "timestamp":  datetime.now().isoformat(),
                    "pred_label": stable_label,
                    "gt_label":   current_gt_label,
                    "confidence": confidence,
                    "is_correct": bool(is_correct)
                })

            threshold = THRESHOLDS.get(stable_label, DEFAULT_THRESHOLD)
            color     = (20, 220, 20) if confidence >= threshold else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{stable_label} {confidence*100:.1f}%",
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    t_end_frame   = time.perf_counter()
    total_ms      = (t_end_frame - t_start_frame) * 1000.0
    fps           = 1000.0 / total_ms if total_ms > 0 else 0.0
    sum_total_ms += total_ms

    row = [
        datetime.now().strftime("%H:%M:%S.%f")[:-3],
        frame_count,
        f"{det_time:.3f}",
        f"{prep_time:.3f}",
        f"{inf_time:.3f}",
        f"{total_ms:.3f}",
        f"{fps:.3f}",
        stable_label,
        f"{confidence:.4f}",
        current_gt_label if current_gt_label is not None else None,
        is_correct,
        int(pred_face_present),
        DETECTOR_NAME
    ]
    for path in [LOG_FILE_AGG, LOG_FILE]:
        with open(path, mode="a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

    if SHOW_FPS:
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Detector: {DETECTOR_NAME}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
        cv2.putText(frame, f"Device: {GPU_INFO['device_name']}", (10, FRAME_SIZE[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 100) if GPU_INFO["available"] else (0, 165, 255), 1)

    if current_gt_label is not None:
        cv2.putText(frame, f"GT: {current_gt_label}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "GT: tekan 0=mengantuk / 1=senyum", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

    if session_active:
        elapsed_s   = time.perf_counter() - session_start
        remaining_s = SESSION_DURATION - elapsed_s
        if remaining_s <= 0:
            session_active = False
            print("[INFO] Sesi selesai! Tekan S untuk sesi baru atau ESC untuk keluar.")
        else:
            cv2.putText(frame, f"SESI AKTIF | Sisa: {remaining_s:.1f}s", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "Tekan S untuk mulai sesi (klik window dulu!)", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

    cv2.imshow(f"[{DETECTOR_NAME}-DNN] Gesture Classification", frame)
    key = cv2.waitKey(30) & 0xFF

    if key == 27:
        print("[INFO] ESC ditekan. Menutup sistem...")
        break

    if key in (ord("s"), ord("S")):
        if not session_active:
            session_active = True
            session_start  = time.perf_counter()
            label_hist.clear()
            print(f"[INFO] Sesi dimulai! Durasi: {SESSION_DURATION} detik")
            print("[INFO] Tekan 0=mengantuk atau 1=senyum untuk set GT")
        else:
            print("[INFO] Sesi sudah aktif.")

    if session_active and (time.perf_counter() - session_start) <= SESSION_DURATION:
        if key in KEY_TO_LABEL:
            current_gt_label = KEY_TO_LABEL[key]
            label_hist.clear()
            print(f"[INFO] Ground-truth di-set ke: {current_gt_label}")
        if key in (ord("c"), ord("C")):
            current_gt_label = None
            label_hist.clear()
            print("[INFO] Ground-truth di-clear")
    elif not session_active:
        if key in (ord("c"), ord("C")):
            current_gt_label = None
            label_hist.clear()
            print("[INFO] Ground-truth di-clear")

cap.release()
cv2.destroyAllWindows()

# ==============================
# 5) RINGKASAN & LAPORAN
# ==============================
print("=" * 80)
print("RINGKASAN PERFORMA SISTEM DETEKSI & KLASIFIKASI GESTURE WAJAH")
print(f"Detektor: {DETECTOR_NAME} ({FEATURE_NAME})")
print(f"Device  : {GPU_INFO['device_name']}")
print("=" * 80)

total_time = time.perf_counter() - t_all0
avg_fps    = frame_count / total_time if total_time > 0 else 0.0
mean_det   = sum_det_ms   / det_calls  if det_calls  > 0 else 0.0
mean_prep  = sum_prep_ms  / inf_calls  if inf_calls  > 0 else 0.0
mean_inf   = sum_inf_ms   / inf_calls  if inf_calls  > 0 else 0.0
mean_total = sum_total_ms / frame_count if frame_count > 0 else 0.0

print("STATISTIK UMUM")
print(f"  Total frame diproses : {frame_count} frames")
print(f"  Durasi pengujian     : {total_time:.2f} detik")
print(f"  FPS rata-rata        : {avg_fps:.2f} FPS")
print("WAKTU PEMROSESAN")
print(f"  Deteksi wajah per call  : {mean_det:.3f} ms")
print(f"  Preprocessing per face  : {mean_prep:.3f} ms")
print(f"  Inferensi DNN per face  : {mean_inf:.3f} ms")
print(f"  Total pipeline per frame: {mean_total:.3f} ms")

macro_prec = macro_rec = macro_f1 = 0.0
micro_tp = micro_fp = micro_fn = 0
micro_prec = micro_rec = micro_f1 = 0.0
valid_classes = 0
gesture_rows  = []

print("METRIK AKURASI PER KELAS (berdasarkan Ground-Truth)")
print("-" * 80)
print(f"{'Kelas':<12} {'TP':>4} {'FP':>4} {'FN':>4} {'Support':>7}   {'Precision':>10} {'Recall':>10} {'F1-Score':>10}")
print("-" * 80)

for cls in CLASS_NAMES:
    tp      = conf_matrix[cls][cls]
    fp      = sum(conf_matrix[t][cls] for t in CLASS_NAMES if t != cls)
    fn      = sum(conf_matrix[cls][p] for p in CLASS_NAMES if p != cls)
    support = sum(conf_matrix[cls].values())
    if support == 0:
        print(f"{cls:<12} {tp:>4} {fp:>4} {fn:>4} {0:>7}   {''  :>10} {''  :>10} {''  :>10}")
        gesture_rows.append((cls, tp, fp, fn, support, None, None, None))
        continue
    precision = safe_div(tp, tp + fp)
    recall    = safe_div(tp, tp + fn)
    f1        = safe_div(2 * precision * recall, precision + recall)
    macro_prec  += precision
    macro_rec   += recall
    macro_f1    += f1
    micro_tp    += tp
    micro_fp    += fp
    micro_fn    += fn
    valid_classes += 1
    print(f"{cls:<12} {tp:>4} {fp:>4} {fn:>4} {support:>7}   {precision:>10.4f} {recall:>10.4f} {f1:>10.4f}")
    gesture_rows.append((cls, tp, fp, fn, support, precision, recall, f1))

print("-" * 80)
if valid_classes > 0:
    macro_prec /= valid_classes
    macro_rec  /= valid_classes
    macro_f1   /= valid_classes
    print(f"{'Macro Avg':<12} {''  :>4} {''  :>4} {''  :>4} {''  :>7}   {macro_prec:>10.4f} {macro_rec:>10.4f} {macro_f1:>10.4f}")
    micro_prec = safe_div(micro_tp, micro_tp + micro_fp)
    micro_rec  = safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1   = safe_div(2 * micro_prec * micro_rec, micro_prec + micro_rec)
    if micro_tp + micro_fp + micro_fn > 0:
        print(f"{'Micro Avg':<12} {''  :>4} {''  :>4} {''  :>4} {''  :>7}   {micro_prec:>10.4f} {micro_rec:>10.4f} {micro_f1:>10.4f}")

total_samples    = sum(sum(conf_matrix[t].values()) for t in CLASS_NAMES)
overall_accuracy = safe_div(micro_tp, total_samples) if total_samples > 0 else 0
print(f"{'Overall Acc':<12} {''  :>4} {''  :>4} {''  :>4} {total_samples:>7}   {overall_accuracy:>10.4f} {''  :>10} {''  :>10}")
print(f"{'Accuracy %':<12} {''  :>4} {''  :>4} {''  :>4} {total_samples:>7}   {overall_accuracy*100:>10.2f}% {''  :>10} {''  :>10}")

print("=" * 80)
print("RINGKASAN AKURASI KLASIFIKASI GESTURE")
print("-" * 80)
if total_samples > 0:
    wrong = total_samples - micro_tp
    print(f"  Overall Accuracy gesture : {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"  Total Sampel Evaluasi    : {total_samples} samples")
    print(f"  Prediksi Benar           : {micro_tp} samples")
    print(f"  Prediksi Salah           : {wrong} samples")
    print(f"  Total False Positive FP  : {micro_fp} samples")
    print(f"  Total False Negative FN  : {micro_fn} samples")
else:
    print("  Tidak ada data ground-truth untuk menghitung akurasi klasifikasi gesture.")

print("=" * 80)
print("CONFUSION MATRIX GESTURE")
print("-" * 80)
print("Format: Baris = Ground-Truth, Kolom = Prediksi")
gt_pred_label = "GT \\ Pred"
hdr = f"{gt_pred_label:<14}" + "".join(f"{cls:>12}" for cls in CLASS_NAMES)
print(hdr)
print("-" * len(hdr))
for true_cls in CLASS_NAMES:
    row_str = f"{true_cls:<14}"
    for pred_cls in CLASS_NAMES:
        row_str += f"{conf_matrix[true_cls][pred_cls]:>12}"
    print(row_str)

print("=" * 80)
print(f"[✓] Log run ini disimpan di  : {LOG_FILE}")
print(f"[✓] Log agregat semua run    : {LOG_FILE_AGG}")

# ==============================
# 6) SIMPAN REPORT .TXT
# ==============================
try:
    lines = [
        "=" * 80,
        "LAPORAN HASIL SISTEM HOG + DNN (GPU-Accelerated)",
        "=" * 80,
        f"Tanggal/Waktu : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Run Ke        : {RUN_NO}",
        f"Log CSV Run   : {LOG_FILE}",
        f"Log CSV Agg   : {LOG_FILE_AGG}",
        f"Device        : {GPU_INFO['device_name']}",
        "",
        "KONFIGURASI",
        "-" * 80,
        f"  Detector    : {DETECTOR_NAME} ({FEATURE_NAME})",
        f"  Frame Size  : {FRAME_SIZE}",
        f"  Det Every N : {DETECT_EVERY_N}",
        f"  Classes     : {CLASS_NAMES}",
        f"  Thresholds  : {THRESHOLDS}",
        "",
        "STATISTIK UMUM",
        "-" * 80,
        f"  Total frame : {frame_count}",
        f"  Durasi (s)  : {total_time:.2f}",
        f"  Avg FPS     : {avg_fps:.2f}",
        "",
        "WAKTU PEMROSESAN",
        "-" * 80,
        f"  Deteksi wajah per call  : {mean_det:.3f} ms",
        f"  Preprocessing per face  : {mean_prep:.3f} ms",
        f"  Inferensi DNN per face  : {mean_inf:.3f} ms",
        f"  Total pipeline per frame: {mean_total:.3f} ms",
        "",
        "METRIK KLASIFIKASI GESTURE PER KELAS",
        "-" * 80,
        f"{'Kelas':<12} {'TP':>4} {'FP':>4} {'FN':>4} {'Support':>7}   {'Precision':>10} {'Recall':>10} {'F1':>10}",
    ]
    for cls, tp, fp, fn, support, prec, rec, f1v in gesture_rows:
        if prec is None:
            lines.append(f"{cls:<12} {tp:>4} {fp:>4} {fn:>4} {support:>7}   {''  :>10} {''  :>10} {''  :>10}")
        else:
            lines.append(f"{cls:<12} {tp:>4} {fp:>4} {fn:>4} {support:>7}   {prec:>10.4f} {rec:>10.4f} {f1v:>10.4f}")
    lines += [
        "",
        f"  Macro Precision : {macro_prec:.4f}",
        f"  Macro Recall    : {macro_rec:.4f}",
        f"  Macro F1        : {macro_f1:.4f}",
        f"  Micro Precision : {micro_prec:.4f}",
        f"  Micro Recall    : {micro_rec:.4f}",
        f"  Micro F1        : {micro_f1:.4f}",
        f"  Overall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%) | {total_samples} samples",
        "",
        "CONFUSION MATRIX KLASIFIKASI GESTURE (GT x Pred)",
        "-" * 80,
        hdr,
        "-" * len(hdr),
    ]
    for true_cls in CLASS_NAMES:
        row_str = f"{true_cls:<14}"
        for pred_cls in CLASS_NAMES:
            row_str += f"{conf_matrix[true_cls][pred_cls]:>12}"
        lines.append(row_str)
    lines.append("=" * 80)

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[✓] Report .txt disimpan di  : {REPORT_FILE}")
except Exception as e:
    print(f"[⚠] Gagal menyimpan report txt: {e}")

print("=" * 80)
