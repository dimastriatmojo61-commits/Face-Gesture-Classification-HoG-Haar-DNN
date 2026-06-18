"""
Realtime HOG detector (dlib) - deteksi wajah (ya/tidak).
Kontrol:
- Tekan 'S' untuk mulai sesi (durasi SESSION_DURATION detik)
- Tekan 'f' = ada wajah, 'n' = tidak ada wajah (GT persisten)
- Tekan 'C' untuk clear GT
- Tekan ESC untuk keluar

CATATAN GPU:
File ini menggunakan dlib HOG Detector murni (CPU).
GPU RTX 2050 tidak digunakan pada file deteksi wajah saja.
GPU digunakan pada file *_dnn_* untuk inferensi model klasifikasi.
"""

import csv
import os
import time
import cv2

try:
    import dlib
except Exception:
    dlib = None

SESSION_DURATION = 5
FRAME_SIZE = (640, 480)
WINDOW_NAME = "HOG-Realtime"

_hog_detector = None


def get_detector():
    global _hog_detector
    if _hog_detector is None:
        _hog_detector = dlib.get_frontal_face_detector()
    return _hog_detector


def detect_face_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return get_detector()(gray, 0)


def safe_div(a, b):
    return a / b if b else 0.0


def next_run_number(log_dir, base):
    i = 1
    while os.path.exists(os.path.join(log_dir, f"{base}_{i}.csv")):
        i += 1
    return i


def main():
    if dlib is None:
        print("[ERROR] dlib tidak ditemukan. Install: pip install dlib")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_SIZE[1])
    if not cap.isOpened():
        print("[ERROR] Gagal membuka kamera")
        return

    print("[INFO] Deteksi wajah real-time — HOG (dlib CPU)")
    print(f"[INFO] Resolusi: {FRAME_SIZE[0]}x{FRAME_SIZE[1]}")
    print("[INFO] Tekan S=mulai sesi | f=ada wajah | n=tidak ada wajah | C=clear GT | ESC=keluar\n")

    frame_idx = TP = FP = FN = TN = 0
    results = []
    session_active = False
    session_start = None
    gt = None
    sum_total_ms = 0.0
    frame_count = 0
    t_all0 = time.perf_counter()

    while True:
        t_start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, FRAME_SIZE)

        rects = detect_face_hog(frame)
        detected = len(rects) > 0
        for rect in rects:
            x1 = max(0, rect.left())
            y1 = max(0, rect.top())
            x2 = min(FRAME_SIZE[0] - 1, rect.right())
            y2 = min(FRAME_SIZE[1] - 1, rect.bottom())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        total_ms = (time.perf_counter() - t_start) * 1000
        fps = 1000.0 / total_ms if total_ms else 0.0
        sum_total_ms += total_ms
        frame_count += 1

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, FRAME_SIZE[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Mode: CPU (HoG Dlib)", (10, FRAME_SIZE[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if session_active:
            elapsed = time.perf_counter() - session_start
            remaining = SESSION_DURATION - elapsed
            if remaining <= 0:
                session_active = False
                gt = None
                print(f"[INFO] Sesi selesai! Frame tercatat: {frame_idx}")
                print("[INFO] Tekan S untuk sesi baru atau ESC keluar.")
            else:
                lbl = "ADA WAJAH" if gt else "TIDAK ADA WAJAH"
                cv2.putText(frame, f"SESI AKTIF | Sisa: {remaining:.1f}s", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
                cv2.putText(frame, f"GT: {lbl if gt is not None else 'Tekan f=wajah / n=tidak'}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 100) if gt is not None else (0, 165, 255), 2)
                if gt is not None:
                    pred = detected
                    if gt and pred:
                        TP += 1
                        result = "TP"
                    elif (not gt) and pred:
                        FP += 1
                        result = "FP"
                    elif gt and (not pred):
                        FN += 1
                        result = "FN"
                    else:
                        TN += 1
                        result = "TN"
                    frame_idx += 1
                    results.append({"frame": frame_idx, "gt": int(gt), "pred": int(pred), "result": result, "fps": f"{fps:.3f}"})
                else:
                    cv2.putText(frame, "Tekan f atau n untuk set GT", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)
        else:
            cv2.putText(frame, "Tekan S untuk mulai sesi", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 0), 2)

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break
        if key in (ord("s"), ord("S")):
            if not session_active:
                session_active = True
                session_start = time.perf_counter()
                gt = None
                print(f"[INFO] Sesi dimulai ({SESSION_DURATION}s). Tekan f/n untuk GT.")
            else:
                print("[INFO] Sesi sudah aktif.")
        if session_active:
            if key == ord("f"):
                gt = True
                print("[INFO] GT = ADA WAJAH")
            elif key == ord("n"):
                gt = False
                print("[INFO] GT = TIDAK ADA WAJAH")
            elif key in (ord("c"), ord("C")):
                gt = None
                print("[INFO] GT dibersihkan")

    cap.release()
    cv2.destroyAllWindows()

    total_time = time.perf_counter() - t_all0
    avg_fps = frame_count / total_time if total_time else 0
    mean_ms = sum_total_ms / frame_count if frame_count else 0
    total_s = TP + FP + FN + TN
    precision = safe_div(TP, TP + FP)
    recall = safe_div(TP, TP + FN)
    f1 = safe_div(2 * precision * recall, precision + recall)
    accuracy = safe_div(TP + TN, total_s)

    print("\n" + "=" * 80)
    print("CONFUSION MATRIX — HOG (dlib) | Mode: CPU")
    print(f" TP={TP} FP={FP} FN={FN} TN={TN}")
    print(f" Precision : {precision:.4f}")
    print(f" Recall    : {recall:.4f}")
    print(f" F1-Score  : {f1:.4f}")
    print(f" Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f" Total frame: {frame_count} | Avg FPS: {avg_fps:.2f} | Avg ms/frame: {mean_ms:.3f}")
    print("=" * 80)

    log_dir = "log_hog_wajah_csv"
    log_base = "log_hog_wajah"
    os.makedirs(log_dir, exist_ok=True)
    run_no = next_run_number(log_dir, log_base)
    log_run = os.path.join(log_dir, f"{log_base}_{run_no}.csv")
    log_agg = os.path.join(log_dir, f"{log_base}.csv")
    fieldnames = ["frame", "gt", "pred", "result", "fps"]

    def write_log(path):
        new_file = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if new_file:
                writer.writeheader()
            writer.writerows(results)
            writer.writerow({})
            writer.writerow({
                "frame": "SUMMARY",
                "gt": TP,
                "pred": FP,
                "result": f"FN={FN},TN={TN},P={precision:.4f},R={recall:.4f},F1={f1:.4f},Acc={accuracy:.4f}",
                "fps": f"AvgFPS={avg_fps:.2f},ms={mean_ms:.3f}"
            })

    write_log(log_run)
    write_log(log_agg)
    print(f"[✓] Log: {log_run} | Agregat: {log_agg}")


if __name__ == "__main__":
    main()
