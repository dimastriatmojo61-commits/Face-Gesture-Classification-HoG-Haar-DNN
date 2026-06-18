"""
================================================================================
SCRIPT PERBANDINGAN PERFORMA DETEKSI WAJAH
HOG (Dlib) vs Haar Cascade (OpenCV) - FRAME LEVEL
================================================================================
"""

import pandas as pd
import os
import re

HAAR_CSV = os.path.join("log_haarcascade_wajah_csv", "log_haar_wajah.csv")
HOG_CSV  = os.path.join("log_hog_wajah_csv",          "log_hog_wajah.csv")
REPORT_TXT = "wajah_comparison_report.txt"


def load_summary_aggregate(csv_path: str, name: str):
    if not os.path.exists(csv_path):
        print(f"[ERROR] File CSV {name} tidak ditemukan di: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path, header=None,
                         names=["frame", "gt", "pred", "result", "fps"])
    except Exception as e:
        print(f"[ERROR] Gagal membaca {name} ({csv_path}): {e}")
        return None

    summary_rows = df[df["frame"] == "SUMMARY"]
    if summary_rows.empty:
        print(f"[ERROR] Baris SUMMARY tidak ditemukan di {name} ({csv_path})")
        return None

    def _extract_float(pattern, text):
        m = re.search(pattern, str(text))
        return float(m.group(1)) if m else 0.0

    tp = summary_rows["gt"].astype(float).sum()
    fp = summary_rows["pred"].astype(float).sum()
    fn = tn = 0
    avg_fps_list = []
    mean_ms_list = []

    for _, row in summary_rows.iterrows():
        result_str = str(row["result"])
        fps_str    = str(row["fps"])
        fn  += _extract_float(r"FN=([0-9.]+)",          result_str)
        tn  += _extract_float(r"TN=([0-9.]+)",          result_str)
        val  = _extract_float(r"AvgFPS=([0-9.]+)",      fps_str)
        if val: avg_fps_list.append(val)
        val  = _extract_float(r"MeanFrameMs=([0-9.]+)", fps_str)
        if val: mean_ms_list.append(val)

    total     = tp + fp + fn + tn
    precision = tp / (tp + fp)     if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)     if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    accuracy  = (tp + tn) / total  if total > 0 else 0.0

    avg_fps   = sum(avg_fps_list)  / len(avg_fps_list)  if avg_fps_list  else 0.0
    mean_ms   = sum(mean_ms_list)  / len(mean_ms_list)  if mean_ms_list  else 0.0
    durasi    = len(summary_rows) * 15  # setiap sesi = 15 detik

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": accuracy, "total": int(total),
        "avg_fps": avg_fps, "mean_ms": mean_ms,
        "durasi_detik": durasi,
        "jumlah_run": len(summary_rows),
    }


def main():
    lines = []
    def out(s=""):
        lines.append(s)
        print(s)

    out("\n" + "=" * 120)
    out("PERBANDINGAN PERFORMA DETEKSI WAJAH: HOG (Dlib) vs Haar Cascade (OpenCV)")
    out("=" * 120)

    haar = load_summary_aggregate(HAAR_CSV, "Haar Cascade (OpenCV)")
    hog  = load_summary_aggregate(HOG_CSV,  "HOG (Dlib)")

    if haar is None or hog is None:
        out("[ERROR] Tidak dapat memuat summary dari salah satu/both CSV.")
        return

    out("\n" + "-" * 120)
    out("RINGKASAN METRIK DETEKSI WAJAH (FRAME-LEVEL)")
    out("-" * 120)
    out(f"{'Metrik':<25} {'Haar Cascade':<20} {'HOG (Dlib)':<20} {'Selisih (HOG-Haar)':<20}")
    out("-" * 120)

    rows = [
        ("TP",                    haar["tp"],         hog["tp"],         "+d"),
        ("FP",                    haar["fp"],         hog["fp"],         "+d"),
        ("FN",                    haar["fn"],         hog["fn"],         "+d"),
        ("TN",                    haar["tn"],         hog["tn"],         "+d"),
        ("Precision",             haar["precision"],  hog["precision"],  "+.4f"),
        ("Recall",                haar["recall"],     hog["recall"],     "+.4f"),
        ("F1-Score",              haar["f1"],         hog["f1"],         "+.4f"),
        ("Accuracy",              haar["accuracy"],   hog["accuracy"],   "+.4f"),
        ("Total Frame",           haar["total"],      hog["total"],      "+d"),
        ("FPS Rata-rata",         haar["avg_fps"],    hog["avg_fps"],    "+.2f"),
        ("Waktu/Frame (ms)",      haar["mean_ms"],    hog["mean_ms"],    "+.3f"),
        ("Durasi Pengujian (s)",  haar["durasi_detik"], hog["durasi_detik"], "+d"),
        ("Jumlah Run/Sesi",       haar["jumlah_run"], hog["jumlah_run"], "+d"),
    ]

    for label, h_val, g_val, fmt in rows:
        diff = g_val - h_val
        if "d" in fmt:
            out(f"{label:<25} {h_val:<20} {g_val:<20} {diff:+}")
        else:
            prec = fmt.replace("+", "").replace("f", "").replace(".", "")
            out(f"{label:<25} {h_val:<20{fmt[1:]}} {g_val:<20{fmt[1:]}} {diff:{fmt}}")

    out("-" * 120)

    # Analisis per metrik
    out("\n📊 ANALISIS PERBANDINGAN:")
    out("-" * 120)

    checks = [
        ("Precision",   haar["precision"], hog["precision"]),
        ("Recall",      haar["recall"],    hog["recall"]),
        ("F1-Score",    haar["f1"],        hog["f1"]),
        ("Accuracy",    haar["accuracy"],  hog["accuracy"]),
        ("FPS",         haar["avg_fps"],   hog["avg_fps"]),
    ]
    for label, h_val, g_val in checks:
        if abs(g_val - h_val) < 1e-4:
            out(f"  {label:<15} → SETARA")
        elif g_val > h_val:
            out(f"  {label:<15} → HOG (Dlib) LEBIH BAIK    [{g_val:.4f} vs {h_val:.4f}]")
        else:
            out(f"  {label:<15} → Haar Cascade LEBIH BAIK  [{h_val:.4f} vs {g_val:.4f}]")

    # Kecepatan
    if abs(hog["mean_ms"] - haar["mean_ms"]) < 1.0:
        out(f"  {'Kecepatan':<15} → SETARA")
    elif hog["mean_ms"] < haar["mean_ms"]:
        out(f"  {'Kecepatan':<15} → HOG (Dlib) LEBIH CEPAT   [{hog['mean_ms']:.3f} ms vs {haar['mean_ms']:.3f} ms]")
    else:
        out(f"  {'Kecepatan':<15} → Haar Cascade LEBIH CEPAT [{haar['mean_ms']:.3f} ms vs {hog['mean_ms']:.3f} ms]")

    # Skor akhir
    hog_score  = sum(1 for _, h, g in checks if g > h + 1e-4)
    haar_score = sum(1 for _, h, g in checks if h > g + 1e-4)
    if hog["mean_ms"]  < haar["mean_ms"]  - 1.0: hog_score  += 1
    if haar["mean_ms"] < hog["mean_ms"]   - 1.0: haar_score += 1

    out("\n" + "-" * 120)
    out(f"SKOR AKHIR  →  HOG (Dlib): {hog_score} poin  |  Haar Cascade: {haar_score} poin")
    if hog_score > haar_score:
        out(f"[HASIL] HOG (Dlib) menunjukkan performa deteksi wajah LEBIH BAIK secara keseluruhan.")
    elif haar_score > hog_score:
        out(f"[HASIL] Haar Cascade menunjukkan performa deteksi wajah LEBIH BAIK secara keseluruhan.")
    else:
        out("[HASIL] Kedua detektor menunjukkan performa YANG SETARA.")

    out("\nCatatan:")
    out("  TP  : Ada wajah di frame, sistem memberi kotak di wajah tersebut.")
    out("  FP  : Sistem memberi kotak pada objek non-wajah / frame kosong.")
    out("  FN  : Wajah ada di frame, tapi tidak muncul bounding box sama sekali.")
    out("  TN  : Frame kosong tanpa manusia, dan sistem tidak memberi kotak.")
    out("  FPS : Rata-rata dari seluruh run/sesi yang telah dilakukan.")
    out("=" * 120)

    try:
        with open(REPORT_TXT, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"\n[✓] Laporan disimpan ke: {REPORT_TXT}")
    except Exception as e:
        print(f"\n[WARNING] Gagal menyimpan laporan: {e}")


if __name__ == "__main__":
    main()