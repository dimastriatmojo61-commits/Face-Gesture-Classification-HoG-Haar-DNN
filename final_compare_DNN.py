"""
================================================================================
SCRIPT PERBANDINGAN PERFORMA DETEKTOR WAJAH HOG+DNN vs HAAR CASCADE+DNN
================================================================================
"""

import csv
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

HOG_LOG_FILE   = os.path.join("log_hog_dnn_csv",          "log_hog_dnn.csv")
HAAR_LOG_FILE  = os.path.join("log_haarcascade_dnn_csv",   "log_haar_dnn.csv")
OUTPUT_REPORT  = "detector_comparison_report.txt"

SESSION_DURATION = 15  # detik, sesuai sistem utama


def load_csv_results(file_path):
    if not os.path.exists(file_path):
        print(f"[ERROR] File {file_path} tidak ditemukan!")
        return None
    try:
        try:
            df = pd.read_csv(file_path, on_bad_lines='skip')
        except TypeError:
            df = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines=False)

        # Normalisasi nama kolom
        if 'Pred_Label'      in df.columns and 'Label'        not in df.columns:
            df['Label'] = df['Pred_Label']
        if 'Confidence'      in df.columns and 'Conf'         not in df.columns:
            df['Conf']  = df['Confidence']
        if 'Total_Frame_ms'  in df.columns and 'Total_ms'     not in df.columns:
            df['Total_ms'] = df['Total_Frame_ms']

        return df
    except Exception as e:
        print(f"[ERROR] Gagal membaca {file_path}: {e}")
        return None


def calculate_statistics(df):
    if df is None or len(df) == 0:
        return None

    numeric_cols = ['Det_Time_ms', 'Inf_Time_ms', 'Total_ms', 'FPS', 'Conf']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'Label' not in df.columns:
        df['Label'] = 'unknown'

    df_valid = df[(df['Label'] != 'None') & df['Label'].notna()].copy()

    def s_mean(s): return s.dropna().mean() if len(s.dropna()) > 0 else 0.0
    def s_min(s):  return s.dropna().min()  if len(s.dropna()) > 0 else 0.0
    def s_max(s):  return s.dropna().max()  if len(s.dropna()) > 0 else 0.0
    def s_std(s):  return s.dropna().std()  if len(s.dropna()) > 1 else 0.0

    # Hitung durasi dari jumlah baris SUMMARY (tiap sesi = 15 detik)
    jumlah_run = 0
    durasi_detik = 0
    if 'Frame_Index' in df.columns:
        summary_rows = df[df['Frame_Index'].astype(str).str.upper() == 'SUMMARY']
        jumlah_run   = len(summary_rows)
    # fallback: estimasi dari total frame & FPS
    if jumlah_run == 0:
        jumlah_run = max(1, round(len(df) / (s_mean(df['FPS']) * SESSION_DURATION))
                         if s_mean(df['FPS']) > 0 else 1)
    durasi_detik = jumlah_run * SESSION_DURATION

    # Akurasi
    accuracy = 0.0
    accuracy_samples = 0
    if 'GT_Label' in df.columns and 'Is_Correct' in df.columns:
        df_acc = df[(df['GT_Label'] != 'None') & df['GT_Label'].notna()].copy()
        if len(df_acc) > 0:
            df_acc['Is_Correct_Num'] = pd.to_numeric(df_acc['Is_Correct'], errors='coerce')
            accuracy_samples = int(df_acc['Is_Correct_Num'].notna().sum())
            if accuracy_samples > 0:
                accuracy = float(df_acc['Is_Correct_Num'].sum()) / accuracy_samples * 100.0

    return {
        'total_frames':     len(df),
        'valid_frames':     len(df_valid),
        'none_frames':      len(df) - len(df_valid),
        'det_time_mean':    s_mean(df['Det_Time_ms'])   if 'Det_Time_ms' in df.columns else 0.0,
        'det_time_std':     s_std(df['Det_Time_ms'])    if 'Det_Time_ms' in df.columns else 0.0,
        'inf_time_mean':    s_mean(df['Inf_Time_ms'])   if 'Inf_Time_ms' in df.columns else 0.0,
        'inf_time_std':     s_std(df['Inf_Time_ms'])    if 'Inf_Time_ms' in df.columns else 0.0,
        'total_time_mean':  s_mean(df['Total_ms'])      if 'Total_ms'    in df.columns else 0.0,
        'total_time_std':   s_std(df['Total_ms'])       if 'Total_ms'    in df.columns else 0.0,
        'fps_mean':         s_mean(df['FPS'])           if 'FPS'         in df.columns else 0.0,
        'fps_min':          s_min(df['FPS'])            if 'FPS'         in df.columns else 0.0,
        'fps_max':          s_max(df['FPS'])            if 'FPS'         in df.columns else 0.0,
        'fps_std':          s_std(df['FPS'])            if 'FPS'         in df.columns else 0.0,
        'conf_mean':        s_mean(df_valid['Conf'])    if len(df_valid) > 0 else 0.0,
        'conf_std':         s_std(df_valid['Conf'])     if len(df_valid) > 0 else 0.0,
        'label_distribution': dict(df_valid['Label'].value_counts()) if 'Label' in df_valid.columns else {},
        'accuracy':          accuracy,
        'accuracy_samples':  accuracy_samples,
        'durasi_detik':      durasi_detik,
        'jumlah_run':        jumlah_run,
    }


def print_comparison_table(hog, haar):
    print("\n" + "=" * 120)
    print("PERBANDINGAN PERFORMA DETEKSI & KLASIFIKASI GESTURE WAJAH: HOG+DNN vs Haar Cascade+DNN")
    print("=" * 120)
    print(f"{'Metrik':<25} {'HOG+DNN':<20} {'Haar+DNN':<20} {'Selisih (HOG-Haar)':<20}")
    print("-" * 120)

    rows_int = [
        ("Total Frame",          hog['total_frames'],    haar['total_frames']),
        ("Frame Valid",          hog['valid_frames'],    haar['valid_frames']),
        ("Frame None",           hog['none_frames'],     haar['none_frames']),
        ("Durasi Pengujian (s)", hog['durasi_detik'],    haar['durasi_detik']),
        ("Jumlah Run/Sesi",      hog['jumlah_run'],      haar['jumlah_run']),
    ]
    for lbl, g, h in rows_int:
        print(f"{lbl:<25} {g:<20} {h:<20} {g - h:+}")

    rows_float = [
        ("Accuracy (%)",         hog['accuracy'],        haar['accuracy'],       ".2f"),
        ("FPS Rata-rata",        hog['fps_mean'],        haar['fps_mean'],       ".2f"),
        ("FPS Min",              hog['fps_min'],         haar['fps_min'],        ".2f"),
        ("FPS Max",              hog['fps_max'],         haar['fps_max'],        ".2f"),
        ("Deteksi Mean (ms)",    hog['det_time_mean'],   haar['det_time_mean'],  ".3f"),
        ("Inferensi Mean (ms)",  hog['inf_time_mean'],   haar['inf_time_mean'],  ".3f"),
        ("Total Mean (ms)",      hog['total_time_mean'], haar['total_time_mean'],".3f"),
        ("Conf Mean",            hog['conf_mean'],       haar['conf_mean'],      ".4f"),
    ]
    for lbl, g, h, fmt in rows_float:
        diff = g - h
        print(f"{lbl:<25} {g:<20{fmt}} {h:<20{fmt}} {diff:+{fmt}}")

    print("-" * 120)
    print("DISTRIBUSI PREDIKSI GESTURE:")
    all_labels = set(hog['label_distribution']) | set(haar['label_distribution'])
    for label in sorted(all_labels):
        gc = hog['label_distribution'].get(label, 0)
        hc = haar['label_distribution'].get(label, 0)
        gp = gc / hog['valid_frames']  * 100 if hog['valid_frames']  > 0 else 0
        hp = hc / haar['valid_frames'] * 100 if haar['valid_frames'] > 0 else 0
        print(f"  {label.upper():<20} HOG+DNN: {gc:<8} ({gp:>5.1f}%)   Haar+DNN: {hc:<8} ({hp:>5.1f}%)")
    print("=" * 120)


def analyze_comparison(hog, haar):
    print("\n📊 ANALISIS PERBANDINGAN DETAIL:")
    print("-" * 120)

    hog_score = haar_score = 0

    # 1. Tingkat deteksi
    hog_det_rate  = (hog['valid_frames']  / hog['total_frames']  * 100
                     if hog['total_frames']  > 0 else 0)
    haar_det_rate = (haar['valid_frames'] / haar['total_frames'] * 100
                     if haar['total_frames'] > 0 else 0)
    print(f"\n1. TINGKAT DETEKSI WAJAH:")
    print(f"   HOG+DNN  : {hog_det_rate:.2f}%  ({hog['valid_frames']} / {hog['total_frames']} frame)")
    print(f"   Haar+DNN : {haar_det_rate:.2f}%  ({haar['valid_frames']} / {haar['total_frames']} frame)")
    if   hog_det_rate  > haar_det_rate + 0.01: print("   → HOG+DNN LEBIH BAIK");  hog_score  += 1
    elif haar_det_rate > hog_det_rate  + 0.01: print("   → Haar+DNN LEBIH BAIK"); haar_score += 1
    else:                                       print("   → SETARA")

    # 2. FPS
    fps_diff = hog['fps_mean'] - haar['fps_mean']
    print(f"\n2. FPS RATA-RATA (Durasi {SESSION_DURATION}s per sesi):")
    print(f"   HOG+DNN  : {hog['fps_mean']:.2f} FPS  (min {hog['fps_min']:.2f} | max {hog['fps_max']:.2f} | σ={hog['fps_std']:.2f})")
    print(f"   Haar+DNN : {haar['fps_mean']:.2f} FPS  (min {haar['fps_min']:.2f} | max {haar['fps_max']:.2f} | σ={haar['fps_std']:.2f})")
    if   fps_diff >  1.0: print(f"   → HOG+DNN LEBIH CEPAT   {fps_diff:.2f} FPS ({fps_diff/haar['fps_mean']*100:.1f}% lebih tinggi)"); hog_score  += 1
    elif fps_diff < -1.0: print(f"   → Haar+DNN LEBIH CEPAT  {abs(fps_diff):.2f} FPS ({abs(fps_diff)/hog['fps_mean']*100:.1f}% lebih tinggi)"); haar_score += 1
    else:                 print("   → FPS SETARA")

    # 3. Waktu deteksi wajah
    det_diff = hog['det_time_mean'] - haar['det_time_mean']
    print(f"\n3. WAKTU DETEKSI WAJAH (Det_Time_ms):")
    print(f"   HOG+DNN  : {hog['det_time_mean']:.3f} ms  (σ={hog['det_time_std']:.3f})")
    print(f"   Haar+DNN : {haar['det_time_mean']:.3f} ms  (σ={haar['det_time_std']:.3f})")
    if   det_diff < -1.0: print(f"   → HOG+DNN LEBIH CEPAT   {abs(det_diff):.3f} ms"); hog_score  += 1
    elif det_diff >  1.0: print(f"   → Haar+DNN LEBIH CEPAT  {det_diff:.3f} ms");       haar_score += 1
    else:                 print("   → SETARA")

    # 4. Waktu inferensi
    inf_diff = hog['inf_time_mean'] - haar['inf_time_mean']
    print(f"\n4. WAKTU INFERENSI DNN (Inf_Time_ms):")
    print(f"   HOG+DNN  : {hog['inf_time_mean']:.3f} ms  (σ={hog['inf_time_std']:.3f})")
    print(f"   Haar+DNN : {haar['inf_time_mean']:.3f} ms  (σ={haar['inf_time_std']:.3f})")
    if   inf_diff < -1.0: print(f"   → HOG+DNN LEBIH CEPAT   {abs(inf_diff):.3f} ms"); hog_score  += 1
    elif inf_diff >  1.0: print(f"   → Haar+DNN LEBIH CEPAT  {inf_diff:.3f} ms");       haar_score += 1
    else:                 print("   → SETARA")

    # 5. Total waktu per frame
    tot_diff = hog['total_time_mean'] - haar['total_time_mean']
    print(f"\n5. TOTAL WAKTU PER FRAME (Total_ms):")
    print(f"   HOG+DNN  : {hog['total_time_mean']:.3f} ms  (σ={hog['total_time_std']:.3f})")
    print(f"   Haar+DNN : {haar['total_time_mean']:.3f} ms  (σ={haar['total_time_std']:.3f})")
    if   tot_diff < -1.0: print(f"   → HOG+DNN LEBIH CEPAT   {abs(tot_diff):.3f} ms"); hog_score  += 1
    elif tot_diff >  1.0: print(f"   → Haar+DNN LEBIH CEPAT  {tot_diff:.3f} ms");       haar_score += 1
    else:                 print("   → SETARA")

    # 6. Confidence
    conf_diff = hog['conf_mean'] - haar['conf_mean']
    print(f"\n6. CONFIDENCE SCORE:")
    print(f"   HOG+DNN  : {hog['conf_mean']:.4f}  (σ={hog['conf_std']:.4f})")
    print(f"   Haar+DNN : {haar['conf_mean']:.4f}  (σ={haar['conf_std']:.4f})")
    if   conf_diff >  0.01: print(f"   → HOG+DNN LEBIH PERCAYA DIRI  {conf_diff:.4f}"); hog_score  += 1
    elif conf_diff < -0.01: print(f"   → Haar+DNN LEBIH PERCAYA DIRI {abs(conf_diff):.4f}"); haar_score += 1
    else:                   print("   → SETARA")

    # 7. Akurasi klasifikasi
    acc_diff = hog['accuracy'] - haar['accuracy']
    print(f"\n7. AKURASI KLASIFIKASI GESTURE:")
    print(f"   HOG+DNN  : {hog['accuracy']:.2f}%  (n={hog['accuracy_samples']} frame dg GT_Label)")
    print(f"   Haar+DNN : {haar['accuracy']:.2f}%  (n={haar['accuracy_samples']} frame dg GT_Label)")
    if hog['accuracy_samples'] > 0 and haar['accuracy_samples'] > 0:
        if   acc_diff >  1.0: print(f"   → HOG+DNN akurasi LEBIH TINGGI  ({hog['accuracy']:.2f}% vs {haar['accuracy']:.2f}%)"); hog_score  += 1
        elif acc_diff < -1.0: print(f"   → Haar+DNN akurasi LEBIH TINGGI ({haar['accuracy']:.2f}% vs {hog['accuracy']:.2f}%)"); haar_score += 1
        else:                 print("   → Akurasi SETARA (selisih < 1%)")
    else:
        print("   → Data GT_Label tidak cukup untuk perbandingan akurasi.")

    # 8. Durasi & run
    print(f"\n8. DURASI & SESI PENGUJIAN (per sesi = {SESSION_DURATION} detik):")
    print(f"   HOG+DNN  : {hog['jumlah_run']} sesi  →  total {hog['durasi_detik']} detik")
    print(f"   Haar+DNN : {haar['jumlah_run']} sesi  →  total {haar['durasi_detik']} detik")

    # Kesimpulan
    print("\n" + "=" * 120)
    print("KESIMPULAN:")
    print("-" * 120)
    print(f"\n  SKOR  →  HOG+DNN: {hog_score} poin  |  Haar+DNN: {haar_score} poin\n")
    if   hog_score  > haar_score:  print(f"  🏆 HOG+DNN UNGGUL dengan selisih {hog_score - haar_score} poin")
    elif haar_score > hog_score:   print(f"  🏆 Haar+DNN UNGGUL dengan selisih {haar_score - hog_score} poin")
    else:                          print("  ⚖️  PERFORMA SETARA")

    print("\nREKOMENDASI:")
    print("-" * 120)
    if hog_score > haar_score:
        print("  Untuk aplikasi real-time yang memerlukan konsistensi tinggi → Gunakan HOG+DNN")
    elif haar_score > hog_score:
        print("  Untuk aplikasi real-time yang memerlukan kecepatan deteksi   → Gunakan Haar+DNN")
    else:
        print("  Pilih berdasarkan prioritas: kecepatan → Haar+DNN | akurasi → HOG+DNN")
    print("=" * 120)


def main():
    print("[INFO] Membaca hasil eksekusi sistem...")
    df_hog  = load_csv_results(HOG_LOG_FILE)
    df_haar = load_csv_results(HAAR_LOG_FILE)

    if df_hog is None or df_haar is None:
        print("[ERROR] Pastikan sudah menjalankan final_hogdnn.py dan final_haarcascadednn.py")
        return

    print("[INFO] Menghitung statistik...")
    hog_stats  = calculate_statistics(df_hog)
    haar_stats = calculate_statistics(df_haar)

    if hog_stats is None or haar_stats is None:
        print("[ERROR] Gagal menghitung statistik.")
        return

    print_comparison_table(hog_stats, haar_stats)
    analyze_comparison(hog_stats, haar_stats)

    # Simpan laporan
    import sys
    from io import StringIO
    buf = StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf

    print_comparison_table(hog_stats, haar_stats)
    analyze_comparison(hog_stats, haar_stats)

    sys.stdout = old_stdout
    report_text = buf.getvalue()

    try:
        with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
            f.write(f"LAPORAN PERBANDINGAN DETEKTOR WAJAH\n")
            f.write(f"Tanggal : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Durasi  : {SESSION_DURATION} detik per sesi\n")
            f.write("=" * 120 + "\n")
            f.write(report_text)
        print(f"\n[✓] Laporan disimpan ke: {OUTPUT_REPORT}")
    except Exception as e:
        print(f"\n[WARNING] Gagal menyimpan laporan: {e}")


if __name__ == "__main__":
    main()