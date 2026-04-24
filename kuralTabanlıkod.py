import cv2
import numpy as np
import os
import time
import csv

# --- AYARLAR ---
PATH_TEST_IMG = r"C:\Users\aaaa\Desktop\test_images"
PATH_TEST_MASK = r"C:\Users\aaaa\Desktop\test_masks"
PATH_DEBUG_OUT = r"C:\Users\aaaa\Desktop\Sonuc_KuralTabanli_Metrikli"

# --- YARDIMCI: Siddet Analizi ---
def analyze_severity(mean_intensity):
    # Röntgen negatif oldugu icin: Dusuk deger (Siyah) = Derin Curuk
    if mean_intensity < 60:
        return "KRITIK (Derin)"
    elif mean_intensity < 100:
        return "ORTA SEVIYE"
    else:
        return "BASLANGIC"

# --- ISLEM ---
def process_professional():
    # Klasör yoksa oluştur
    if not os.path.exists(PATH_DEBUG_OUT):
        os.makedirs(PATH_DEBUG_OUT)
        
    rapor_path = "KuralTabanli_Performans_Raporu.csv"
    
    print("PROFESYONEL KURAL TABANLI MOD BASLATILDI...")
    print("OZELLIKLER: Detaylı Metrik Analizi (IoU, Dice, F1) + Derinlik Analizi")

    sayac = 0
    islenen_dosya = 0
    
    # Ortalama metrikler için toplam tutucular
    sum_iou = 0
    sum_dice = 0
    sum_precision = 0
    sum_recall = 0
    sum_f1 = 0

    # CSV dosyasını aç
    with open(rapor_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Hibrit kod ile uyumlu başlıklar
        writer.writerow(["Dosya", "IoU", "Dice", "Precision", "Recall", "F1", "Siddet", "Sure_ms"])

        # Klasördeki dosyaları oku
        for filename_with_ext in os.listdir(PATH_TEST_IMG):
            img_path = os.path.join(PATH_TEST_IMG, filename_with_ext)
            if not os.path.isfile(img_path):
                continue
                
            filename = os.path.splitext(filename_with_ext)[0]
            
            # Maske yolu kontrolü
            mask_path = os.path.join(PATH_TEST_MASK, f"{filename}.png")
            if not os.path.exists(mask_path):
                mask_path = os.path.join(PATH_TEST_MASK, f"{filename}.jpg")
            if not os.path.exists(mask_path):
                continue

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None or gt_mask is None:
                continue

            # --- ZAMANLAYICI BASLAT ---
            start_time = time.perf_counter()

            # 1. GÖRÜNTÜ İŞLEME
            proc_img = cv2.GaussianBlur(img, (5, 5), 0)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            proc_img = clahe.apply(proc_img)
            thresh = cv2.adaptiveThreshold(proc_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY_INV, 45, 5)

            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel_dilate)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open)

            # 2. TAHMİN VE ANALİZ
            prediction_mask = np.zeros_like(img, dtype=np.uint8)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            leke_sayisi = 0
            toplam_yogunluk = 0

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 10: 
                    cv2.drawContours(prediction_mask, [cnt], -1, 255, cv2.FILLED)
                    mask_roi = np.zeros_like(img, dtype=np.uint8)
                    cv2.drawContours(mask_roi, [cnt], -1, 255, cv2.FILLED)
                    mean_val = cv2.mean(img, mask=mask_roi)
                    toplam_yogunluk += mean_val[0]
                    leke_sayisi += 1

            # --- ZAMANLAYICI DURDUR ---
            ms = int((time.perf_counter() - start_time) * 1000)

            # 3. METRİK HESAPLAMA (GÜNCELLENEN BÖLÜM) [cite: 31, 32, 33]
            # GT Maskeyi temizle ve eşik değerine getir
            gt_resized = cv2.resize(gt_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            _, gt_binary = cv2.threshold(gt_resized, 127, 255, cv2.THRESH_BINARY)

            intersection = cv2.bitwise_and(prediction_mask, gt_binary)
            TP = cv2.countNonZero(intersection)
            FP = cv2.countNonZero(prediction_mask) - TP
            FN = cv2.countNonZero(gt_binary) - TP

            # Formüller [cite: 32, 33]
            iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
            dice = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Ortalama için toplamlara ekle
            sum_iou += iou
            sum_dice += dice
            sum_precision += precision
            sum_recall += recall
            sum_f1 += f1
            islenen_dosya += 1

            # Şiddet durumu
            avg_intensity = (toplam_yogunluk / leke_sayisi) if leke_sayisi > 0 else 255
            siddet_durumu = analyze_severity(avg_intensity)
            
            # CSV Satırı Yazdır
            writer.writerow([filename, round(iou, 4), round(dice, 4), round(precision, 4), 
                             round(recall, 4), round(f1, 4), siddet_durumu, ms])

            # 4. GÖRSELLEŞTİRME (OPSİYONEL - İLK 50 DOSYA)
            if sayac < 50:
                debug_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                gt_cnt, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(debug_img, gt_cnt, -1, (0, 255, 0), 2) # Yeşil: Gerçek Çürük

                color_map = cv2.applyColorMap(img, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(debug_img, 0.7, color_map, 0.3, 0)
                final_vis = debug_img.copy()
                final_vis = cv2.copyTo(overlay, prediction_mask, final_vis)

                # Bilgileri Yaz
                cv2.putText(final_vis, f"Dice: {dice:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(final_vis, f"Siddet: {siddet_durumu}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.putText(final_vis, f"Sure: {ms}ms", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                cv2.imwrite(os.path.join(PATH_DEBUG_OUT, f"Result_{filename}.jpg"), final_vis)

            sayac += 1
            if sayac % 20 == 0:
                print(f"İşlenen: {sayac} | Dice: {dice:.2f} | Şiddet: {siddet_durumu}")

        # SONUÇ ÖZETİ (CSV ve Konsol) 
        if islenen_dosya > 0:
            avg_iou = sum_iou / islenen_dosya
            avg_dice = sum_dice / islenen_dosya
            avg_prec = sum_precision / islenen_dosya
            avg_rec = sum_recall / islenen_dosya
            avg_f1 = sum_f1 / islenen_dosya

            writer.writerow([])
            writer.writerow(["GENEL ORTALAMA PERFORMANS"])
            writer.writerow(["IoU", round(avg_iou, 4), "Dice", round(avg_dice, 4), 
                             "Precision", round(avg_prec, 4), "Recall", round(avg_rec, 4), "F1", round(avg_f1, 4)])

            print("\n" + "="*50)
            print(f"KURAL TABANLI GENEL SONUÇLAR ({islenen_dosya} Dosya)")
            print(f"Ortalama IoU:       {avg_iou:.4f}")
            print(f"Ortalama Dice:      {avg_dice:.4f}")
            print(f"Ortalama Precision: {avg_prec:.4f}")
            print(f"Ortalama Recall:    {avg_rec:.4f}")
            print(f"Ortalama F1-Skor:   {avg_f1:.4f}")
            print("="*50)

    print(f"\nRapor kaydedildi: {rapor_path}")

if __name__ == "__main__":
    process_professional()