# 🦷 Dental Görüntü Analizi ve Diş Çürüğü Tespit Sistemi

Bu proje, panoramik dental röntgen görüntülerinde diş çürüklerini tespit etmek ve analiz etmek amacıyla geliştirilmiş hibrit bir sistemdir. Geleneksel kural tabanlı görüntü işleme teknikleri ile modern derin öğrenme (Deep Learning) mimarilerini bir araya getirerek daha hassas sonuçlar sunar.

## 🚀 Öne Çıkan Özellikler

* **Üçlü Analiz Modu:**
    * **Fizik Tabanlı:** Görüntü yoğunluğu ve kural tabanlı algoritmalarla hızlı analiz.
    * **ML (Makine Öğrenmesi):** EfficientNet-B4 tabanlı derin öğrenme modeli ile yüksek doğrulukta segmentasyon.
    * **Hibrit Mod:** Fiziksel kurallar ve ML sonuçlarını birleştiren "Altın Oran" yaklaşımı.
* **Kullanıcı Dostu Arayüz:** PyQt5 ile geliştirilmiş, röntgen ve maske yükleme imkanı sunan profesyonel GUI.
* **Detaylı Metrik Raporlama:** IoU, Dice Katsayısı, Precision ve Recall metrikleri ile performans ölçümü.
* **Şiddet Analizi:** Tespit edilen çürüklerin derinliğine göre (Başlangıç, Orta, Kritik) sınıflandırılması.

## 📂 Proje Yapısı

* `main_ui.py`: Uygulamanın ana mantığı ve buton işlevlerinin yönetildiği dosya.
* `dental_ui.py`: PyQt5 ile tasarlanmış kullanıcı arayüzü bileşenleri.
* `mlkod.txt`: Derin öğrenme modelinin eğitim ve çıkarım (inference) süreçleri.
* `kuralTabanlıkod.py`: Görüntü işleme teknikleri ile yapılan kural tabanlı analizler.
* `hibritkod.txt`: Fiziksel ve ML tabanlı modellerin birleştirildiği hibrit algoritma.

## 🛠️ Kurulum

1. Bu depoyu klonlayın:
   ```bash
   git clone [https://github.com/kullaniciadi/dental-projesi.git](https://github.com/kullaniciadi/dental-projesi.git)
   
2. Gerekli kütüphaneleri yükleyin:
   pip install -r requirements.txt
(Gerekli ana kütüphaneler: torch, torchvision, opencv-python, PyQt5, numpy, monai)

3. Uygulamayı başlatın:
   python main_ui.py

## 📊 Performans ve Metrik Analizi

Projenin başarısı, test veri seti üzerinde standart tıbbi görüntüleme metrikleri kullanılarak ölçülmüştür. Hibrit modelimiz, sadece ML veya sadece kural tabanlı sistemlere göre daha dengeli sonuçlar vermektedir.

### Elde Edilen Ortalama Değerler:

| Metrik | Açıklama | Başarı Oranı |
| **Dice Katsayısı** | Maske benzerlik oranı(F1-Skor)| %45.88 |
| **IoU (Jaccard Index)** | Kesişim/Birleşim oranı | %31.96 |
| **Precision** | Doğru pozitif tahmin hassasiyeti | %41.38 |
| **Recall (Duyarlılık)** | Gerçek çürükleri yakalama oranı | %62.72 |

Not: Bu degerler; EfficientNet-B4 derin öğrenme mimarisi ile geleneksel görüntü işleme kurallarını birleştiren "Altın Oran" hibrit modelinin final test sonuçlarını temsil etmektedir

Hibrit model, yapay zeka ve fiziksel analizleri birleştirerek diş çürüğü tespitinde tekil yöntemlere göre çok daha dengeli ve güvenilir sonuçlar vermektedir.














