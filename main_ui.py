import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b4
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog,
                              QMessageBox, QPushButton, QLabel, QTextEdit)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from dental_ui import Ui_MainWindow

# ============================================================
# AYARLAR
# ============================================================
PATH_MODEL    = r"C:\Users\Eda Gül ULUSOY\Desktop\dental_project\best_model.pth"
ML_THRESHOLD  = 0.35
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# ML MODEL MİMARİSİ
# ============================================================
class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch)
        )
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch=256):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=6,  dilation=6,  bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=18, dilation=18, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.pool  = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(out_ch * 5, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout(0.5))
    def forward(self, x):
        size = x.shape[2:]
        x1 = self.conv1(x); x2 = self.conv2(x); x3 = self.conv3(x); x4 = self.conv4(x)
        x5 = F.interpolate(self.pool(x), size=size, mode="bilinear", align_corners=True)
        return self.project(torch.cat([x1, x2, x3, x4, x5], dim=1))

class DentalSegNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        backbone     = efficientnet_b4(pretrained=False)
        self.encoder = backbone.features
        self.aspp    = ASPP(in_ch=1792, out_ch=256)
        self.dec5    = DecoderBlock(256, 112, 128)
        self.dec4    = DecoderBlock(128,  56,  64)
        self.dec3    = DecoderBlock( 64,  32,  32)
        self.dec2    = DecoderBlock( 32,  24,  16)
        self.dec1    = DecoderBlock( 16,   0,   8)
        self.s1_conv = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True), ConvBnRelu(48, 8))
        self.dec0_conv = nn.Sequential(ConvBnRelu(8 + 8, 8), ConvBnRelu(8, 8))
        self.seg_head  = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, num_classes, kernel_size=1)
        )
        self.dropout = nn.Dropout2d(p=0.3)

    def forward(self, x):
        skips = {}
        for i, block in enumerate(self.encoder):
            x = block(x)
            if i == 0: skips["s1"] = x
            elif i == 1: skips["s2"] = x
            elif i == 2: skips["s3"] = x
            elif i == 3: skips["s4"] = x
            elif i == 4: skips["s5"] = x
        x = self.aspp(x)
        x = self.dropout(x)
        x = self.dec5(x, skips["s5"])
        x = self.dec4(x, skips["s4"])
        x = self.dec3(x, skips["s3"])
        x = self.dec2(x, skips["s2"])
        x = self.dec1(x, None)
        s1_up = self.s1_conv(skips["s1"])
        x = torch.cat([x, s1_up], dim=1)
        x = self.dec0_conv(x)
        return self.seg_head(x)

# ============================================================
# BACKEND FONKSİYONLARI
# ============================================================
_ml_model = None

def load_ml_model():
    global _ml_model
    if _ml_model is not None:
        return _ml_model
    model = DentalSegNet().to(DEVICE)
    model.eval()
    if os.path.exists(PATH_MODEL):
        ckpt = torch.load(PATH_MODEL, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        print("[INFO] ML modeli yüklendi.")
    else:
        print(f"[HATA] Model bulunamadı: {PATH_MODEL}")
    _ml_model = model
    return _ml_model

def run_physics(img_path, mask_path):
    img     = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    gt_mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

    proc = cv2.GaussianBlur(img, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    proc = clahe.apply(proc)
    thresh = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 45, 5)
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_opn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, k_dil)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,   k_opn)

    pred_mask    = np.zeros_like(img, dtype=np.uint8)
    contours, _  = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    leke = 0; yogunluk = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 10:
            cv2.drawContours(pred_mask, [cnt], -1, 255, cv2.FILLED)
            roi = np.zeros_like(img, dtype=np.uint8)
            cv2.drawContours(roi, [cnt], -1, 255, cv2.FILLED)
            yogunluk += cv2.mean(img, mask=roi)[0]
            leke += 1

    avg_int = yogunluk / leke if leke > 0 else 255
    if avg_int < 60:    siddet = "KRİTİK (Derin)"
    elif avg_int < 100: siddet = "ORTA SEVİYE"
    else:               siddet = "BAŞLANGIÇ"

    debug  = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    gt_r   = cv2.resize(gt_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    _, gt_r = cv2.threshold(gt_r, 10, 255, cv2.THRESH_BINARY)
    gt_cnt, _ = cv2.findContours(gt_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug, gt_cnt, -1, (0, 255, 0), 2)

    color_map = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    overlay   = cv2.addWeighted(debug, 0.7, color_map, 0.3, 0)
    final_vis = debug.copy()
    final_vis = cv2.copyTo(overlay, pred_mask, final_vis)

    intersection = cv2.bitwise_and(pred_mask, gt_r)
    durum = "TESPİT EDİLDİ ✓" if cv2.countNonZero(intersection) > 0 else "ISKALI"

    cv2.putText(final_vis, f"Durum: {durum}",   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(final_vis, f"Siddet: {siddet}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    cv2.putText(final_vis, f"Leke: {leke}",     (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    rapor = f"Durum: {durum}\nŞiddet: {siddet}\nTespit Edilen Leke: {leke}"
    return [bgr_to_qimage(final_vis)], rapor

def run_ml(img_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tempfile

    model       = load_ml_model()
    img_bgr     = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img_bgr, (512, 512))
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm    = img_rgb.astype(np.float32) / 255.0
    tensor      = torch.from_numpy(np.transpose(img_norm, (2, 0, 1))).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits1  = model(tensor)
        prob1    = torch.sigmoid(logits1).squeeze().cpu().numpy()
        t_flip   = torch.flip(tensor, dims=[3])
        logits2  = model(t_flip)
        prob2    = np.fliplr(torch.sigmoid(logits2).squeeze().cpu().numpy())
        prob_map = (prob1 + prob2) / 2.0

    mask_bin = (prob_map > ML_THRESHOLD).astype(np.uint8)

    overlay = img_norm.copy()
    overlay[mask_bin == 1] = overlay[mask_bin == 1] * 0.4 + np.array([1.0, 0.2, 0.2]) * 0.6

    num_labels, label_map = cv2.connectedComponents(mask_bin.astype(np.uint8))
    region_colored = np.zeros((*mask_bin.shape, 3), dtype=np.float32)
    colors = [[1.0,0.2,0.2],[0.2,0.8,1.0],[1.0,0.8,0.1],[0.6,1.0,0.2],[1.0,0.4,1.0]]
    for lbl in range(1, num_labels):
        c = colors[(lbl - 1) % len(colors)]
        region_colored[label_map == lbl] = c

    cavity_pct  = mask_bin.sum() / mask_bin.size * 100
    num_regions = num_labels - 1
    durum       = "ÇÜRÜK TESPİT EDİLDİ ⚠️" if mask_bin.sum() > 0 else "SAĞLAM ✅"

    images_q = []
    for title, data, cmap, alpha_data in [
        ("Olasılık Haritası", prob_map,       "hot",  None),
        ("Binary Maske",      mask_bin,        "gray", None),
        ("Overlay",           overlay,         None,   None),
        ("Bölge Haritası",    img_norm,        "gray", region_colored),
    ]:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(title, fontsize=11)
        ax.axis("off")
        if cmap:
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(data)
        if alpha_data is not None:
            ax.imshow(alpha_data, alpha=0.6)
        plt.tight_layout()
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        plt.savefig(tmp.name, dpi=100, bbox_inches="tight")
        plt.close(fig)
        pix = QPixmap(tmp.name).copy()
        images_q.append(pix)
        os.unlink(tmp.name)

    rapor = (f"Durum: {durum}\n"
             f"Bölge Sayısı: {num_regions}\n"
             f"Çürük Alan: %{cavity_pct:.2f}\n"
             f"Max Olasılık: {prob_map.max():.4f}")
    return images_q, rapor

def run_hybrid(img_path, mask_path):
    model    = load_ml_model()
    img_bgr  = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Görüntü okunamadı: {img_path}")
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Rule mask
    proc = cv2.GaussianBlur(img_gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    proc = clahe.apply(proc)
    thresh = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 45, 5)
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k_opn = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, k_dil)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,   k_opn)
    rule_mask = np.zeros_like(img_gray, dtype=np.uint8)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) > 10:
            cv2.drawContours(rule_mask, [cnt], -1, 255, cv2.FILLED)

    # ML mask
    img_resized = cv2.resize(img_bgr, (512, 512))
    img_rgb  = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    tensor   = torch.from_numpy(np.transpose(img_norm, (2, 0, 1))).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits1    = model(tensor)
        prob1      = torch.sigmoid(logits1).squeeze().cpu().numpy()
        t_flip     = torch.flip(tensor, dims=[3])
        logits2    = model(t_flip)
        prob2      = np.fliplr(torch.sigmoid(logits2).squeeze().cpu().numpy())
        prob_final = (prob1 + prob2) / 2.0
    ml_mask_small = (prob_final > ML_THRESHOLD).astype(np.uint8) * 255
    ml_mask = cv2.resize(ml_mask_small, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Hibrit
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ml_dilated = cv2.dilate(ml_mask, k7, iterations=1)
    hybrid     = cv2.bitwise_and(rule_mask, ml_dilated)
    clean      = np.zeros_like(hybrid)
    cnts2, _   = cv2.findContours(hybrid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts2:
        if cv2.contourArea(cnt) > 100:
            cv2.drawContours(clean, [cnt], -1, 255, cv2.FILLED)
    k5         = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k5)

    # GT değerlendirme
    gt_mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        raise ValueError(f"Mask okunamadı: {mask_path}")
    gt_r    = cv2.resize(gt_mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    _, gt_r = cv2.threshold(gt_r, 127, 255, cv2.THRESH_BINARY)
    inter   = cv2.bitwise_and(final_mask, gt_r)
    TP = cv2.countNonZero(inter)
    FP = cv2.countNonZero(final_mask) - TP
    FN = cv2.countNonZero(gt_r) - TP
    iou       = TP/(TP+FP+FN) if (TP+FP+FN) > 0 else 0
    precision = TP/(TP+FP)    if (TP+FP)    > 0 else 0
    recall    = TP/(TP+FN)    if (TP+FN)    > 0 else 0

    rule_vis = cv2.cvtColor(rule_mask, cv2.COLOR_GRAY2BGR)
    ml_vis   = cv2.cvtColor(ml_mask,   cv2.COLOR_GRAY2BGR)

    rapor = (f"IoU: {iou:.4f}\n"
             f"Precision: {precision:.4f}\n"
             f"Recall: {recall:.4f}\n"
             f"TP: {TP}  FP: {FP}  FN: {FN}")
    return [bgr_to_qimage(rule_vis), bgr_to_qimage(ml_vis)], rapor

# ============================================================
# YARDIMCI
# ============================================================
def bgr_to_qimage(bgr_img):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()

# ============================================================
# ARKAPLAN İŞ PARÇACIĞI
# ============================================================
class AnalysisThread(QThread):
    finished = pyqtSignal(list, str)
    error    = pyqtSignal(str)

    def __init__(self, mode, img_path, mask_path):
        super().__init__()
        self.mode      = mode
        self.img_path  = img_path
        self.mask_path = mask_path

    def run(self):
        try:
            if self.mode == "physics":
                images, rapor = run_physics(self.img_path, self.mask_path)
            elif self.mode == "ml":
                images, rapor = run_ml(self.img_path)
            else:
                images, rapor = run_hybrid(self.img_path, self.mask_path)
            self.finished.emit(images, rapor)
        except Exception as e:
            self.error.emit(str(e))

# ============================================================
# ANA PENCERE
# ============================================================
RESULT_LABEL_STYLE = (
    "font-size: 11px; color: white; background-color: #1e1e1e;"
    "border-radius: 4px; padding: 4px;"
)

def make_result_textedit(parent):
    """Scrollable, read-only metin kutusu — sonuç raporları için."""
    te = QTextEdit(parent)
    te.setReadOnly(True)
    te.setStyleSheet(RESULT_LABEL_STYLE)
    te.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    te.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return te

class DentalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.xray_path  = None
        self.mask_path  = None
        self.threads    = []

        self.physics_images = []
        self.physics_idx    = 0
        self.ml_images      = []
        self.ml_idx         = 0
        self.hybrid_images  = []
        self.hybrid_idx     = 0

        # ── Buton bağlantıları ───────────────────────────────
        self.ui.pushButton.clicked.connect(self.load_xray)
        self.ui.pushButton_2.clicked.connect(self.load_mask)
        self.ui.pushButton_3.clicked.connect(self.analyze_physics)
        self.ui.pushButton_4.clicked.connect(self.analyze_ml)
        self.ui.pushButton_5.clicked.connect(self.analyze_hybrid)

        # ── Fizik ok butonları ───────────────────────────────
        self.btn_phy_prev = QPushButton("◀", self)
        self.btn_phy_next = QPushButton("▶", self)
        self.btn_phy_prev.clicked.connect(lambda: self.navigate_image("physics", -1))
        self.btn_phy_next.clicked.connect(lambda: self.navigate_image("physics", +1))
        self.btn_phy_prev.hide()
        self.btn_phy_next.hide()

        # ── ML ok butonları ──────────────────────────────────
        self.btn_ml_prev = QPushButton("◀", self)
        self.btn_ml_next = QPushButton("▶", self)
        self.btn_ml_prev.clicked.connect(lambda: self.navigate_image("ml", -1))
        self.btn_ml_next.clicked.connect(lambda: self.navigate_image("ml", +1))

        # ── Hibrit ok butonları ──────────────────────────────
        self.btn_hyb_prev = QPushButton("◀", self)
        self.btn_hyb_next = QPushButton("▶", self)
        self.btn_hyb_prev.clicked.connect(lambda: self.navigate_image("hybrid", -1))
        self.btn_hyb_next.clicked.connect(lambda: self.navigate_image("hybrid", +1))

        # ── Sonuç metin alanları (scrollable) ───────────────
        self.label_physics_result = make_result_textedit(self)
        self.label_ml_result      = make_result_textedit(self)
        self.label_hybrid_result  = make_result_textedit(self)

    # ── Görüntü Yükleme ──────────────────────────────────────
    # Röntgen klasörü → mask klasörü eşleştirme tablosu
    MASK_FOLDER_MAP = {
        "test_images": "test_masks",
        "images_val":  "masks_val",
        "images":      "masks",
    }

    def find_auto_mask(self, xray_path):
        """Röntgen yolundan otomatik mask yolunu türetir. Bulunamazsa None döner."""
        xray_path = os.path.normpath(xray_path)
        folder    = os.path.basename(os.path.dirname(xray_path))
        filename  = os.path.basename(xray_path)
        parent    = os.path.dirname(os.path.dirname(xray_path))

        mask_folder = self.MASK_FOLDER_MAP.get(folder)
        if not mask_folder:
            return None

        # Aynı uzantıyla dene, bulamazsan diğer yaygın uzantıları dene
        for ext in [None, ".png", ".jpg", ".jpeg"]:
            if ext is None:
                candidate = os.path.join(parent, mask_folder, filename)
            else:
                base = os.path.splitext(filename)[0]
                candidate = os.path.join(parent, mask_folder, base + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def load_xray(self):
        path, _ = QFileDialog.getOpenFileName(self, "Röntgen Seç", "", "Görüntü (*.png *.jpg *.jpeg)")
        if path:
            self.xray_path = path
            self.show_image(self.ui.label, QPixmap(path))

            # Otomatik mask bul
            mask_path = self.find_auto_mask(path)
            if mask_path:
                self.mask_path = mask_path
                self.show_image(self.ui.label_2, QPixmap(mask_path))
            else:
                self.mask_path = None
                self.ui.label_2.clear()

    
    def calculate_all_metrics(self, pred_image, gt_path):
        # 1. Maske yolu yoksa veya dosya mevcut değilse sıfır döndür
        if gt_path is None or not os.path.exists(gt_path):
            print("[UYARI] Maske dosyası bulunamadı, metrikler hesaplanamıyor.")
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        try:
            # Görüntü format dönüşümleri (Önceki adımda yaptığımız QPixmap/QImage kontrolleri)
            if not hasattr(pred_image, 'shape'):
                if isinstance(pred_image, QPixmap):
                    temp_image = pred_image.toImage()
                else:
                    temp_image = pred_image
                temp_image = temp_image.convertToFormat(QImage.Format_RGBA8888)
                width, height = temp_image.width(), temp_image.height()
                ptr = temp_image.bits()
                ptr.setsize(height * width * 4)
                arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
                pred_image_np = cv2.cvtColor(arr, cv2.COLOR_RGBA2GRAY)
            else:
                pred_image_np = pred_image

            # Gerçek maskeyi oku
            img_array = np.fromfile(gt_path, np.uint8)
            gt_mask = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
            # Boyut eşitleme ve eşikleme
            if pred_image_np.shape[:2] != gt_mask.shape[:2]:
                gt_mask = cv2.resize(gt_mask, (pred_image_np.shape[1], pred_image_np.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            _, pred_bin = cv2.threshold(pred_image_np, 127, 255, cv2.THRESH_BINARY)
            _, gt_bin = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)

            # TP, FP, FN hesaplama
            intersection = cv2.bitwise_and(pred_bin, gt_bin)
            tp = cv2.countNonZero(intersection)
            fp = cv2.countNonZero(pred_bin) - tp
            fn = cv2.countNonZero(gt_bin) - tp

            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            return iou, dice, precision, recall, f1

        except Exception as e:
            print(f"[HATA] Metrik hesaplama sırasında hata: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0



    def load_mask(self):
        path, _ = QFileDialog.getOpenFileName(self, "Mask Seç", "", "Görüntü (*.png *.jpg *.jpeg)")
        if path:
            self.mask_path = path
            self.show_image(self.ui.label_2, QPixmap(path))

    def show_image(self, label, pixmap):
        if isinstance(pixmap, QImage):
            pixmap = QPixmap.fromImage(pixmap)
        scaled = pixmap.scaled(label.width(), label.height(),
                               Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)
        label.setAlignment(Qt.AlignCenter)

    # ── Giriş Kontrolü ───────────────────────────────────────
    def check_inputs(self):
        if not self.xray_path or not self.mask_path:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce röntgen ve mask görüntülerini yükleyin.")
            return False
        return True

    # ── Analiz Fonksiyonları ─────────────────────────────────
    def analyze_physics(self):
        if not self.check_inputs(): return
        self.ui.label_4.setText("İşleniyor...")
        t = AnalysisThread("physics", self.xray_path, self.mask_path)
        t.finished.connect(self.on_physics_done)
        t.error.connect(self.on_error)
        t.start(); self.threads.append(t)

    def analyze_ml(self):
        if not self.check_inputs(): return
        self.ui.label_5.setText("İşleniyor... (ML modeli yükleniyor)")
        t = AnalysisThread("ml", self.xray_path, self.mask_path)
        t.finished.connect(self.on_ml_done)
        t.error.connect(self.on_error)
        t.start(); self.threads.append(t)

    def analyze_hybrid(self):
        if not self.check_inputs(): return
        self.ui.label_6.setText("İşleniyor...")
        t = AnalysisThread("hybrid", self.xray_path, self.mask_path)
        t.finished.connect(self.on_hybrid_done)
        t.error.connect(self.on_error)
        t.start(); self.threads.append(t)

    # ── Sonuç Callback'leri ──────────────────────────────────
    def on_physics_done(self, images, rapor):
        self.physics_images = images
        self.physics_idx    = 0
        
        # İlk analiz görüntüsünü (maskeyi) ekranda göster
        self.show_image(self.ui.label_4, images[0])
        
        # 1. METRİK HESAPLAMA (Tahmin maskesi ve Gerçek maske karşılaştırması)
        # images[0] kural tabanlı algoritmanın ürettiği maske sonucudur
        iou, dice, prec, rec, f1 = self.calculate_all_metrics(images[0], self.mask_path)

        # 2. RAPOR METNİNİ ZENGİNLEŞTİRME (HTML formatında)
        detayli_rapor = (
            f"<b style='color:#4CAF50;'>FİZİK TABANLI ANALİZ RAPORU</b><br>"
            f"{'-'*35}<br>"
            f"<b>Dice Skoru:</b> %{dice*100:.2f}<br>"
            f"<b>Precision:</b> %{prec*100:.2f}<br>"
            f"<b>Recall:</b> %{rec*100:.2f}<br>"
            f"<b>F1-Score:</b> %{f1*100:.2f}<br>"
            f"<b>IoU:</b> {iou:.4f}<br><br>"
            f"<b>Fizik Tabanlı Model Detayları:</b><br>{rapor.replace('\n', '<br>')}"
        )

        # 3. ARAYÜZ GÜNCELLEME
        # ScrollArea içindeki label'a HTML formatında yazıyoruz
        self.label_physics_result.setText(detayli_rapor)
        
        # Ok butonlarını görünür yap (Eğer birden fazla sonuç görseli varsa)
        if len(images) > 1:
            self.btn_phy_prev.show()
            self.btn_phy_next.show()

        print("[Fizik Raporu Tamamlandı]")

    def on_ml_done(self, images, rapor):
        self.ml_images = images
        self.ml_idx    = 0
        
        # ML modelinin ürettiği ilk maskeyi ekranda göster
        self.show_image(self.ui.label_5, images[0])
        
        # 1. METRİK HESAPLAMA
        # Modelin tahmini (images[0]) ile gerçek maskeyi (self.mask_path) karşılaştırır
        iou, dice, prec, rec, f1 = self.calculate_all_metrics(images[0], self.mask_path)

        # 2. RAPOR METNİNİ OLUŞTURMA (HTML FORMATI)
        # ScrollArea içerisinde düzgün görünmesi için HTML etiketleri kullanılır
        ml_rapor_metni = (
            f"<b style='color:#2196F3;'>ML TABANLI ANALİZ RAPORU</b><br>"
            f"{'-'*35}<br>"
            f"<b>Görsel:</b> 1/{len(images)}<br>"
            f"<b>Dice Skoru:</b> %{dice*100:.2f}<br>"
            f"<b>Precision:</b> %{prec*100:.2f}<br>"
            f"<b>Recall:</b> %{rec*100:.2f}<br>"
            f"<b>F1-Score:</b> %{f1*100:.2f}<br>"
            f"<b>IoU:</b> {iou:.4f}<br><br>"
            f"<b>ML Model Detayları:</b><br>{rapor.replace('\n', '<br>')}"
        )

        # 3. ARAYÜZÜ GÜNCELLE
        # ScrollArea içindeki label'a metni basar
        self.label_ml_result.setText(ml_rapor_metni)
        
        # Ok butonlarını duruma göre gösterir
        if len(images) > 1:
            self.btn_ml_prev.show()
            self.btn_ml_next.show()

        print("[ML Raporu Tamamlandı]")

    def on_hybrid_done(self, images, rapor):
        self.hybrid_images = images
        self.hybrid_idx    = 0
        
        # Hibrit sonucun (Altın Oran) ilk görselini göster
        self.show_image(self.ui.label_6, images[0])
        
        # 1. METRİK HESAPLAMA
        # Hibrit maske (images[0]) ile gerçek maskeyi karşılaştır
        iou, dice, prec, rec, f1 = self.calculate_all_metrics(images[0], self.mask_path)

        # 2. HİBRİT TASARIMLI RAPOR METNİ (HTML)
        # Fizik yeşil, ML maviydi; Hibrit için Mor/Eflatun tonu (Altın Oran vurgusu)
        hybrid_rapor_metni = (
            f"<b style='color:#00E5FF;'>HİBRİT ANALİZ RAPORU</b><br>"
            f"<b>Görsel:</b> 1/{len(images)}<br>"
            f"{'-'*35}<br>"
            f"<b>Dice Skoru:</b> %{dice*100:.2f}<br>"
            f"<b>Precision:</b> %{prec*100:.2f}<br>"
            f"<b>Recall:</b> %{rec*100:.2f}<br>"
            f"<b>F1-Score:</b> %{f1*100:.2f}<br>"
            f"<b>IoU Değeri:</b> {iou:.4f}<br><br>"
            f"<b>Hibrit Model Detayları:</b><br>{rapor.replace('\n', '<br>')}"
        )

        # 3. ARAYÜZÜ GÜNCELLE
        # ScrollArea içindeki label'a aktar
        self.label_hybrid_result.setText(hybrid_rapor_metni)
        
        # Navigasyon butonlarını göster (varsa)
        if len(images) > 1:
            self.btn_hyb_prev.show()
            self.btn_hyb_next.show()

        print("[Hibrit Raporu Tamamlandı]")

    def on_error(self, msg):
        QMessageBox.critical(self, "Hata", f"Analiz sırasında hata oluştu:\n{msg}")

    # ── Görsel Geçiş ─────────────────────────────────────────
    def navigate_image(self, mode, direction):
        if mode == "physics" and self.physics_images:
            self.physics_idx = (self.physics_idx + direction) % len(self.physics_images)
            self.show_image(self.ui.label_4, self.physics_images[self.physics_idx])
        elif mode == "ml" and self.ml_images:
            self.ml_idx = (self.ml_idx + direction) % len(self.ml_images)
            self.show_image(self.ui.label_5, self.ml_images[self.ml_idx])
            self.label_ml_result.setPlainText(
                f"Görsel {self.ml_idx+1}/{len(self.ml_images)}\n"
                + self.label_ml_result.toPlainText().split("\n", 1)[-1]
            )
        elif mode == "hybrid" and self.hybrid_images:
            self.hybrid_idx = (self.hybrid_idx + direction) % len(self.hybrid_images)
            self.show_image(self.ui.label_6, self.hybrid_images[self.hybrid_idx])
            self.label_hybrid_result.setPlainText(
                f"Görsel {self.hybrid_idx+1}/{len(self.hybrid_images)}\n"
                + self.label_hybrid_result.toPlainText().split("\n", 1)[-1]
            )

    
    # ── Pencere Yeniden Boyutlandırma ────────────────────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        h = self.height()

        box_w = int(w * 0.35)
        box_h = int(h * 0.30)

        self.ui.groupBox.setGeometry(int(w*0.05), 40, box_w, box_h)
        self.ui.groupBox_2.setGeometry(int(w*0.55), 40, box_w, box_h)
        self.ui.label.setGeometry(10, 20, box_w-20, box_h-30)
        self.ui.label_2.setGeometry(10, 20, box_w-20, box_h-30)
        self.ui.pushButton.setGeometry(int(w*0.05), 40+box_h+10, box_w, 30)
        self.ui.pushButton_2.setGeometry(int(w*0.55), 40+box_h+10, box_w, 30)

        res_y  = 40 + box_h + 60
        res_w  = int(w * 0.28)
        res_h  = int(h * 0.30)

        self.ui.groupBox_3.setGeometry(int(w*0.02), res_y, res_w, res_h)
        self.ui.groupBox_4.setGeometry(int(w*0.36), res_y, res_w, res_h)
        self.ui.groupBox_5.setGeometry(int(w*0.70), res_y, res_w, res_h)
        self.ui.label_4.setGeometry(10, 20, res_w-20, res_h-30)
        self.ui.label_5.setGeometry(10, 20, res_w-20, res_h-30)
        self.ui.label_6.setGeometry(10, 20, res_w-20, res_h-30)

        # --- SONUÇ ALANLARI (YÜKSEKLİK ARTIRILDI) ---
        # ScrollArea olan bu alanların boyu 160 piksele çıkarıldı
        result_h = 160 
        result_y = res_y + res_h + 5
    
        # ScrollArea kullanıyorsan objelerin ismi scroll_phy vb. olabilir, 
        # senin kodundaki isimlendirmeye göre (self.label_..._result) güncelledim:
        self.label_physics_result.setGeometry(int(w*0.02), result_y, res_w, result_h)
        self.label_ml_result.setGeometry(int(w*0.36),      result_y, res_w, result_h)
        self.label_hybrid_result.setGeometry(int(w*0.70),  result_y, res_w, result_h)

        # --- OK BUTONLARI (KONUMU AŞAĞI KAYDIRILDI) ---
        arrow_y  = result_y + result_h + 8 # Boşluk payı eklendi
        arrow_w  = 35
        arrow_h  = 28

        self.btn_phy_prev.setGeometry(int(w*0.02),                    arrow_y, arrow_w, arrow_h)
        self.btn_phy_next.setGeometry(int(w*0.02)+res_w-arrow_w,     arrow_y, arrow_w, arrow_h)

        self.btn_ml_prev.setGeometry(int(w*0.36),                     arrow_y, arrow_w, arrow_h)
        self.btn_ml_next.setGeometry(int(w*0.36)+res_w-arrow_w,      arrow_y, arrow_w, arrow_h)

        self.btn_hyb_prev.setGeometry(int(w*0.70),                    arrow_y, arrow_w, arrow_h)
        self.btn_hyb_next.setGeometry(int(w*0.70)+res_w-arrow_w,     arrow_y, arrow_w, arrow_h)

        # --- ANA ANALİZ BUTONLARI ---
        btn_y = arrow_y + arrow_h + 8 # Okların altına yerleştirildi
        self.ui.pushButton_3.setGeometry(int(w*0.02), btn_y, res_w, 30)
        self.ui.pushButton_4.setGeometry(int(w*0.36), btn_y, res_w, 30)
        self.ui.pushButton_5.setGeometry(int(w*0.70), btn_y, res_w, 30)

        # Görselleri Yeniden Çiz
        if self.xray_path:
           self.show_image(self.ui.label, QPixmap(self.xray_path))
        if self.mask_path:
            self.show_image(self.ui.label_2, QPixmap(self.mask_path))
        if self.physics_images:
            self.show_image(self.ui.label_4, self.physics_images[self.physics_idx])
        if self.ml_images:
            self.show_image(self.ui.label_5, self.ml_images[self.ml_idx])
        if self.hybrid_images:
            self.show_image(self.ui.label_6, self.hybrid_images[self.hybrid_idx])


if __name__ == "__main__":
    app    = QApplication(sys.argv)
    window = DentalApp()
    window.show()
    sys.exit(app.exec_())
