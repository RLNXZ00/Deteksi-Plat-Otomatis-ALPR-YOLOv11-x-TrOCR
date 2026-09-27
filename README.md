<div align="center">

# 🚗🇮🇩 ALPR Indonesia: End-to-End Automatic License Plate Recognition System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/YOLOv11s-Ultralytics-00FFFF.svg?logo=yolo&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![HuggingFace TrOCR](https://img.shields.io/badge/TrOCR-Base-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/microsoft/trocr-base-printed)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8.svg?logo=opencv&logoColor=white)](https://opencv.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20Kaggle-success.svg)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Sistem Pengenalan Plat Nomor Kendaraan Bermotor (TNKB) Indonesia Berbasis Deep Learning 2-Stage Pipeline (YOLOv11s + TrOCR Base) dengan Engine Validasi & Disambiguasi Karakter Posisi-Kritis serta GUI Kamera Real-Time.**

[Fitur Utama](#-fitur-utama) • [Arsitektur Pipeline](#-arsitektur-pipeline) • [Hasil Evaluasi](#-hasil-evaluasi--benchmark) • [Struktur Repositori](#-struktur-repositori) • [Panduan Instalasi](#-panduan-instalasi--setup) • [Cara Penggunaan](#-cara-penggunaan) • [Logika Post-Processing](#-logika-post-processing-tnkb)

---

</div>

## 📌 Sekilas Proyek

Sistem **Automatic License Plate Recognition (ALPR)** atau **Automatic Number Plate Recognition (ANPR)** ini dirancang khusus untuk menangani karakteristik unik dan kompleksitas **Tanda Nomor Kendaraan Bermotor (TNKB) di Indonesia**, seperti:
- Format TNKB dua baris (baris atas: nomor polisi, baris bawah: masa berlaku pajak e.g. `05.28`).
- Ambiguitas visual karakter yang sering tertukar pada plat cetak/timbul (`O` vs `0`, `I`/`1` vs `L`, `B` vs `8`, `Z` vs `2`, `S` vs `5`).
- Variasi pencahayaan, sudut miring (*perspective tilt*), resolusi rendah, dan jarak kamera.

Sistem memadukan ketangguhan **YOLOv11s** (*state-of-the-art detector*) untuk lokalisasi plat, **TrOCR Base** (*Vision Transformer encoder-decoder*) untuk pembacaan teks, serta **Exhaustive Minimum-Changes Engine** untuk memvalidasi dan mengoreksi karakter sesuai format resmi Korlantas RI.

---

## ✨ Fitur Utama

- 🎯 **2-Stage State-of-the-Art Architecture**:
  - **Stage 1 (Deteksi)**: YOLOv11s dilatih khusus pada dataset plat nomor Indonesia (**mAP@50: 95.11%**, Precision: 87.56%, Recall: 90.71%).
  - **Stage 2 (Rekognisi)**: TrOCR Base (*Vision Transformer*) membaca teks plat tanpa memerlukan segmentasi karakter per-huruf (*end-to-end sequence recognition*).
- 📐 **Asymmetric Smart Padding (8% Horizontal, 5% Vertical)**: Mencegah huruf awal (e.g. `B`, `D`, `AB`) dan huruf akhir seri terpotong saat proses crop bounding box.
- 🔄 **Two-Line Fallback Mechanism**: Jika OCR membaca baris bawah (angka bulan/tahun pajak seperti `1219`), pipeline otomatis melakukan fallback crop 75% area atas plat untuk membaca nomor polisi utama.
- 🧠 **Exhaustive Min-Changes Disambiguation Engine**: Algoritma cerdas yang mengevaluasi seluruh kemungkinan segmentasi `[Wilayah] [Angka] [Seri]` dan memilih kandidat dengan perubahan karakter minimum yang valid secara aturan TNKB (lulus 14/14 skenario unit test).
- 🎥 **Simulasi Kamera Laptop Real-Time (30 FPS Lancar di CPU)**:
  - *Asynchronous Multithreading*: Inferensi OCR berjalan di background thread sehingga preview webcam tidak freeze/lag.
  - *CPU-Optimized*: Menggunakan *Greedy Search Decoding* (`num_beams=1`), memangkas latensi OCR dari 10.3s menjadi ~1.5s pada laptop standar (Intel Core i5 tanpa GPU diskrit).
- 🪟 **Dual GUI & Pop-up System**:
  - **Jendela Pop-up Native**: Menampilkan foto crop plat resolusi tinggi, plat nomor dengan spasi rapi, badge validasi hijau/oranye, confidence, dan latensi komputasi.
  - **Futuristic Glassmorphism HUD**: Overlay neon pada frame kamera dengan bounding box dinamis dan ringkasan deteksi.
- 📊 **Pencatatan Otomatis ke CSV**: Hasil deteksi valid otomatis direkam ke [riwayat_deteksi.csv](riwayat_deteksi.csv) lengkap dengan timestamp dan confidence score.

---

## 🏗 Arsitektur Pipeline

```mermaid
flowchart TD
    A[Input: Frame Kamera / Citra Kendaraan] --> B[YOLOv11s Plate Detector]
    B -->|Bounding Box + Conf >= 0.20| C[Asymmetric Crop & Padding\n+8% Lebar, +5% Tinggi]
    C --> D[TrOCR Base OCR Recognizer]
    D --> E{Apakah Hasil Valid TNKB\natau Hanya Angka Pajak?}
    E -->|Angka Pajak / Tidak Valid| F[Fallback: Crop 75% Area Atas]
    F --> G[TrOCR Secondary Inference]
    E -->|Valid / Karakter Lengkap| H[Post-Processing & Disambiguation Engine]
    G --> H
    H --> I[Exhaustive Minimum-Changes Search\nKoreksi Posisi Huruf/Angka Sesuai Pola TNKB]
    I --> J{Validasi Regex TNKB\n^[A-Z]{1,2} [0-9]{1,4} [A-Z]{1,3}$}
    J -->|Valid| K[Format Output: B 1234 XYZ]
    J -->|Non-Standar| L[Format Output Non-Standar]
    K --> M[GUI Pop-up Window + Frame HUD + CSV Logger]
    L --> M
```

---

## 📊 Hasil Evaluasi & Benchmark

Pengujian dilakukan pada test set independen berisi 372 gambar plat nomor Indonesia di berbagai kondisi:

### 1. Performa Detektor Plat (YOLOv11s Exp-C)

| Metrik | Nilai | Keterangan |
| :--- | :---: | :--- |
| **Precision** | **87.56%** | Akurasi bounding box plat yang terdeteksi |
| **Recall** | **90.71%** | Kemampuan menemukan plat di seluruh skenario |
| **mAP@50** | **95.11%** | Rata-rata presisi pada IoU threshold 0.50 |
| **mAP@50-95** | **60.20%** | Rata-rata presisi pada berbagai tingkat overlap ketat |

### 2. Performa End-to-End (E2E Pipeline)

| Metrik Evaluasi | Nilai | Penjelasan |
| :--- | :---: | :--- |
| **Conditional OCR Accuracy** | **77.97%** (230/295) | Akurasi pembacaan tepat 100% pada plat yang terdeteksi |
| **E2E Exact Match** | **61.83%** (230/372) | Akurasi seluruh test set termasuk gambar tanpa plat |
| **Character Error Rate (CER)** | **0.1115** | Rata-rata kesalahan per karakter (sangat rendah) |

---

## 📁 Struktur Repositori

```text
├── Python Notebook/
│   ├── Master_Pipeline_E2E.ipynb        # Notebook evaluasi E2E Pipeline lengkap
│   ├── YOLOv11_Plate_Detector.ipynb      # Notebook pelatihan detektor YOLOv11
│   └── TrOCR_Text_Extraction.ipynb       # Notebook pelatihan recognizer TrOCR
├── simulasi_kamera_laptop.py             # Script simulasi kamera webcam live + GUI Pop-up
├── jalankan_simulasi_kamera.bat         # Launcher Windows 1-klik untuk simulasi kamera
├── requirements.txt                     # Daftar dependensi library Python
├── riwayat_deteksi.csv                  # File log riwayat pembacaan plat nomor
│
├── Plate Detector Weight/
│   ├── model_registry.json              # Metadata bobot model detektor
│   ├── yolo_plate_exp_a_best.pt         # Bobot YOLOv11n (Nano)
│   ├── yolo_plate_exp_b_best.pt         # Bobot YOLOv11s (Small baseline)
│   └── yolo_plate_exp_c_best.pt         # Bobot YOLOv11s (Terbaik - mAP@50: 95.11%)
│
├── OCR Extraction Text Weight EXP 1/    # Bobot TrOCR Small
│   ├── model.safetensors
│   └── ...
├── OCR Extraction Text Weight EXP 2/    # Bobot TrOCR Base (Terbaik - 1.33 GB)
│   ├── model.safetensors
│   └── ...
│
├── Panduan Konsep dan Implementasi/     # Dokumentasi arsitektur dan teknis mendalam
│   ├── ARSITEKTUR_PIPELINE.md
│   ├── DETAIL_PROSEDURE.md
│   └── INTRODUCTION.md
│
└── hasil_tangkapan_kamera/              # Folder penyimpanan snapshot gambar kamera
```

---

## 💻 Panduan Instalasi & Setup

### 1. Clone Repositori
```bash
git clone https://github.com/username/alpr-indonesia.git
cd alpr-indonesia
```

### 2. Buat Lingkungan Virtual (Disarankan)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instal Dependensi
Pilih instalasi sesuai ketersediaan perangkat keras Anda:

* **Opsi A: Untuk Pengguna CPU (Laptop Standar tanpa GPU NVIDIA)**:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install ultralytics transformers opencv-python pillow
  ```

* **Opsi B: Untuk Pengguna GPU NVIDIA (CUDA)**:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  pip install ultralytics transformers opencv-python pillow
  ```

---

## 🚀 Cara Penggunaan

### 1. Menjalankan Simulasi Kamera Webcam Live
Pastikan webcam laptop terhubung dan tidak sedang digunakan aplikasi lain (Zoom, Teams, dsb.):

* **Cara 1 (Windows 1-Klik)**: Klik ganda file [jalankan_simulasi_kamera.bat](jalankan_simulasi_kamera.bat).
* **Cara 2 (Terminal)**:
  ```bash
  python simulasi_kamera_laptop.py
  ```

#### 🎮 Tombol Navigasi & Shortcut:
| Tombol | Fungsi |
| :---: | :--- |
| **`[P]`** | **Toggle Pop-up Window**: Menutup atau memunculkan kembali jendela pop-up hasil OCR. |
| **`[SPACE]`** | **Simpan Snapshot Ganda**: Menyimpan foto frame kamera + kartu pop-up OCR ke folder `screenshots/`. |
| **`[F]`** | **Putar / Balik Kamera**: Mengubah orientasi jika sensor webcam terbalik (Vertikal $\rightarrow$ 180° $\rightarrow$ Mirror $\rightarrow$ Normal). |
| **`[T]`** | **Toggle Scan**: Beralih antara mode *Auto-Scan* terus-menerus dan *Manual Scan*. |
| **`[Q]` / `[ESC]`** | **Keluar**: Menutup semua jendela aplikasi dan menyimpan log ke CSV. |

---

### 2. Menjalankan Evaluasi Batch pada Notebook
Buka [Python Notebook/Master_Pipeline_E2E.ipynb](Python%20Notebook/Master_Pipeline_E2E.ipynb) di Jupyter Lab, VS Code, atau upload ke Google Colab / Kaggle untuk mengevaluasi seluruh pipeline pada folder citra pengujian.

---

## 🔍 Logika Post-Processing TNKB

Aturan Tanda Nomor Kendaraan Bermotor (TNKB) di Indonesia memiliki struktur gramatikal yang kaku:
$$\underbrace{\text{[1--2 Huruf]}}_{\text{Kode Wilayah}} \quad \underbrace{\text{[1--4 Angka]}}_{\text{Nomor Polisi}} \quad \underbrace{\text{[1--3 Huruf]}}_{\text{Seri Akhir}}$$

### Matriks Disambiguasi Karakter:
| Karakter Asli | Jika Berada di Bagian Wilayah / Seri (Huruf) | Jika Berada di Bagian Nomor (Angka) |
| :---: | :---: | :---: |
| **`0`** (Nol) | Dikonversi ke **`O`** atau **`D`** | Tetap **`0`** |
| **`1`** (Satu) | Dikonversi ke **`I`** atau **`L`** | Tetap **`1`** |
| **`2`** (Dua) | Dikonversi ke **`Z`** | Tetap **`2`** |
| **`4`** (Empat) | Dikonversi ke **`A`** | Tetap **`4`** |
| **`5`** (Lima) | Dikonversi ke **`S`** | Tetap **`5`** |
| **`8`** (Delapan) | Dikonversi ke **`B`** | Tetap **`8`** |
| **`O`** (Huruf O) | Tetap **`O`** | Dikonversi ke **`0`** |
| **`I` / `L`** | Tetap **`I` / `L`** | Dikonversi ke **`1`** |
| **`B`** (Huruf B) | Tetap **`B`** | Dikonversi ke **`8`** |

Algoritma **Exhaustive Search** memeriksa semua kemungkinan panjang substring wilayah ($1 \le L_1 \le 2$), angka ($1 \le L_2 \le 4$), dan seri ($1 \le L_3 \le 3$), lalu memilih kandidat dengan nilai edit distance terkecil terhadap teks mentah TrOCR.

---

## 🛠 Kebutuhan Perangkat Keras (*System Requirements*)

- **Minimum (CPU Only)**:
  - Processor: Intel Core i5 Generasi ke-8 / AMD Ryzen 5 ke atas.
  - RAM: 8 GB.
  - Storage: 3 GB ruang kosong.
  - OS: Windows 10/11 atau Ubuntu 20.04+.
- **Rekomendasi (GPU)**:
  - GPU: NVIDIA GTX 1660 / RTX 3050 ke atas (VRAM $\ge$ 4 GB).
  - RAM: 16 GB.
  - CUDA: Version 11.8 atau 12.x.

---

## 📜 Lisensi & Atribusi

Proyek ini dirilis di bawah lisensi **MIT License**.

### Penghargaan & Referensi:
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) untuk model pendeteksi objek berkecepatan tinggi.
- [Microsoft TrOCR](https://github.com/microsoft/unilm/tree/master/trocr) & [HuggingFace Transformers](https://huggingface.co/) untuk arsitektur Vision Transformer OCR.
- Komunitas dataset plat nomor kendaraan bermotor Indonesia di Kaggle dan Roboflow.
