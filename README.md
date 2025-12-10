# 🧠 Implementasi Algoritma Gaussian Mixture Model (GMM) untuk Deteksi Pola Tidur Berbasis Sinyal EEG

Proyek ini merupakan aplikasi berbasis **Streamlit** yang mengimplementasikan **Gaussian Mixture Model (GMM)** untuk melakukan *unsupervised clustering* pada sinyal **Electroencephalography (EEG)** dalam rangka mendeteksi dan menganalisis tahapan tidur manusia (Wake, NREM, dan REM).

Aplikasi ini dirancang untuk keperluan akademik dan penelitian, khususnya dalam bidang **signal processing**, **machine learning**, dan **sleep stage analysis**.

---

## 📌 Fitur Utama

- Unggah dan pratinjau dataset EEG dalam format CSV  
- Statistik deskriptif sinyal EEG  
- Pemilihan fitur/channel EEG dan EOG secara interaktif  
- Visualisasi sinyal EEG berdasarkan rentang waktu atau indeks sampel  
- Implementasi **Gaussian Mixture Model (GMM)** dari nol (tanpa `scikit-learn`)  
- Standarisasi fitur otomatis  
- Visualisasi kurva konvergensi log-likelihood  
- Visualisasi klaster 2D (dengan PCA jika dimensi fitur > 2)  
- Ekstraksi fitur domain frekuensi menggunakan Welch Power Spectral Density  
- Estimasi probabilitas tahapan tidur:
  - Wake (W)
  - N1
  - N2
  - N3
  - REM (R)  
- Perhitungan **Bayesian Information Criterion (BIC)**  
- Deteksi indikasi apnea berbasis probabilitas REM dan N1  

---

## 🗂️ Struktur Proyek
├── app.py                  
├── extrait_wSleepPage01.csv    
├── README.md                  
└── requirements.txt            


---

## ⚙️ Teknologi dan Library

- Python 3.8 atau lebih baru
- Streamlit
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy

---

## 📊 Dataset

Aplikasi menerima dataset EEG dalam format **CSV** dengan karakteristik:

- Kolom sinyal EEG/EOG (misal: `EEG Fp1`, `EEG Fp2`, dll.)
- Kolom waktu (opsional), seperti:
  - `Time (s)`
  - `SS`
- Nilai numerik dapat menggunakan:
  - Titik (`.`) atau
  - Koma (`,`) sebagai pemisah desimal  

Dataset :
extrait_wSleepPage01.csv


---

## 🚀 Cara Menjalankan Aplikasi

### 1. Clone Repository
```bash
git clone https://github.com/username/nama-repository.git
cd nama-repository

### 2. Install Dependensi
pip install -r requirements.txt

### 3. Jalankan Aplikasi Streamlit
streamlit run app.py

## 🧪 Alur Kerja Aplikasi

1. Unggah file CSV sinyal EEG  
2. Pilih channel EEG/EOG yang akan dianalisis  
3. Visualisasikan sinyal EEG  
4. Jalankan pelatihan model Gaussian Mixture Model (GMM)  
5. Amati hasil analisis:
   - Label klaster
   - Plot klaster 2D
   - Kurva konvergensi log-likelihood
   - Probabilitas tiap tahapan tidur
   - Nilai Bayesian Information Criterion (BIC)


