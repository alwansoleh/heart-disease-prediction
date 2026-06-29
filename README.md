# Heart Disease Prediction — Machine Learning Project

Proyek ini dikembangkan sebagai tugas besar mata kuliah Pembelajaran Mesin.  
Tujuan utama adalah memprediksi risiko penyakit jantung pada pasien menggunakan tiga algoritma klasifikasi Machine Learning.

---

## Identitas

| Keterangan | Detail |
|---|---|
| Nama | Nur Tsalits Alwan Mubarok |
| NIM  | 103132400039 |
| Mata Kuliah | Pembelajaran Mesin |

---

## Deskripsi Permasalahan

Penyakit jantung merupakan salah satu penyebab kematian tertinggi di dunia. Deteksi dini sangat penting untuk mencegah komplikasi lebih lanjut. Proyek ini bertujuan mengembangkan model Machine Learning yang mampu memprediksi apakah seorang pasien memiliki risiko penyakit jantung berdasarkan data klinis seperti usia, tekanan darah, kadar kolesterol, dan hasil EKG.

**Jenis masalah:** Binary Classification  
**Label target:** `0` = Tidak ada penyakit jantung | `1` = Ada penyakit jantung

---

## Dataset

| Informasi | Detail |
|---|---|
| **Nama** | Heart Disease UCI (Cleveland) |
| **Sumber** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) |
| **Jumlah Data** | 303 sampel |
| **Jumlah Fitur** | 13 fitur |
| **Format** | CSV |

### Deskripsi Fitur

| Fitur | Deskripsi |
|---|---|
| `age` | Umur pasien (tahun) |
| `sex` | Jenis kelamin (1=Laki-laki, 0=Perempuan) |
| `cp` | Tipe nyeri dada (0–3) |
| `trestbps` | Tekanan darah istirahat (mm Hg) |
| `chol` | Kolesterol serum (mg/dl) |
| `fbs` | Gula darah puasa > 120 mg/dl (1=Ya, 0=Tidak) |
| `restecg` | Hasil EKG istirahat (0–2) |
| `thalach` | Detak jantung maksimum yang dicapai |
| `exang` | Angina akibat olahraga (1=Ya, 0=Tidak) |
| `oldpeak` | Depresi segmen ST akibat olahraga |
| `slope` | Kemiringan segmen ST puncak (0–2) |
| `ca` | Jumlah pembuluh darah utama (0–3) |
| `thal` | Hasil tes thallium (1=Normal, 2=Fixed Defect, 3=Reversible Defect) |

---

## Tahapan Preprocessing

1. **Handling Missing Values** — Nilai yang hilang (ditandai `?`) diisi dengan nilai median kolom tersebut.
2. **Binarisasi Target** — Nilai target asli (0–4) dikonversi menjadi biner: nilai > 0 menjadi `1` (ada penyakit), sisanya `0`.
3. **Train-Test Split** — Data dibagi 80% training dan 20% testing dengan stratified sampling.
4. **Standarisasi Fitur** — Semua fitur dinormalisasi menggunakan `StandardScaler` agar tidak ada fitur yang mendominasi karena perbedaan skala.

---

## Metode yang Digunakan

Tiga model klasifikasi dibandingkan dalam proyek ini:

| Model | Konfigurasi |
|---|---|
| **Support Vector Machine (SVM)** | Kernel RBF, `probability=True` |
| **K-Nearest Neighbors (KNN)** | `n_neighbors=7` |
| **Random Forest** | `n_estimators=100`, `random_state=42` |

Evaluasi dilakukan menggunakan:
- Accuracy
- ROC-AUC Score
- 5-Fold Cross Validation
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)

---

## Struktur Repository

```
heart-disease-prediction/
├── data/
│   └── processed.cleveland.data        # Dataset Cleveland Heart Disease
├── source/
│   ├── train.py                         # Pipeline utama (EDA, preprocessing, training, evaluasi)
│   └── predict.py                       # Prediksi data baru
├── models/                              # Dibuat otomatis setelah menjalankan train.py
│   ├── SVM.pkl
│   ├── KNN.pkl
│   ├── Random_Forest.pkl
│   └── scaler.pkl
├── outputs/                             # Dibuat otomatis setelah menjalankan train.py
│   ├── distribusi_target.png
│   ├── heatmap_korelasi.png
│   ├── distribusi_umur.png
│   ├── cm_SVM.png
│   ├── cm_KNN.png
│   ├── cm_Random_Forest.png
│   ├── roc_curve.png
│   ├── perbandingan_model.png 
│   └── hasil_evaluasi.csv
├── notebook_heart_disease.ipynb         # Eksplorasi interaktif (EDA + training)
├── requirements.txt
└── README.md
```

---

## Cara Menjalankan Program

### 1. Clone Repository

```bash
git clone https://github.com/alwansoleh/heart-disease-prediction.git
cd heart-disease-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Jalankan Training & Evaluasi

```bash
python source/train.py
```

### 4. Prediksi Data Baru

```bash
python source/predict.py
```

### 5. Eksplorasi via Notebook (Opsional)

Buka `notebook_heart_disease.ipynb` menggunakan Jupyter Notebook atau JupyterLab untuk melihat eksplorasi data dan training secara interaktif.

```bash
jupyter notebook notebook_heart_disease.ipynb
```

---

## Hasil Eksperimen dan Evaluasi

Hasil berikut diperoleh dari eksekusi `train.py` pada dataset Cleveland (233 sampel setelah cleaning, split 80:20).

| Model | Accuracy | ROC-AUC | CV Accuracy (5-Fold) |
|---|---|---|---|
| **SVM** | **0.9362** | **0.9722** | 0.8605 |
| KNN | 0.8723 | 0.9269 | 0.8337 |
| Random Forest | 0.9362 | 0.9667 | 0.8657 |

**Model terbaik berdasarkan ROC-AUC: SVM (0.9722)**

**Visualisasi yang dihasilkan:**
- Distribusi kelas target
- Heatmap korelasi antar fitur
- Distribusi umur berdasarkan target
- Confusion Matrix per model
- ROC Curve perbandingan ketiga model
- Bar chart perbandingan performa model

---

## Kesimpulan

1. Ketiga model berhasil mengklasifikasikan risiko penyakit jantung dengan performa yang baik.
2. **SVM** menunjukkan performa terbaik berdasarkan ROC-AUC (0.9722), menandakan kemampuannya memisahkan kelas secara optimal pada ruang fitur yang telah dinormalisasi.
3. **Random Forest** memiliki CV Accuracy tertinggi (0.8657), menunjukkan generalisasi yang stabil lintas fold.
4. Fitur yang paling berpengaruh dalam prediksi antara lain: `thal`, `ca`, `cp`, dan `oldpeak`.
5. Proyek ini menunjukkan bahwa Machine Learning dapat menjadi alat bantu yang berguna dalam deteksi dini penyakit jantung berdasarkan data klinis dasar.

---

## Referensi

- Detrano, R., et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* American Journal of Cardiology.
- UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/datasets/Heart+Disease
- Scikit-learn Documentation: https://scikit-learn.org
