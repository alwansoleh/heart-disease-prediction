"""
predict.py
==========
Script untuk prediksi menggunakan model yang sudah dilatih.

Contoh penggunaan:
    python src/predict.py

Input berupa data pasien, output berupa prediksi risiko penyakit jantung.
"""

import numpy as np
import joblib

FEATURE_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

FEATURE_DESC = {
    "age"     : "Umur (tahun)",
    "sex"     : "Jenis Kelamin (1=Laki-laki, 0=Perempuan)",
    "cp"      : "Tipe Nyeri Dada (0-3)",
    "trestbps": "Tekanan Darah Istirahat (mm Hg)",
    "chol"    : "Kolesterol Serum (mg/dl)",
    "fbs"     : "Gula Darah Puasa > 120 mg/dl (1=Ya, 0=Tidak)",
    "restecg" : "Hasil EKG Istirahat (0-2)",
    "thalach" : "Detak Jantung Maksimum",
    "exang"   : "Angina karena Olahraga (1=Ya, 0=Tidak)",
    "oldpeak" : "Depresi ST (0-6.2)",
    "slope"   : "Kemiringan ST (0-2)",
    "ca"      : "Jumlah Pembuluh Utama (0-3)",
    "thal"    : "Thal (1=Normal, 2=Fixed Defect, 3=Reversible Defect)",
}


def predict(model_name: str = "Random_Forest"):
    """
    Prediksi satu sampel data pasien menggunakan model yang dipilih.

    Args:
        model_name: Salah satu dari 'SVM', 'KNN', 'Random_Forest'
    """
    print("=" * 50)
    print("  PREDIKSI PENYAKIT JANTUNG")
    print("=" * 50)
    print(f"\nMenggunakan model: {model_name}\n")

    # Load model dan scaler
    model  = joblib.load(f"models/{model_name}.pkl")
    scaler = joblib.load("models/scaler.pkl")

    # Contoh data pasien (bisa diubah sesuai kebutuhan)
    sample = {
        "age"     : 63,
        "sex"     : 1,
        "cp"      : 3,
        "trestbps": 145,
        "chol"    : 233,
        "fbs"     : 1,
        "restecg" : 0,
        "thalach" : 150,
        "exang"   : 0,
        "oldpeak" : 2.3,
        "slope"   : 0,
        "ca"      : 0,
        "thal"    : 1,
    }

    print("Data Pasien:")
    print("-" * 35)
    for key, val in sample.items():
        print(f"  {FEATURE_DESC[key]:<40}: {val}")

    X = np.array([[sample[f] for f in FEATURE_NAMES]])
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0]

    print("\n" + "=" * 50)
    if pred == 1:
        print(f"  ⚠️  Hasil: RISIKO PENYAKIT JANTUNG TERDETEKSI")
    else:
        print(f"  ✅  Hasil: TIDAK TERDETEKSI PENYAKIT JANTUNG")
    print(f"  Probabilitas Sehat  : {prob[0]*100:.1f}%")
    print(f"  Probabilitas Sakit  : {prob[1]*100:.1f}%")
    print("=" * 50)

    return pred, prob


if __name__ == "__main__":
    predict(model_name="Random_Forest")
