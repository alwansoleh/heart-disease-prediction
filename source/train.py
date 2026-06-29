"""
Heart Disease Prediction - Main Training Script
================================================
Membandingkan 3 model Machine Learning:
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Random Forest Classifier

Dataset: Heart Disease UCI (Cleveland)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Load dataset dari file CSV."""
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope",
        "ca", "thal", "target"
    ]
    df = pd.read_csv(path, names=column_names, na_values="?")
    print(f"[INFO] Dataset loaded: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    Tahapan preprocessing:
    1. Handle missing values
    2. Binarisasi target (0 = tidak sakit, 1 = sakit)
    3. Feature-target split
    4. Train-test split
    5. Standardisasi fitur
    """
    # Handle missing values dengan median
    df = df.fillna(df.median(numeric_only=True))

    # Binarisasi target: nilai > 0 → 1 (penyakit jantung)
    df["target"] = (df["target"] > 0).astype(int)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"[INFO] Train: {X_train_sc.shape[0]} | Test: {X_test_sc.shape[0]}")
    print(f"[INFO] Distribusi target → 0: {(y==0).sum()} | 1: {(y==1).sum()}")

    return X_train_sc, X_test_sc, y_train, y_test, scaler, X.columns.tolist()


# ─────────────────────────────────────────────
# 3. EDA (EXPLORATORY DATA ANALYSIS)
# ─────────────────────────────────────────────

def eda(df: pd.DataFrame, output_dir: str):
    """Simpan visualisasi EDA ke folder outputs."""
    os.makedirs(output_dir, exist_ok=True)
    df_eda = df.copy()
    df_eda["target"] = (df_eda["target"] > 0).astype(int)

    # (a) Distribusi target
    fig, ax = plt.subplots(figsize=(5, 4))
    df_eda["target"].value_counts().plot(kind="bar", color=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_title("Distribusi Kelas Target")
    ax.set_xlabel("0 = Sehat  |  1 = Penyakit Jantung")
    ax.set_ylabel("Jumlah")
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/distribusi_target.png", dpi=120)
    plt.close()

    # (b) Heatmap korelasi
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df_eda.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Heatmap Korelasi Fitur")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/heatmap_korelasi.png", dpi=120)
    plt.close()

    # (c) Distribusi umur berdasarkan target
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color in zip([0, 1], ["#2ecc71", "#e74c3c"]):
        df_eda[df_eda["target"] == label]["age"].hist(
            bins=20, alpha=0.6, color=color, label=f"Target {label}", ax=ax
        )
    ax.set_title("Distribusi Umur berdasarkan Target")
    ax.set_xlabel("Umur")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/distribusi_umur.png", dpi=120)
    plt.close()

    print("[INFO] EDA plots tersimpan di folder outputs/")


# ─────────────────────────────────────────────
# 4. TRAINING & EVALUASI MODEL
# ─────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test, output_dir: str):
    """Latih & evaluasi 3 model, simpan hasil perbandingan."""

    models = {
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results   = {}
    best_model_name = None
    best_auc  = 0

    for name, model in models.items():
        print(f"\n{'─'*40}")
        print(f"  Model: {name}")
        print(f"{'─'*40}")

        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_prob)
        cv   = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()

        results[name] = {"Accuracy": acc, "ROC-AUC": auc, "CV Accuracy": cv}

        print(f"  Accuracy  : {acc:.4f}")
        print(f"  ROC-AUC   : {auc:.4f}")
        print(f"  CV (5-fold): {cv:.4f}")
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred, target_names=["Sehat", "Sakit"]))

        # Confusion Matrix
        _plot_confusion_matrix(y_test, y_pred, name, output_dir)

        # Simpan model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, f"models/{name.replace(' ', '_')}.pkl")

        if auc > best_auc:
            best_auc = auc
            best_model_name = name

    # Tabel perbandingan
    df_results = pd.DataFrame(results).T.round(4)
    print(f"\n{'═'*50}")
    print("  PERBANDINGAN MODEL")
    print(f"{'═'*50}")
    print(df_results.to_string())
    print(f"\n  🏆 Model Terbaik: {best_model_name} (ROC-AUC: {best_auc:.4f})")

    df_results.to_csv(f"{output_dir}/hasil_evaluasi.csv")

    # ROC Curve
    _plot_roc_curve(X_test, y_test, models, output_dir)

    # Bar chart perbandingan
    _plot_comparison(df_results, output_dir)

    return results, best_model_name


def _plot_confusion_matrix(y_test, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Sehat", "Sakit"],
                yticklabels=["Sehat", "Sakit"], ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_ylabel("Aktual")
    ax.set_xlabel("Prediksi")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cm_{model_name.replace(' ', '_')}.png", dpi=120)
    plt.close()


def _plot_roc_curve(X_test, y_test, models, output_dir):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#3498db", "#e67e22", "#2ecc71"]
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC Curve — Perbandingan Model")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curve.png", dpi=120)
    plt.close()


def _plot_comparison(df_results, output_dir):
    df_results[["Accuracy", "ROC-AUC", "CV Accuracy"]].plot(
        kind="bar", figsize=(8, 5), colormap="Set2", edgecolor="black"
    )
    plt.title("Perbandingan Performa Model")
    plt.ylabel("Score")
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/perbandingan_model.png", dpi=120)
    plt.close()


# ─────────────────────────────────────────────
# 5. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────

def plot_feature_importance(model, feature_names, output_dir):
    """Plot feature importance dari Random Forest."""
    importance = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importance})
    fi_df = fi_df.sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    fi_df.plot(kind="barh", x="Feature", y="Importance",
               color="#3498db", edgecolor="black", ax=ax, legend=False)
    ax.set_title("Feature Importance — Random Forest")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png", dpi=120)
    plt.close()
    print("[INFO] Feature importance plot tersimpan.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH  = "data/processed.cleveland.data"
    OUTPUT_DIR = "outputs"

    print("=" * 50)
    print("  HEART DISEASE PREDICTION")
    print("=" * 50)

    df = load_data(DATA_PATH)

    print("\n[STEP 1] EDA ...")
    eda(df, OUTPUT_DIR)

    print("\n[STEP 2] Preprocessing ...")
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess(df)

    print("\n[STEP 3] Training & Evaluasi ...")
    results, best = train_and_evaluate(X_train, X_test, y_train, y_test, OUTPUT_DIR)

    print("\n[STEP 4] Feature Importance ...")
    rf_model = joblib.load("models/Random_Forest.pkl")
    plot_feature_importance(rf_model, feature_names, OUTPUT_DIR)

    # Simpan scaler
    joblib.dump(scaler, "models/scaler.pkl")

    print("\n✅ Selesai! Semua output tersimpan di folder outputs/ dan models/")
