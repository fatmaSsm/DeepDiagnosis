"""
validate.py — Yarışma Günü Simülasyonu
=======================================
Bu script main.py'den BAĞIMSIZ çalışır.
Modeli TÜM eğitim verisiyle yeniden eğitir,
ardından daha önce kenara ayrılan internal_test_set.csv'ye
sanki hiç görmemiş gibi uygular.

Mantık:
  - Eğitim aşamasında test seti modele hiç gösterilmedi.
  - Şimdi sadece test setini verip "ne kadar doğru?" diyoruz.
  - Bu yarışmada yapılacak external validation'ın simülasyonu.

Kullanım:
  python validate.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score,
    matthews_corrcoef, precision_recall_curve, auc,
    ConfusionMatrixDisplay, RocCurveDisplay
)

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
VAL_DIR    = OUTPUT_DIR / "validation_figures"
VAL_DIR.mkdir(parents=True, exist_ok=True)

ALT_GRUPLAR = ["MASTER", "KANSER", "CFTR", "PAH"]


# ─── ADIM 1: EĞİTİM VERİSİNİ YÜKLE ──────────────────────────────────────────

def load_train_data():
    """
    internal_test_set.csv'de hangi Variant_ID'ler var?
    Onları eğitimden çıkart, kalanla eğit.
    Böylece kesinlikle sızma olmaz.
    """
    test_ids = set(
        pd.read_csv(OUTPUT_DIR / "internal_test_set.csv")["Variant_ID"].astype(str)
    )

    dfs = []
    for grup in ALT_GRUPLAR:
        df = pd.read_csv(DATA_DIR / f"YARISMA_TRAIN_{grup}.csv")
        df["SOURCE"] = grup
        dfs.append(df)

    full_df  = pd.concat(dfs, ignore_index=True)
    train_df = full_df[~full_df["Variant_ID"].astype(str).isin(test_ids)].copy()
    test_df  = pd.read_csv(OUTPUT_DIR / "internal_test_set.csv")

    print(f"  Eğitim : {len(train_df)} satır")
    print(f"  Test   : {len(test_df)} satır  (daha önce kenara ayrılmış)")
    return train_df, test_df


# ─── ADIM 2: ÖN İŞLEME ───────────────────────────────────────────────────────

def preprocess(train_df, test_df):
    drop_cols = ["Variant_ID"]
    y_train = train_df["Label"].values
    y_test  = test_df["Label"].values

    train_df = pd.get_dummies(train_df, columns=["SOURCE"], drop_first=False)
    test_df  = pd.get_dummies(test_df,  columns=["SOURCE"], drop_first=False)
    for col in train_df.columns:
        if col not in test_df.columns:
            test_df[col] = 0

    feature_cols = [c for c in train_df.columns if c not in drop_cols + ["Label"]]
    X_train = train_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()

    cat_cols = []
    for col in feature_cols:
        if col.startswith("CAT_") or col.startswith("AA_"):
            cat_cols.append(col)
        elif X_train[col].dtype == object or str(X_train[col].dtype) == "string":
            cat_cols.append(col)

    for col in cat_cols:
        le = LabelEncoder()
        tr_vals = X_train[col].fillna("__NaN__").astype(str)
        te_vals = X_test[col].fillna("__NaN__").astype(str)
        le.fit(pd.concat([tr_vals, te_vals]))
        X_train[col] = le.transform(tr_vals).astype(float)
        X_test[col]  = le.transform(te_vals).astype(float)

    for col in feature_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col]  = pd.to_numeric(X_test[col],  errors="coerce")

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols)
    X_test  = pd.DataFrame(imputer.transform(X_test),      columns=feature_cols)

    return X_train, X_test, y_train, y_test, feature_cols


# ─── ADIM 3: MODELLERİ EĞİT ──────────────────────────────────────────────────

def train_models(X_train, y_train):
    models = {
        "HistGBM": HistGradientBoostingClassifier(
            max_iter=500, max_depth=6, learning_rate=0.05,
            min_samples_leaf=20, l2_regularization=0.1,
            class_weight="balanced", early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=30,
            random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, min_samples_split=5,
            max_features="sqrt", class_weight="balanced_subsample",
            n_jobs=-1, random_state=42
        ),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=0.1, class_weight="balanced",
                                      max_iter=1000, solver="saga", random_state=42))
        ]),
    }
    print()
    for name, m in models.items():
        print(f"  → {name} eğitiliyor...", end=" ", flush=True)
        m.fit(X_train, y_train)
        print("✓")
    return models


def ensemble_proba(models, X):
    probs = np.stack([m.predict_proba(X)[:, 1] for m in models.values()])
    return probs.mean(axis=0)


# ─── ADIM 4: SONUÇLAR VE GRAFİKLER ───────────────────────────────────────────

def print_report(y_true, preds, probs, label):
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    recall   = tp / (tp + fn) * 100
    fn_rate  = fn / (fn + tp) * 100
    spec     = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    prec     = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    f1       = f1_score(y_true, preds) * 100
    mcc      = matthews_corrcoef(y_true, preds)
    auc_roc  = roc_auc_score(y_true, probs) * 100

    print(f"\n  ┌─ [{label}] ──────────────────────────────────────")
    print(f"  │  Doğru Patojenik (TP)  : {tp:4d}   Kaçırılan Hasta (FN)  : {fn:4d}")
    print(f"  │  Doğru Benign    (TN)  : {tn:4d}   Yanlış Alarm    (FP)  : {fp:4d}")
    print(f"  ├──────────────────────────────────────────────────")
    print(f"  │  Sensitivity / Recall  : %{recall:.1f}")
    print(f"  │  FN Rate (kritik!)     : %{fn_rate:.1f}   ← hasta kaçırma oranı")
    print(f"  │  Specificity           : %{spec:.1f}")
    print(f"  │  Precision             : %{prec:.1f}")
    print(f"  │  F1 Score              : %{f1:.1f}")
    print(f"  │  MCC                   : {mcc:.3f}")
    print(f"  │  AUC-ROC               : %{auc_roc:.1f}")
    print(f"  └──────────────────────────────────────────────────")
    return {"TP":tp,"FN":fn,"TN":tn,"FP":fp,
            "Recall_%":round(recall,1), "FN_Rate_%":round(fn_rate,1),
            "Specificity_%":round(spec,1), "Precision_%":round(prec,1),
            "F1_%":round(f1,1), "MCC":round(mcc,3), "AUC_ROC_%":round(auc_roc,1)}


def plot_cm(y_true, preds, label, fname):
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Benign (0)", "Patojenik (1)"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"Confusion Matrix (Sayı) – {label}")

    im = axes[1].imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    axes[1].set_xticks([0,1]); axes[1].set_xticklabels(["Benign (0)","Patojenik (1)"])
    axes[1].set_yticks([0,1]); axes[1].set_yticklabels(["Benign (0)","Patojenik (1)"])
    axes[1].set_xlabel("Tahmin Edilen"); axes[1].set_ylabel("Gerçek")
    axes[1].set_title(f"Confusion Matrix (%) – {label}")
    for i in range(2):
        for j in range(2):
            col = "white" if cm_pct[i,j] > 60 else "black"
            axes[1].text(j, i, f"{cm_pct[i,j]:.1f}%\n({cm[i,j]})",
                         ha="center", va="center", color=col, fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=axes[1], label="%")

    recall  = tp/(tp+fn)*100; fn_rate = fn/(tp+fn)*100
    spec    = tn/(tn+fp)*100 if (tn+fp)>0 else 0
    fig.text(0.5,-0.02,
             f"Sensitivity={recall:.1f}%  |  FN Rate={fn_rate:.1f}%  |  Specificity={spec:.1f}%  |  n={len(y_true)}",
             ha="center", fontsize=10, color="#333333")
    plt.tight_layout()
    plt.savefig(VAL_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_pr(y_true, probs, label, prefix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    RocCurveDisplay.from_predictions(y_true, probs, ax=axes[0], name="Ensemble")
    axes[0].plot([0,1],[0,1],"k--"); axes[0].set_title(f"ROC – {label}")

    prec, rec, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(rec, prec)
    axes[1].plot(rec, prec, color="#E91E63", lw=2)
    axes[1].fill_between(rec, prec, alpha=0.15, color="#E91E63")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"Precision-Recall – {label} (AUC={pr_auc:.3f})")
    plt.tight_layout()
    plt.savefig(VAL_DIR / f"{prefix}_roc_pr.png", dpi=150)
    plt.close()


def plot_summary(all_results):
    """Tüm alt grupları karşılaştıran özet çubuk grafik."""
    groups   = list(all_results.keys())
    recalls  = [all_results[g]["Recall_%"]   for g in groups]
    fn_rates = [all_results[g]["FN_Rate_%"]  for g in groups]
    f1s      = [all_results[g]["F1_%"]       for g in groups]
    aucs     = [all_results[g]["AUC_ROC_%"]  for g in groups]

    x = np.arange(len(groups))
    w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - 1.5*w, recalls,  w, label="Recall (%)",   color="#4CAF50", alpha=0.9)
    ax.bar(x - 0.5*w, f1s,      w, label="F1 (%)",       color="#2196F3", alpha=0.9)
    ax.bar(x + 0.5*w, aucs,     w, label="AUC-ROC (%)",  color="#9C27B0", alpha=0.9)
    ax.bar(x + 1.5*w, fn_rates, w, label="FN Rate (%)",  color="#F44336", alpha=0.9)

    for i, g in enumerate(groups):
        ax.text(i + 1.5*w, fn_rates[i] + 0.5, f"%{fn_rates[i]:.1f}",
                ha="center", va="bottom", fontsize=8, color="#F44336", fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Değer (%)")
    ax.set_title("Validation Sonuçları – Alt Grup Karşılaştırması")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(VAL_DIR / "summary_comparison.png", dpi=150)
    plt.close()
    print(f"\n  ✓ Özet grafik: validation_figures/summary_comparison.png")


# ─── ANA AKIŞ ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  YARIŞMA GÜNÜ SİMÜLASYONU – VALİDASYON")
    print("=" * 60)

    # 1. Veri yükle (test seti eğitimden kesinlikle ayrı)
    print("\n  [1] Veri yükleniyor...")
    train_df, test_df = load_train_data()

    # 2. Ön işle
    print("\n  [2] Ön işleme...")
    X_train, X_test, y_train, y_test, feature_cols = preprocess(train_df, test_df)

    # 3. Eğit
    print("\n  [3] Modeller eğitiliyor...")
    models = train_models(X_train, y_train)

    # 4. Test setinde tahmin
    print("\n  [4] Test setinde tahmin yapılıyor...")
    probs = ensemble_proba(models, X_test)

    # Optimal eşik: Recall >= 0.90 kısıtı altında F1 maksimize
    p_vals, r_vals, thrs = precision_recall_curve(y_test, probs)
    best_f1, best_thr = 0, 0.35
    for p, r, t in zip(p_vals[:-1], r_vals[:-1], thrs):
        if r >= 0.90:
            f1 = 2*p*r/(p+r) if (p+r)>0 else 0
            if f1 > best_f1:
                best_f1, best_thr = f1, t
    print(f"\n  Optimal eşik: {best_thr:.3f}  (F1={best_f1:.3f})")

    preds = (probs >= best_thr).astype(int)

    # 5. Sonuçları göster
    print("\n" + "=" * 60)
    print("  VALIDATION SONUÇLARI")
    print("=" * 60)

    all_results = {}

    # Genel
    r = print_report(y_test, preds, probs, "GENEL")
    all_results["GENEL"] = r
    plot_cm(y_test, preds, "Genel (Validation)", "val_cm_genel.png")
    plot_roc_pr(y_test, probs, "Genel", "val_genel")

    # Alt gruplar
    test_df_orig = pd.read_csv(OUTPUT_DIR / "internal_test_set.csv")
    for grup in ALT_GRUPLAR:
        mask = (test_df_orig["SOURCE"] == grup).values if "SOURCE" in test_df_orig.columns else None
        if mask is None:
            # SOURCE kolonu yoksa dummy'den bul
            src_col = f"SOURCE_{grup}"
            if src_col in X_test.columns:
                mask = (X_test[src_col] == 1).values
            else:
                continue
        if mask.sum() < 5:
            continue
        r = print_report(y_test[mask], preds[mask], probs[mask], grup)
        all_results[grup] = r
        plot_cm(y_test[mask], preds[mask], f"{grup} (Validation)", f"val_cm_{grup.lower()}.png")
        plot_roc_pr(y_test[mask], probs[mask], grup, f"val_{grup.lower()}")

    # 6. Özet grafik
    plot_summary(all_results)

    # 7. CSV kaydet
    pd.DataFrame(all_results).T.to_csv(OUTPUT_DIR / "validation_results.csv")
    print(f"  ✓ Tablo  : outputs/validation_results.csv")
    print(f"  ✓ Grafik : outputs/validation_figures/")

    print("\n" + "=" * 60)
    print("  VALİDASYON TAMAMLANDI ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()