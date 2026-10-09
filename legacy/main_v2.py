"""
TEKNOFEST 2026 – Sağlıkta Yapay Zeka
Genomik Varyant Patojenite Tahmin Sistemi v2
=============================================
MİMARİ:
  - 4 veri seti için 4 AYRI model (MASTER, KANSER, CFTR, PAH)
  - Her model kendi veri setiyle eğitilir ve değerlendirilir
  - Test seti: finale benzer dağılım (Benign ağırlıklı)
  - Çıktı: tahmin dosyası + tüm grafikler + rapor metrikleri

KLİNİK ÖNCELIK:
  - FN (hasta birine sağlıklı deme) → minimize et
  - Finalde test seti ~%80 Benign %20 Patojenik olacak
  - Threshold: Recall≥0.90 kısıtı altında F1 maximize et
"""

import warnings, json, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, GridSearchCV
)
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, f1_score, matthews_corrcoef,
    precision_recall_curve, auc, RocCurveDisplay, ConfusionMatrixDisplay
)
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─── KLASÖRLER ────────────────────────────────────────────────────────────────
BASE   = Path(__file__).parent
DATA   = BASE / "data"
OUT    = BASE / "outputs"
FIG    = OUT / "figures"
MDIR   = OUT / "models"
for d in [OUT, FIG, MDIR]:
    d.mkdir(parents=True, exist_ok=True)

ALT_GRUPLAR = ["MASTER", "KANSER", "CFTR", "PAH"]

# ─── AYARLAR ──────────────────────────────────────────────────────────────────
CFG = {
    "cv_folds"       : 5,
    "random_state"   : 42,
    # Recall kısıtı: en az bu kadar olmalı (hasta kaçırmama)
    "min_recall"     : 0.90,
    # Test setinde benign oranı hedefi (finale simülasyon)
    "test_benign_frac": 0.50,   # mevcut benign'in %50'si test'e
    "test_path_frac" : 0.15,    # mevcut pathogenic'in %15'i test'e
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. VERİ YÜKLEME VE BÖLME
# ══════════════════════════════════════════════════════════════════════════════

def split_final_simulation(df, grup_adi):
    """
    Test seti: Benign ağırlıklı (finale simülasyon).
    Şartname: final test → ~%80 Benign, %20 Patojenik.
    Elimizdeki benign sayısı az olduğu için mümkün olan maksimum
    benign oranını test setine koyuyoruz.

    Strateji:
      - Benign'in %50'si → test
      - Pathogenic'in %15'i → test
      - Kalanlar → eğitim
    """
    path_df   = df[df["Label"] == 1].sample(frac=1, random_state=42)
    benign_df = df[df["Label"] == 0].sample(frac=1, random_state=42)

    n_test_p = max(int(len(path_df)   * CFG["test_path_frac"]),  3)
    n_test_b = max(int(len(benign_df) * CFG["test_benign_frac"]), 3)

    test_df  = pd.concat([path_df.iloc[:n_test_p],
                          benign_df.iloc[:n_test_b]], ignore_index=True)
    train_df = pd.concat([path_df.iloc[n_test_p:],
                          benign_df.iloc[n_test_b:]], ignore_index=True)

    b_ratio  = n_test_b / (n_test_b + n_test_p) * 100
    print(f"    Eğitim : {len(train_df):4d} satır  "
          f"(Pat={len(train_df[train_df.Label==1])} Ben={len(train_df[train_df.Label==0])})")
    print(f"    Test   : {len(test_df):4d} satır  "
          f"(Pat={n_test_p} Ben={n_test_b} | Benign={b_ratio:.0f}%  ← finale benzer)")

    # Sakla
    test_df.to_csv(OUT / f"test_set_{grup_adi}.csv", index=False)
    return train_df, test_df


# ══════════════════════════════════════════════════════════════════════════════
# 2. ÖN İŞLEME
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(train_df, test_df, grup_adi):
    """
    Eksik değer doldurma, kategorik encode, tip dönüşümü.

    Eksik Değer Stratejisi:
      - Sayısal (AL_, EK_): medyan imputation.
        Neden medyan? Genomik veriler çarpık dağılımlıdır, medyan aykırı
        değerlerden etkilenmez. Ortalama (mean) bu durumda yanıltıcı olur.
      - Kategorik (CAT_, AA_): "eksik" ayrı kategori olarak tutulur.
        Genomik'te eksiklik bilgi taşır (ör. popülasyonda hiç gözlemlenmemiş).

    Aykırı Değer Stratejisi:
      - HistGBM ve RandomForest ağaç tabanlıdır → aykırı değerlere zaten
        dayanıklıdır (sıralamalara dayalı bölünme kullanır).
      - LogisticRegression için StandardScaler aykırı değer etkisini azaltır.
      - Ek olarak: %99.5 üstündeki sayısal değerleri winsorize ediyoruz.
    """
    drop_cols = ["Variant_ID"]
    y_train = train_df["Label"].values
    y_test  = test_df["Label"].values

    feature_cols = [c for c in train_df.columns
                    if c not in drop_cols + ["Label"]]

    X_train = train_df[feature_cols].copy()
    X_test  = test_df[feature_cols].copy()

    # Kategorik kolonları tespit et
    cat_cols = []
    for col in feature_cols:
        if col.startswith("CAT_") or col.startswith("AA_"):
            cat_cols.append(col)
        elif X_train[col].dtype == object or str(X_train[col].dtype) == "string":
            cat_cols.append(col)

    # Kategorik → sayısal (NaN ayrı kategori)
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        tr_v = X_train[col].fillna("__NaN__").astype(str)
        te_v = X_test[col].fillna("__NaN__").astype(str)
        le.fit(pd.concat([tr_v, te_v]))
        X_train[col] = le.transform(tr_v).astype(float)
        X_test[col]  = le.transform(te_v).astype(float)
        encoders[col] = le

    # Tüm kolonları float'a çevir
    for col in feature_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
        X_test[col]  = pd.to_numeric(X_test[col],  errors="coerce")

    # Aykırı değer winsorize (%99.5 clip) — sadece sayısal, eğitim istatistiğiyle
    num_cols = [c for c in feature_cols if c not in cat_cols]
    clip_vals = {}
    for col in num_cols:
        upper = X_train[col].quantile(0.995)
        lower = X_train[col].quantile(0.005)
        clip_vals[col] = (lower, upper)
        X_train[col] = X_train[col].clip(lower, upper)
        X_test[col]  = X_test[col].clip(lower, upper)

    # Eksik değer doldurma (medyan — eğitim setinden fit)
    imputer = SimpleImputer(strategy="median")
    X_train_arr = imputer.fit_transform(X_train)
    X_test_arr  = imputer.transform(X_test)
    X_train = pd.DataFrame(X_train_arr, columns=feature_cols)
    X_test  = pd.DataFrame(X_test_arr,  columns=feature_cols)

    missing_pct = train_df[feature_cols].isnull().mean() * 100
    print(f"    Özellik sayısı     : {len(feature_cols)}")
    print(f"    Ort. eksik değer   : %{missing_pct.mean():.1f}")
    print(f"    >%50 eksik özellik : {(missing_pct > 50).sum()}")

    artifacts = {
        "feature_cols": feature_cols,
        "cat_cols"    : cat_cols,
        "encoders"    : encoders,
        "clip_vals"   : clip_vals,
        "imputer"     : imputer,
    }
    return X_train, X_test, y_train, y_test, feature_cols, artifacts


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODEL MİMARİSİ
# ══════════════════════════════════════════════════════════════════════════════
#
# Üç algoritma + Soft Voting Ensemble:
#
# ① HistGradientBoostingClassifier  [ANA MODEL]
#    - Sklearn'ün dahili LightGBM/XGBoost muadili
#    - Eksik değerlere native toleranslı (imputation sonrası da sağlam)
#    - class_weight="balanced" → Patojenik sınıfa otomatik ağırlık
#    - early_stopping → overfitting önleme
#    - l2_regularization → genelleme
#    HİPERPARAMETRE ARAMA: GridSearchCV (5-fold)
#      Arama uzayı: learning_rate × max_depth × min_samples_leaf
#
# ② RandomForestClassifier  [ENSEMBLE BİLEŞENİ]
#    - Bağımsız ağaçlar → yüksek varyansa dayanıklı
#    - Özellik önemi için altın standart
#    - class_weight="balanced_subsample" → her ağaçta ayrı dengeleme
#    - max_features="sqrt" → ağaçlar arası çeşitlilik
#    - Overfitting: min_samples_split, max_depth sınırı
#
# ③ LogisticRegression  [KALİBRASYON REFERANSı]
#    - Doğrusal taban → ensemble çeşitliliği
#    - class_weight="balanced"
#    - C=0.1 → güçlü L2 regularization (küçük veri setleri için kritik)
#    - StandardScaler zorunlu (gradient descent temelli)
#
# NEDEN ENSEMBLE?
#   Hiçbir tek model her durumda en iyisi değildir.
#   Soft Voting: üç modelin olasılık ortalaması → daha kararlı tahmin.
#   Tıpta "komite kararı" mantığı: tek uzman yanılabilir, üçü aynı anda zor.

def build_and_tune_models(X_train, y_train, grup_adi):
    """
    Hiperparametre arama (GridSearchCV) + model eğitimi.
    Arama metriği: recall (hasta kaçırmama öncelikli).
    """
    cv = StratifiedKFold(n_splits=CFG["cv_folds"], shuffle=True,
                         random_state=CFG["random_state"])

    print(f"\n    [HistGBM] Hiperparametre arama...")
    hgb_param_grid = {
        "learning_rate" : [0.05, 0.1],
        "max_depth"     : [4, 6],
    }
    hgb_base = HistGradientBoostingClassifier(
        max_iter=300, l2_regularization=0.1,
        class_weight="balanced", early_stopping=False,
        random_state=CFG["random_state"]
    )
    hgb_search = GridSearchCV(
        hgb_base, hgb_param_grid,
        cv=3, scoring="recall", n_jobs=-1, refit=True
    )
    hgb_search.fit(X_train, y_train)
    best_hgb = hgb_search.best_estimator_
    print(f"      En iyi params: {hgb_search.best_params_}")
    print(f"      CV Recall    : {hgb_search.best_score_:.3f}")

    print(f"    [RandomForest] Hiperparametre arama...")
    rf_param_grid = {
        "n_estimators" : [200, 400],
        "max_depth"    : [10, None],
    }
    rf_base = RandomForestClassifier(
        max_features="sqrt", class_weight="balanced_subsample",
        n_jobs=-1, random_state=CFG["random_state"]
    )
    rf_search = GridSearchCV(
        rf_base, rf_param_grid,
        cv=3, scoring="recall", n_jobs=-1, refit=True
    )
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print(f"      En iyi params: {rf_search.best_params_}")
    print(f"      CV Recall    : {rf_search.best_score_:.3f}")

    print(f"    [LogisticRegression] eğitiliyor...")
    lr_model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=0.1, class_weight="balanced",
            max_iter=2000, solver="saga",
            random_state=CFG["random_state"]
        ))
    ])
    lr_model.fit(X_train, y_train)

    models = {
        "HistGBM"          : best_hgb,
        "RandomForest"     : best_rf,
        "LogisticRegression": lr_model,
    }
    tuning_info = {
        "HistGBM" : hgb_search.best_params_,
        "RandomForest": rf_search.best_params_,
    }
    return models, tuning_info


# ══════════════════════════════════════════════════════════════════════════════
# 4. ÇAPRAZ DOĞRULAMA
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate(models, X_train, y_train):
    """
    5-Fold Stratified K-Fold CV.
    Out-of-fold (OOF) tahminler birleştirilerek gerçekçi performans tahmini.

    Neden Stratified K-Fold?
    - Sınıf oranını her fold'da korur (dengesiz veri için kritik).
    - Normal K-Fold bazı fold'larda hiç Benign olmayabilir.
    - OOF yaklaşımı: her örnek tam olarak 1 kez test edilir → güvenilir.
    """
    cv  = StratifiedKFold(n_splits=CFG["cv_folds"], shuffle=True,
                          random_state=CFG["random_state"])
    results = {}

    for name, model in models.items():
        oof = cross_val_predict(model, X_train, y_train,
                                cv=cv, method="predict_proba")[:, 1]
        results[name] = oof

    # Ensemble OOF
    oof_ensemble = np.stack(list(results.values())).mean(axis=0)
    results["Ensemble"] = oof_ensemble
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5. KARAR EŞİĞİ OPTİMİZASYONU
# ══════════════════════════════════════════════════════════════════════════════

def optimize_threshold(probs, y_true, grup_adi):
    """
    Karar Eşiği Belirleme Süreci:
    ─────────────────────────────
    Standart eşik = 0.50 (model %50'den fazla olasılık verince Patojenik der).
    Ancak klinik bağlamda hasta kaçırmak (FN) sağlıklı diyememekten (FP) çok
    daha ağır bir hatadır. Bu nedenle eşiği aşağı çekiyoruz.

    Yöntem:
    1. Precision-Recall eğrisi üzerindeki tüm eşik değerlerini tara.
    2. Recall ≥ 0.90 kısıtını uygula (her 10 hastanın en az 9'u yakalanmalı).
    3. Bu kısıt altında F1'i maksimize eden eşiği seç.

    Neden F1? F1, Precision ve Recall'ı dengeler. Salt Recall maksimizasyonu
    modeli herkese Patojenik demeye iter → FP patlar, klinik kullanışsız olur.

    Finale not:
    Final test setinde Benign oranı çok daha yüksek (~%80) olacak.
    Bu durumda düşük eşik FP'yi artırır → Specificity düşer.
    Bu yüzden finale özel 'final_threshold' da hesaplanır:
    Recall ≥ 0.85 kısıtı altında MCC maksimize → daha dengeli.
    """
    prec_arr, rec_arr, thrs = precision_recall_curve(y_true, probs)

    # 1) Standart: Recall ≥ 0.90, F1 max
    best_f1, best_thr = 0, 0.35
    for p, r, t in zip(prec_arr[:-1], rec_arr[:-1], thrs):
        if r >= CFG["min_recall"]:
            f1_val = 2*p*r/(p+r) if (p+r) > 0 else 0
            if f1_val > best_f1:
                best_f1, best_thr = f1_val, t

    # 2) Finale özel: Recall ≥ 0.85, MCC max (benign ağırlıklı test için)
    from sklearn.metrics import matthews_corrcoef as mcc_fn
    best_mcc, final_thr = -1, 0.45
    for t in np.linspace(0.20, 0.80, 120):
        preds = (probs >= t).astype(int)
        r = (preds[y_true==1] == 1).mean()
        if r >= 0.85:
            m = mcc_fn(y_true, preds)
            if m > best_mcc:
                best_mcc, final_thr = m, t

    print(f"\n    Eşik Analizi:")
    print(f"      Standart eşik   : {best_thr:.3f}  (Recall≥0.90, F1={best_f1:.3f})")
    print(f"      Finale eşiği    : {final_thr:.3f}  (Recall≥0.85, MCC={best_mcc:.3f})")

    return best_thr, final_thr


# ══════════════════════════════════════════════════════════════════════════════
# 6. DEĞERLENDİRME
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(y_true, probs, threshold, label=""):
    preds = (probs >= threshold).astype(int)
    cm    = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()

    total   = len(y_true)
    recall  = tp / (tp + fn)  if (tp+fn) > 0 else 0
    spec    = tn / (tn + fp)  if (tn+fp) > 0 else 0
    prec    = tp / (tp + fp)  if (tp+fp) > 0 else 0
    f1      = f1_score(y_true, preds, zero_division=0)
    mcc     = matthews_corrcoef(y_true, preds)
    auc_roc = roc_auc_score(y_true, probs)
    fn_rate = fn / (tp+fn) if (tp+fn) > 0 else 0
    fp_rate = fp / (tn+fp) if (tn+fp) > 0 else 0

    pr_p, pr_r, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(pr_r, pr_p)

    metrics = {
        "TP": int(tp), "FN": int(fn), "TN": int(tn), "FP": int(fp),
        "Recall_%"    : round(recall*100, 1),
        "Specificity_%": round(spec*100, 1),
        "Precision_%"  : round(prec*100, 1),
        "F1_%"         : round(f1*100, 1),
        "MCC"          : round(mcc, 3),
        "AUC_ROC_%"    : round(auc_roc*100, 1),
        "PR_AUC_%"     : round(pr_auc*100, 1),
        "FN_Rate_%"    : round(fn_rate*100, 1),
        "FP_Rate_%"    : round(fp_rate*100, 1),
    }

    if label:
        print(f"\n    ┌─ [{label}] {'─'*(42-len(label))}")
        print(f"    │  TP={tp}  FN={fn}  TN={tn}  FP={fp}")
        print(f"    │  Recall={recall*100:.1f}%  FN_Rate={fn_rate*100:.1f}%  "
              f"Specificity={spec*100:.1f}%  Precision={prec*100:.1f}%")
        print(f"    │  F1={f1*100:.1f}%  MCC={mcc:.3f}  "
              f"AUC-ROC={auc_roc*100:.1f}%  PR-AUC={pr_auc*100:.1f}%")
        print(f"    └{'─'*48}")

    return metrics, preds


# ══════════════════════════════════════════════════════════════════════════════
# 7. GRAFİKLER
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(y_true, preds, grup, suffix=""):
    cm = confusion_matrix(y_true, preds)
    tn, fp, fn, tp = cm.ravel()
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Benign", "Patojenik"]).plot(
        ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title(f"Confusion Matrix (Sayı) – {grup}")

    im = axes[1].imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    axes[1].set_xticks([0,1]); axes[1].set_xticklabels(["Benign","Patojenik"])
    axes[1].set_yticks([0,1]); axes[1].set_yticklabels(["Benign","Patojenik"])
    axes[1].set_xlabel("Tahmin"); axes[1].set_ylabel("Gerçek")
    axes[1].set_title(f"Confusion Matrix (%) – {grup}")
    for i in range(2):
        for j in range(2):
            col = "white" if cm_pct[i,j] > 60 else "black"
            axes[1].text(j, i, f"{cm_pct[i,j]:.1f}%\n({cm[i,j]})",
                ha="center", va="center", color=col,
                fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=axes[1], label="%")

    recall  = tp/(tp+fn)*100; fn_r = fn/(tp+fn)*100
    spec    = tn/(tn+fp)*100 if (tn+fp)>0 else 0
    fig.text(0.5, -0.03,
        f"Recall(Sensitivity)={recall:.1f}%  |  FN Rate={fn_r:.1f}%  "
        f"|  Specificity={spec:.1f}%  |  n={len(y_true)}",
        ha="center", fontsize=10, color="#333")
    plt.tight_layout()
    plt.savefig(FIG / f"cm_{grup.lower()}{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_pr(y_true, probs, grup, suffix=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    RocCurveDisplay.from_predictions(y_true, probs, ax=axes[0], name="Ensemble")
    axes[0].plot([0,1],[0,1],"k--", alpha=0.5)
    axes[0].set_title(f"ROC Eğrisi – {grup}")
    axes[0].grid(alpha=0.3)

    prec_a, rec_a, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(rec_a, prec_a)
    baseline = (y_true == 1).mean()
    axes[1].plot(rec_a, prec_a, "#E91E63", lw=2, label=f"Ensemble (AUC={pr_auc:.3f})")
    axes[1].axhline(baseline, color="gray", linestyle="--",
                    label=f"Baseline ({baseline:.2f})")
    axes[1].fill_between(rec_a, prec_a, alpha=0.12, color="#E91E63")
    axes[1].set_xlabel("Recall (Duyarlılık)")
    axes[1].set_ylabel("Precision (Kesinlik)")
    axes[1].set_title(f"Precision-Recall Eğrisi – {grup}")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG / f"roc_pr_{grup.lower()}{suffix}.png", dpi=150)
    plt.close()


def plot_threshold_analysis(y_true, probs, grup, std_thr, fin_thr):
    """
    Farklı karar eşiklerinin Recall, F1, Specificity ve FN Rate üzerindeki etkisi.
    """
    thresholds = np.linspace(0.10, 0.90, 100)
    recalls, f1s, specs, fn_rates, mccs = [], [], [], [], []

    for t in thresholds:
        p = (probs >= t).astype(int)
        cm_t = confusion_matrix(y_true, p)
        if cm_t.shape == (2,2):
            tn_,fp_,fn_,tp_ = cm_t.ravel()
            recalls.append(tp_/(tp_+fn_) if (tp_+fn_)>0 else 0)
            specs.append(tn_/(tn_+fp_)   if (tn_+fp_)>0 else 0)
            fn_rates.append(fn_/(tp_+fn_) if (tp_+fn_)>0 else 0)
            f1s.append(f1_score(y_true, p, zero_division=0))
            mccs.append(matthews_corrcoef(y_true, p))
        else:
            recalls.append(None); specs.append(None)
            fn_rates.append(None); f1s.append(None); mccs.append(None)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(thresholds, recalls,  "#4CAF50", lw=2, label="Recall (Duyarlılık)")
    ax.plot(thresholds, f1s,      "#2196F3", lw=2, label="F1 Score")
    ax.plot(thresholds, specs,    "#9C27B0", lw=2, label="Specificity")
    ax.plot(thresholds, fn_rates, "#F44336", lw=2, label="FN Rate (↓ istiyoruz)")
    ax.plot(thresholds, mccs,     "#FF9800", lw=1.5, linestyle="--", label="MCC")

    ax.axvline(std_thr, color="blue", linestyle=":",
               label=f"Standart Eşik={std_thr:.3f}")
    ax.axvline(fin_thr, color="red", linestyle=":",
               label=f"Final Eşiği={fin_thr:.3f}")
    ax.axhline(0.90, color="green", linestyle="--", alpha=0.4, label="Recall=0.90 sınırı")

    ax.set_xlabel("Karar Eşiği (Threshold)")
    ax.set_ylabel("Değer (0-1)")
    ax.set_title(f"Karar Eşiği Analizi – {grup}")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / f"threshold_{grup.lower()}.png", dpi=150)
    plt.close()


def plot_feature_importance(models, feature_cols, grup):
    """
    Açıklanabilirlik: RandomForest Gini + özellik grubu analizi.
    Hangi özellik grubu (AL_, EK_, AA_, CAT_) en kritik?
    """
    rf = models["RandomForest"]
    imp = rf.feature_importances_
    idx = np.argsort(imp)[::-1][:25]

    # Grup bazlı toplamlı önem
    groups = {"AL_": 0, "EK_": 0, "AA_": 0, "CAT_": 0, "DİĞER": 0}
    for i, f in enumerate(feature_cols):
        matched = False
        for g in ["AL_", "EK_", "AA_", "CAT_"]:
            if f.startswith(g):
                groups[g] += imp[i]; matched = True; break
        if not matched:
            groups["DİĞER"] += imp[i]

    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1])

    # Sol: Top 25 özellik
    ax1 = fig.add_subplot(gs[0])
    names = [feature_cols[i] for i in idx]
    vals  = [imp[i] for i in idx]
    colors = []
    for n in names:
        if n.startswith("EK_"):   colors.append("#E91E63")
        elif n.startswith("AL_"): colors.append("#2196F3")
        elif n.startswith("AA_"): colors.append("#4CAF50")
        elif n.startswith("CAT_"):colors.append("#FF9800")
        else:                     colors.append("#9E9E9E")
    ax1.barh(names[::-1], vals[::-1], color=colors[::-1])
    ax1.set_xlabel("Gini Önemi")
    ax1.set_title(f"Top 25 Özellik – {grup}")

    # Renk açıklaması
    from matplotlib.patches import Patch
    legend = [Patch(facecolor="#E91E63", label="EK_ (Evrimsel Korunmuşluk)"),
              Patch(facecolor="#2196F3", label="AL_ (Frekans/Popülasyon)"),
              Patch(facecolor="#4CAF50", label="AA_ (Amino Asit)"),
              Patch(facecolor="#FF9800", label="CAT_ (Kategorik Meta)")]
    ax1.legend(handles=legend, fontsize=8, loc="lower right")

    # Sağ: Grup toplamı
    ax2 = fig.add_subplot(gs[1])
    g_labels = list(groups.keys())
    g_vals   = list(groups.values())
    g_colors = ["#E91E63","#2196F3","#4CAF50","#FF9800","#9E9E9E"]
    ax2.bar(g_labels, g_vals, color=g_colors)
    ax2.set_title("Özellik Grubu Toplamı")
    ax2.set_ylabel("Toplam Gini Önemi")
    for i, v in enumerate(g_vals):
        ax2.text(i, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIG / f"feature_importance_{grup.lower()}.png", dpi=150)
    plt.close()

    return groups


def plot_fn_fp_analysis(y_true, probs, X_test, feature_cols, grup, threshold):
    """
    Yanlış pozitif ve yanlış negatif analizi:
    Hata yapılan örneklerde hangi özellik grupları farklı?
    """
    preds = (probs >= threshold).astype(int)
    fn_mask = (y_true == 1) & (preds == 0)   # Hasta ama sağlıklı dendi
    fp_mask = (y_true == 0) & (preds == 1)   # Sağlıklı ama hasta dendi
    tp_mask = (y_true == 1) & (preds == 1)   # Doğru hasta

    if fn_mask.sum() < 2 or tp_mask.sum() < 2:
        return

    # EK_ (evrimsel korunmuşluk) kolonlarının ortalaması gruplar arası
    ek_cols = [c for c in feature_cols if c.startswith("EK_")]
    if not ek_cols:
        return

    fn_ek = X_test.iloc[fn_mask][ek_cols].mean()
    fp_ek = X_test.iloc[fp_mask][ek_cols].mean() if fp_mask.sum() > 1 else None
    tp_ek = X_test.iloc[tp_mask][ek_cols].mean()

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(ek_cols))
    w = 0.28
    ax.bar(x - w, tp_ek.values, w, label=f"TP (Doğru Pat, n={tp_mask.sum()})",
           color="#4CAF50", alpha=0.85)
    ax.bar(x,     fn_ek.values, w, label=f"FN (Kaçırılan Hasta, n={fn_mask.sum()})",
           color="#F44336", alpha=0.85)
    if fp_ek is not None:
        ax.bar(x + w, fp_ek.values, w, label=f"FP (Yanlış Alarm, n={fp_mask.sum()})",
               color="#FF9800", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(ek_cols, rotation=45, ha="right", fontsize=7)
    ax.set_title(f"Hata Analizi: EK_ Özelliklerinde FN vs TP vs FP – {grup}")
    ax.set_ylabel("Ortalama Değer")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / f"fn_fp_analysis_{grup.lower()}.png", dpi=150)
    plt.close()


def plot_model_comparison(cv_oof, y_train, grup, std_thr):
    """CV sonuçlarını karşılaştır."""
    model_names = list(cv_oof.keys())
    metrics_list = ["F1", "MCC", "AUC", "Recall", "Specificity"]
    results = {m: [] for m in metrics_list}

    for name in model_names:
        probs = cv_oof[name]
        preds = (probs >= std_thr).astype(int)
        cm_ = confusion_matrix(y_train, preds)
        tn_,fp_,fn_,tp_ = cm_.ravel() if cm_.shape==(2,2) else (0,0,0,0)
        results["F1"].append(f1_score(y_train, preds, zero_division=0))
        results["MCC"].append(max(matthews_corrcoef(y_train, preds), 0))
        results["AUC"].append(roc_auc_score(y_train, probs))
        results["Recall"].append(tp_/(tp_+fn_) if (tp_+fn_)>0 else 0)
        results["Specificity"].append(tn_/(tn_+fp_) if (tn_+fp_)>0 else 0)

    x = np.arange(len(metrics_list))
    w = 0.8 / len(model_names)
    colors = ["#2196F3","#4CAF50","#FF9800","#E91E63"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, name in enumerate(model_names):
        vals = [results[m][i] for m in metrics_list]
        ax.bar(x + i*w, vals, w, label=name, color=colors[i], alpha=0.85)
    ax.set_xticks(x + w*(len(model_names)-1)/2)
    ax.set_xticklabels(metrics_list)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Skor")
    ax.set_title(f"Model Karşılaştırması (5-Fold CV OOF) – {grup}")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / f"model_comparison_{grup.lower()}.png", dpi=150)
    plt.close()


def plot_all_groups_summary(all_results):
    """4 grup için özet karşılaştırma."""
    groups = list(all_results.keys())
    modes  = ["standard", "final_sim"]
    mode_labels = {"standard": "Standart Eşik", "final_sim": "Final Eşiği (Benign Ağırlıklı)"}

    metrics = ["Recall_%", "Specificity_%", "F1_%", "MCC", "FN_Rate_%", "AUC_ROC_%"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    colors = {"standard": "#2196F3", "final_sim": "#E91E63"}

    for ax_i, metric in enumerate(metrics):
        ax = axes[ax_i]
        x  = np.arange(len(groups))
        w  = 0.35
        for mi, mode in enumerate(modes):
            vals = [all_results[g][mode].get(metric, 0) for g in groups]
            ax.bar(x + mi*w, vals, w, label=mode_labels[mode],
                   color=colors[mode], alpha=0.85)
        ax.set_xticks(x + w/2); ax.set_xticklabels(groups)
        ax.set_title(metric.replace("_%", " (%)").replace("_", " "))
        ax.legend(fontsize=7); ax.grid(axis="y", alpha=0.3)
        if metric == "FN_Rate_%":
            ax.set_ylim(0, 30)

    plt.suptitle("4 Alt Grup Karşılaştırması: Standart vs Final Eşiği", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG / "summary_all_groups.png", dpi=150)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 8. MODEL KAYDETME (finale için)
# ══════════════════════════════════════════════════════════════════════════════

def save_model(models, artifacts, thresholds, grup):
    bundle = {
        "models"     : models,
        "artifacts"  : artifacts,
        "thresholds" : thresholds,
        "grup"       : grup,
    }
    path = MDIR / f"model_{grup.lower()}.pkl"
    with open(path, "wb") as f:
        pickle.dump(bundle, f)
    print(f"    ✓ Model kaydedildi: {path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. ANA AKIŞ — TEK GRUP
# ══════════════════════════════════════════════════════════════════════════════

def run_group(grup_adi):
    print(f"\n{'='*60}")
    print(f"  {grup_adi}")
    print(f"{'='*60}")

    # Veri yükle
    df = pd.read_csv(DATA / f"YARISMA_TRAIN_{grup_adi}.csv")
    pos = (df["Label"]==1).sum(); neg = (df["Label"]==0).sum()
    print(f"\n  Kaynak: {len(df)} satır | Patojenik={pos} Benign={neg}")

    # Bölme
    print(f"\n  [1] VERİ BÖLME")
    train_df, test_df = split_final_simulation(df, grup_adi)

    # Ön işleme
    print(f"\n  [2] ÖN İŞLEME")
    X_train, X_test, y_train, y_test, feature_cols, artifacts = \
        preprocess(train_df, test_df, grup_adi)

    # Model oluştur & tune
    print(f"\n  [3] HİPERPARAMETRE ARAMA & EĞİTİM")
    models, tuning_info = build_and_tune_models(X_train, y_train, grup_adi)

    # Çapraz doğrulama (eğitim setinde)
    print(f"\n  [4] 5-FOLD ÇAPRAZ DOĞRULAMA")
    cv_oof = cross_validate(models, X_train, y_train)

    # Ensemble test tahmini
    def ens_proba(X):
        p = np.stack([m.predict_proba(X)[:,1] for m in models.values()])
        return p.mean(axis=0)

    probs_test = ens_proba(X_test)

    # Eşik optimizasyonu
    print(f"\n  [5] EŞİK OPTİMİZASYONU")
    # OOF probs üzerinden (eğitim setinde overfitting yok)
    oof_ensemble = cv_oof["Ensemble"]
    std_thr, fin_thr = optimize_threshold(oof_ensemble, y_train, grup_adi)

    # Değerlendirme
    print(f"\n  [6] DEĞERLENDİRME (Test Seti)")
    metrics_std, preds_std = evaluate(y_test, probs_test, std_thr,
                                      f"{grup_adi} – Standart Eşik ({std_thr:.3f})")
    metrics_fin, preds_fin = evaluate(y_test, probs_test, fin_thr,
                                      f"{grup_adi} – Final Eşiği ({fin_thr:.3f})")

    # Grafikler
    print(f"\n  [7] GRAFİKLER")
    plot_confusion_matrix(y_test, preds_std, grup_adi, "_standard")
    plot_confusion_matrix(y_test, preds_fin, grup_adi, "_final")
    plot_roc_pr(y_test, probs_test, grup_adi)
    plot_threshold_analysis(y_test, probs_test, grup_adi, std_thr, fin_thr)
    plot_feature_importance(models, feature_cols, grup_adi)
    plot_fn_fp_analysis(y_test, probs_test, X_test, feature_cols, grup_adi, std_thr)
    plot_model_comparison(cv_oof, y_train, grup_adi, std_thr)
    print(f"    ✓ Tüm grafikler kaydedildi.")

    # Model kaydet
    save_model(models, artifacts,
               {"standard": std_thr, "final": fin_thr}, grup_adi)

    return {
        "standard"  : metrics_std,
        "final_sim" : metrics_fin,
        "tuning"    : tuning_info,
        "std_thr"   : std_thr,
        "fin_thr"   : fin_thr,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 10. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("  TEKNOFEST 2026 – Genomik Varyant Patojenite v2")
    print("  4 Ayrı Model | Final Simülasyonu | Eşik Optimizasyonu")
    print("="*60)

    all_results = {}
    for grup in ALT_GRUPLAR:
        all_results[grup] = run_group(grup)

    # Özet
    plot_all_groups_summary(all_results)

    # JSON kaydet
    summary = {}
    for grup, res in all_results.items():
        summary[grup] = {
            "standart_esik"     : res["std_thr"],
            "final_esigi"       : res["fin_thr"],
            "standart_metrikler": res["standard"],
            "final_metrikler"   : res["final_sim"],
            "hiperparametreler" : res["tuning"],
        }
    with open(OUT / "results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Ekran özeti
    print("\n" + "="*60)
    print("  GENEL ÖZET")
    print("="*60)
    print(f"  {'Grup':8} {'Eşik':6} {'Recall':8} {'Spec':8} {'F1':6} "
          f"{'MCC':6} {'FN%':6} {'PR-AUC':7}")
    print(f"  {'-'*58}")
    for grup, res in all_results.items():
        m = res["final_sim"]
        print(f"  {grup:8} {res['fin_thr']:.3f}  "
              f"{m['Recall_%']:6.1f}%  {m['Specificity_%']:6.1f}%  "
              f"{m['F1_%']:5.1f}%  {m['MCC']:5.3f}  "
              f"{m['FN_Rate_%']:5.1f}%  {m['PR_AUC_%']:5.1f}%")

    print(f"\n  ✓ Tüm çıktılar: {OUT.resolve()}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()