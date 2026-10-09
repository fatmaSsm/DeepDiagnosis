"""
predict.py — Yarışma Günü Tahmin Scripti
=========================================
Kullanım:
  python predict.py --input yeni_veri.csv --grup MASTER
  python predict.py --input yeni_veri.csv --grup KANSER
  python predict.py --input yeni_veri.csv --grup CFTR
  python predict.py --input yeni_veri.csv --grup PAH

  Grup bilinmiyorsa (MASTER modeli tümünü kapsar):
  python predict.py --input yeni_veri.csv --grup MASTER

Çıktı:
  outputs/tahminler_{GRUP}.csv  →  Variant_ID + Label_Pred + Prob_Patojenik

Yarışmada:
  1. Komite etiketsiz CSV verir (Label kolonu yok)
  2. Bu scripti çalıştırırsın
  3. Çıktı CSV'yi teslim edersin
"""

import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
OUT  = BASE / "outputs"
MDIR = OUT / "models"


def load_model(grup):
    path = MDIR / f"model_{grup.lower()}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Model bulunamadı: {path}\n"
            f"Önce 'python main.py' çalıştırın."
        )
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    return bundle


def preprocess_predict(df, artifacts):
    """
    Eğitimde kullanılan aynı ön işleme adımları — test verisine uygula.
    Fit işlemleri eğitim setinden (artifacts içinde saklanmış).
    """
    feature_cols = artifacts["feature_cols"]
    cat_cols     = artifacts["cat_cols"]
    encoders     = artifacts["encoders"]
    clip_vals    = artifacts["clip_vals"]
    imputer      = artifacts["imputer"]

    # Sadece bilinen kolonları al, eksikleri 0 ile doldur
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan

    X = df[feature_cols].copy()

    # Kategorik encode
    for col in cat_cols:
        if col in encoders:
            le  = encoders[col]
            vals = X[col].fillna("__NaN__").astype(str)
            # Eğitimde görülmemiş kategoriler → __NaN__ olarak işle
            known = set(le.classes_)
            vals  = vals.apply(lambda v: v if v in known else "__NaN__")
            X[col] = le.transform(vals).astype(float)

    # Sayısal dönüşüm
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Winsorize (eğitim değerleriyle)
    for col, (lower, upper) in clip_vals.items():
        if col in X.columns:
            X[col] = X[col].clip(lower, upper)

    # Impute
    X_arr = imputer.transform(X)
    X = pd.DataFrame(X_arr, columns=feature_cols)

    return X


def predict(input_path, grup, use_final_threshold=True):
    """
    Etiketsiz veriyi oku, modeli yükle, tahmin üret.

    use_final_threshold=True  → finale özel eşik (benign ağırlıklı test için)
    use_final_threshold=False → standart eşik (recall öncelikli)
    """
    print(f"\n{'='*55}")
    print(f"  TAHMİN – {grup}")
    print(f"{'='*55}")

    # Veriyi oku
    df = pd.read_csv(input_path)
    print(f"  Giriş: {len(df)} satır, {df.shape[1]} kolon")

    has_label = "Label" in df.columns
    if has_label:
        print(f"  ⚠  'Label' kolonu mevcut — değerlendirme modu aktif")

    # Model yükle
    bundle   = load_model(grup)
    models   = bundle["models"]
    artifacts = bundle["artifacts"]
    thrs     = bundle["thresholds"]

    threshold = thrs["final"] if use_final_threshold else thrs["standard"]
    thr_label = "Final Eşiği" if use_final_threshold else "Standart Eşik"
    print(f"  Kullanılan eşik : {threshold:.3f}  ({thr_label})")

    # Ön işleme
    X = preprocess_predict(df.copy(), artifacts)

    # Ensemble tahmin
    probs = np.stack([m.predict_proba(X)[:, 1] for m in models.values()])
    prob_pathogenic = probs.mean(axis=0)
    preds = (prob_pathogenic >= threshold).astype(int)

    # Çıktı oluştur
    result = pd.DataFrame({
        "Variant_ID"      : df["Variant_ID"] if "Variant_ID" in df.columns
                            else range(len(df)),
        "Label_Pred"      : preds,
        "Prob_Patojenik"  : prob_pathogenic.round(4),
        "Karar"           : ["Patojenik" if p == 1 else "Benign" for p in preds],
    })

    pat_n  = (preds == 1).sum()
    ben_n  = (preds == 0).sum()
    print(f"\n  Tahmin özeti:")
    print(f"    Patojenik : {pat_n} ({pat_n/len(preds)*100:.1f}%)")
    print(f"    Benign    : {ben_n} ({ben_n/len(preds)*100:.1f}%)")

    # Eğer Label varsa değerlendirme yap
    if has_label:
        from sklearn.metrics import (confusion_matrix, f1_score,
                                     matthews_corrcoef, roc_auc_score)
        y_true = df["Label"].values
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        recall = tp/(tp+fn) if (tp+fn)>0 else 0
        spec   = tn/(tn+fp) if (tn+fp)>0 else 0
        f1     = f1_score(y_true, preds, zero_division=0)
        mcc    = matthews_corrcoef(y_true, preds)
        auc    = roc_auc_score(y_true, prob_pathogenic)
        fn_r   = fn/(tp+fn) if (tp+fn)>0 else 0

        print(f"\n  Değerlendirme (etiket mevcut):")
        print(f"    TP={tp}  FN={fn}  TN={tn}  FP={fp}")
        print(f"    Recall={recall*100:.1f}%  FN_Rate={fn_r*100:.1f}%  "
              f"Specificity={spec*100:.1f}%")
        print(f"    F1={f1*100:.1f}%  MCC={mcc:.3f}  AUC-ROC={auc*100:.1f}%")

    # Kaydet
    out_path = OUT / f"tahminler_{grup.lower()}.csv"
    result.to_csv(out_path, index=False)
    print(f"\n  ✓ Tahminler kaydedildi: {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="TEKNOFEST 2026 – Varyant Patojenite Tahmin"
    )
    parser.add_argument(
        "--input", required=True,
        help="Etiketsiz CSV dosyası (yarışma gününde verilen veri)"
    )
    parser.add_argument(
        "--grup", required=True,
        choices=["MASTER", "KANSER", "CFTR", "PAH"],
        help="Hangi modeli kullanacaksın?"
    )
    parser.add_argument(
        "--standart", action="store_true",
        help="Final eşiği yerine standart eşiği kullan (varsayılan: final eşiği)"
    )
    args = parser.parse_args()

    result = predict(
        input_path=args.input,
        grup=args.grup,
        use_final_threshold=not args.standart
    )

    print(f"\n  İlk 5 tahmin:")
    print(result.head().to_string(index=False))
    print()


if __name__ == "__main__":
    main()