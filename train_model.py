# -*- coding: utf-8 -*-
"""
BGL log verisinden 15 dakikalik pencereler olusturur ve gelecekteki (bir sonraki
pencerenin) anomalilerini tahmin etmek icin XGBoost modeli egitir.

Girdi : data/parsed.csv  -> timestamp, is_alert, event_id, template
Cikti : XGboost/model.json, metadata.json, metrics.json, feature_importance.csv
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

BASE = Path(r"C:\Users\selcu\OneDrive\Masaüstü\Sofito_bireysel_proje")
DATA_PATH = BASE / "data" / "parsed.csv"
OUT_DIR = BASE / "XGboost"

WINDOW_FREQ = "15min"
TOP_K_EVENTS = 50
LAGS = [1, 2, 4, 8]          # 15dk / 30dk / 1sa / 2sa onceki pencereler
ROLL_WINDOWS = [4, 16, 96]   # son 1sa / 4sa / 24sa hareketli toplamlar
ROLL_STD_WINDOWS = [3, 5, 10]  # dalgalanma (std/z-score) pencereleri
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.sort_values("dt")


def build_window_grid(df):
    """Tum 15 dakikalik izgara uzerinde ozellikleri uretir (hedef ve filtreleme haric)."""
    # Tum pencere izgarasi (bos pencereler sifir ile doldurulur -> lag/hedef anlamli kalir)
    win = df.set_index("dt").resample(WINDOW_FREQ).agg(
        total_events=("is_alert", "size"),
        alert_count=("is_alert", "sum"),
        n_unique_events=("event_id", "nunique"),
    ).fillna(0)

    # event_id bazli sayim matrisi
    counts = (
        df.groupby([pd.Grouper(key="dt", freq=WINDOW_FREQ), "event_id"])
        .size()
        .unstack(fill_value=0)
    )
    counts = counts.reindex(win.index, fill_value=0)
    top_event_ids = counts.sum(axis=0).sort_values(ascending=False).head(TOP_K_EVENTS).index.tolist()
    for eid in top_event_ids:
        col = f"ev_{eid}"
        win[col] = np.log1p(counts[eid])

    # Mevcut pencere agrega ozellikleri (hedef bir sonraki pencere oldugu icin sizinti yok)
    win["cur_total_log"] = np.log1p(win["total_events"])
    win["cur_alerts"] = win["alert_count"]
    win["cur_alert_ratio"] = win["alert_count"] / win["total_events"].clip(lower=1)
    win["cur_unique"] = win["n_unique_events"]

    # Son anomaliden bu yana gecen pencere sayisi (patlamasal/demet davranis icin guclu sinyal)
    alert_flag = (win["alert_count"] > 0).astype(int)
    grp = alert_flag.cumsum()
    win["steps_since_alert"] = alert_flag.groupby(grp).cumcount()
    win.loc[alert_flag == 1, "steps_since_alert"] = 0

    # Zaman gecikme (lag) ve hareketli ortalamalar
    for lag in LAGS:
        win[f"total_lag_{lag}"] = np.log1p(win["total_events"].shift(lag))
        win[f"alerts_lag_{lag}"] = win["alert_count"].shift(lag)
    for rw in ROLL_WINDOWS:
        shifted = win["alert_count"].shift(1)
        win[f"alerts_rollsum_w{rw}"] = shifted.rolling(rw, min_periods=1).sum()
        win[f"total_rollmean_w{rw}"] = (
            win["total_events"].shift(1).rolling(rw, min_periods=1).mean()
        )

    # Dalgalanma ozellikleri: pencere farki, hareketli std, z-score ve alert oran degisimi
    # (anomaliler genellikle son pencerelere gore sapmalarda gizlidir)
    win["total_diff_1"] = win["total_events"].diff(1)
    win["alerts_diff_1"] = win["alert_count"].diff(1)
    for rw in ROLL_STD_WINDOWS:
        base_m = win["total_events"].shift(1).rolling(rw, min_periods=2).mean()
        base_s = win["total_events"].shift(1).rolling(rw, min_periods=2).std()
        win[f"total_rstd_w{rw}"] = base_s
        win[f"total_z_w{rw}"] = (win["total_events"] - base_m) / (base_s + 1.0)
        prev_alerts = win["alert_count"].shift(1).rolling(rw, min_periods=1).sum()
        win[f"alerts_rate_w{rw}"] = win["alert_count"] / (prev_alerts + 1.0)

    # Takvim ozellikleri
    idx = win.index
    minute_of_day = idx.hour.to_numpy() * 60 + idx.minute.to_numpy()
    dow = idx.dayofweek.to_numpy()
    win["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    win["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    win["dow"] = dow

    return win, top_event_ids


def build_window_features(df):
    win, top_event_ids = build_window_grid(df)

    # Hedef: bir sonraki pencerede anomali var mi? (gelecek tahmini -> sizinti yok)
    win["target"] = (win["alert_count"].shift(-1) > 0).astype(int)

    # Tahmin icin hicbir verinin olmadigi (bos) pencereler egitim disi
    win = win[win["total_events"] > 0]
    win = win.dropna(subset=[f"total_lag_{max(LAGS)}"]).dropna(subset=["target"])
    return win, top_event_ids


def make_model(spw, n_estimators=None):
    return xgb.XGBClassifier(
        n_estimators=n_estimators or 1500,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        reg_lambda=5.0,
        scale_pos_weight=spw,
        eval_metric=["aucpr", "auc"],
        early_stopping_rounds=None if n_estimators else 150,
        random_state=RANDOM_STATE,
        tree_method="hist",
        n_jobs=-1,
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Veri yukleniyor...")
    df = load_data()
    print(f"Satir sayisi: {len(df):,} | Aralik: {df['dt'].min()} -> {df['dt'].max()}")

    win, top_event_ids = build_window_features(df)
    feature_cols = [
        c for c in win.columns
        if c not in ("target", "total_events", "alert_count", "n_unique_events")
    ]
    X = win[feature_cols]
    y = win["target"]
    print(f"Pencere sayisi: {len(win)} | Pozitif oran: {y.mean():.4f}")

    # Kronolojik bolme (zaman serisi -> karistirma yok)
    n = len(win)
    i_train = int(n * 0.70)
    i_val = int(n * 0.85)
    X_tr, y_tr = X.iloc[:i_train], y.iloc[:i_train]
    X_va, y_va = X.iloc[i_train:i_val], y.iloc[i_train:i_val]
    X_te, y_te = X.iloc[i_val:], y.iloc[i_val:]
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    print(f"Egitim/Validasyon/Test: {len(X_tr)}/{len(X_va)}/{len(X_te)} | scale_pos_weight={spw:.2f}")

    model = make_model(spw)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_va, y_va)],
        verbose=False,
    )
    best_it = int(model.best_iteration)
    print(f"En iyi iterasyon: {best_it}")

    # Esik degerini validasyon setinde F1'e gore sec (yeniden egitimden ONCE, sizinti olmasin)
    # Guvenlik kisiti: tahmin edilen pozitif orani, gercek pozitif oranin cok uzerine
    # cikamaz -> "her seyi anomali isaretle" dejenerasyonunu engeller.
    val_prob = model.predict_proba(X_va)[:, 1]
    print(f"Val ROC-AUC: {roc_auc_score(y_va, val_prob):.4f} | Val PR-AUC: {average_precision_score(y_va, val_prob):.4f}")
    thresholds = np.linspace(0.05, 0.95, 181)
    max_pred_rate = max(0.10, 2 * float(y_va.mean()))
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = val_prob >= t
        if float(pred.mean()) > max_pred_rate:
            continue
        f1 = f1_score(y_va, pred, zero_division=0)
        if f1 >= best_f1:
            best_f1, best_t = f1, float(t)

    # Erken durdurmada bulunan iterasyon sayisiyla train+val uzerine nihai yeniden egitim
    final_model = make_model(spw, n_estimators=best_it)
    X_trva = pd.concat([X_tr, X_va])
    y_trva = pd.concat([y_tr, y_va])
    final_model.fit(X_trva, y_trva, eval_set=[(X_trva, y_trva)], verbose=False)
    model = final_model

    test_prob = model.predict_proba(X_te)[:, 1]
    test_pred = (test_prob >= best_t).astype(int)
    metrics = {
        "window": WINDOW_FREQ,
        "horizon": f"1 pencere = {WINDOW_FREQ} (bir sonraki pencerede anomali olasiligi)",
        "best_iteration": best_it,
        "threshold": best_t,
        "val_f1": float(best_f1),
        "test_roc_auc": float(roc_auc_score(y_te, test_prob)),
        "test_pr_auc": float(average_precision_score(y_te, test_prob)),
        "test_precision": float(precision_score(y_te, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_te, test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_te, test_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_te, test_pred).tolist(),
        "train_rows": len(X_tr), "val_rows": len(X_va), "test_rows": len(X_te),
    }
    report = classification_report(y_te, test_pred, target_names=["normal", "anomali"], zero_division=0)
    cm = metrics["confusion_matrix"]
    print("\n=== TEST SONUCLARI ===")
    print(f"ROC-AUC: {metrics['test_roc_auc']:.4f} | PR-AUC: {metrics['test_pr_auc']:.4f}")
    print(f"Esik={best_t:.2f} | Precision: {metrics['test_precision']:.4f} | Recall: {metrics['test_recall']:.4f} | F1: {metrics['test_f1']:.4f}")
    print(f"Karar matrisi [TN,FP;FN,TP]: {cm}")
    print(report)

    # Kaydet
    model_path = OUT_DIR / "model.json"
    model.save_model(model_path)

    gain = model.get_booster().get_score(importance_type="gain")
    imp = pd.DataFrame(
        [(f, float(gain.get(f, 0.0))) for f in feature_cols],
        columns=["feature", "gain"],
    )
    imp.sort_values("gain", ascending=False).to_csv(OUT_DIR / "feature_importance.csv", index=False)

    with open(OUT_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "feature_cols": feature_cols,
            "top_event_ids": [int(e) for e in top_event_ids],
            "threshold": best_t,
            "window_freq": WINDOW_FREQ,
            "params": model.get_params(),
            "best_iteration": best_it,
        }, f, ensure_ascii=False, indent=2, default=str)

    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\nModel kaydedildi: {model_path}")


if __name__ == "__main__":
    main()
