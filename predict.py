# -*- coding: utf-8 -*-
"""
Egitilmis XGBoost modeli ile gelecekteki anomalileri tahmin eder.

Kullanim:
    py predict.py <log_verisi.csv> [cikti.csv]

CSV en az su sutunlari icermelidir: timestamp (unix ms), is_alert, event_id, template
(data/parsed.csv ile ayni format). Verinin son penceresi "guncel pencere" kabul edilir;
model bir SONRAKI pencerede (varsayilan 15 dk) anomali olma olasiligini hesaplar.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import xgboost as xgb

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.json"
META_PATH = HERE / "metadata.json"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    csv_path = Path(sys.argv[1])
    out_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["feature_cols"]
    threshold = float(meta["threshold"])

    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)

    from train_model import load_data, build_window_grid
    df = load_data()
    if csv_path.resolve() != (HERE.parent / "data" / "parsed.csv").resolve():
        df = pd.read_csv(csv_path)
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    win, _ = build_window_grid(df)
    row = win[feature_cols].iloc[[-1]]
    if row.isna().to_numpy().any():
        print("UYARI: Yeterli gecmis yok (en az 8 pencere = 2 saatlik veri gerekir); eksik ozellikler 0 ile dolduruluyor.")
        row = row.fillna(0)

    prob = float(model.predict_proba(row)[0, 1])
    ts = win.index[-1]
    print(f"Guncel pencere          : {ts}")
    print(f"Olay sayisi             : {int(win['total_events'].iloc[-1])}")
    print(f"Sonraki pencere anomali P.: {prob:.4f}")
    print(f"Esik ({threshold:.2f}) karari   : {'ANOMALI BEKLENIYOR' if prob >= threshold else 'normal'}")

    if out_csv:
        pd.DataFrame({"window_end": [str(ts)], "anomaly_probability": [prob]}).to_csv(out_csv, index=False)
        print(f"Sonuc kaydedildi: {out_csv}")


if __name__ == "__main__":
    main()
