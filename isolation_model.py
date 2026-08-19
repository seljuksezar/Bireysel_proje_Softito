"""Isolation Forest anomaly detection on parsed BGL logs.

Uses sliding-window features (event frequency, alert ratio, entropy, etc.)
to train an unsupervised Isolation Forest model. Evaluated against known
alert labels as a sanity check.

Usage:
    python isolation_model.py                          # default config
    python isolation_model.py --config config.yaml     # custom config
    python isolation_model.py --contamination 0.05     # custom contamination
"""

import argparse
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def build_features(
    df,
    window_seconds=900,
    step_seconds=900,
    horizon_seconds=900,
    max_seq_len=50,
    min_events_per_window=5,
    vocab_size=5001,
):
    """Build feature vectors from parsed BGL logs.

    For each sliding window, computes:
    - Event frequency histogram (vocab_size dims)
    - Alert ratio in window
    - Number of unique events
    - Sequence length
    - Top-10 event concentration
    - Entropy of event distribution
    - Max single event frequency
    - Label: 1 if alert in next horizon_seconds, else 0
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"].to_numpy()
    events = df["event_id"].to_numpy()
    alerts = df["is_alert"].to_numpy()

    window_ms = window_seconds * 1000
    step_ms = step_seconds * 1000
    horizon_ms = horizon_seconds * 1000

    start = ts[0]
    end = ts[-1]

    rows = []
    labels = []
    starts = []
    prev = None

    for w_start in range(int(start), int(end) + 1, step_ms):
        w_end = w_start + window_ms
        left = int(np.searchsorted(ts, w_start, side="left"))
        if prev is not None and left > prev:
            prev = left
        right = int(np.searchsorted(ts, w_end, side="left"))

        if prev is None:
            prev = left
        if right <= prev:
            continue

        window_events = events[prev:right]
        window_alerts = alerts[prev:right]
        if len(window_events) < min_events_per_window:
            prev = right
            continue

        seq = window_events[:max_seq_len]
        valid = seq[seq > 0]

        # event frequency histogram
        freq = np.zeros(vocab_size, dtype=np.float32)
        if len(valid) > 0:
            counts = Counter(valid.tolist())
            for eid, cnt in counts.items():
                if eid < vocab_size:
                    freq[eid] = cnt / len(valid)

        # meta features
        alert_ratio = float(window_alerts.mean())
        n_unique = len(set(valid.tolist())) if len(valid) > 0 else 0
        seq_len = len(valid)

        if len(valid) > 0:
            top10 = sorted(counts.values(), reverse=True)[:10]
            top10_conc = sum(top10) / len(valid)
            probs = np.array(list(counts.values())) / len(valid)
            entropy = float(-np.sum(probs * np.log2(probs + 1e-10)))
            max_freq = max(counts.values()) / len(valid)
        else:
            top10_conc = 0.0
            entropy = 0.0
            max_freq = 0.0

        meta = np.array([alert_ratio, n_unique, seq_len, top10_conc, entropy, max_freq], dtype=np.float32)
        feature_vec = np.concatenate([freq, meta])

        rows.append(feature_vec)
        starts.append(w_start)

        # label
        h_left = int(np.searchsorted(ts, w_end, side="left"))
        h_right = int(np.searchsorted(ts, w_end + horizon_ms, side="right"))
        label = int(alerts[h_left:h_right].sum() > 0) if h_left < h_right else 0
        labels.append(label)

        prev = right

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    return X, y, starts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--contamination", type=float, default=None,
                        help="Fraction of anomalies (None=auto)")
    parser.add_argument("--output", default="logs/isolation_model.pkl")
    parser.add_argument("--n-trees", type=int, default=200)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    win_cfg = cfg["window"]

    # load parsed data
    print("[data] loading parsed logs...")
    df = pd.read_csv(data_cfg["parsed_path"])
    print(f"  {len(df)} lines, {df['event_id'].nunique()} events")
    print(f"  alert ratio: {df['is_alert'].mean():.2%}")

    vocab_size = int(df["event_id"].max()) + 2

    # build features
    print("[feature] building sliding-window features...")
    X, y, starts = build_features(
        df,
        window_seconds=win_cfg["window_seconds"],
        step_seconds=win_cfg["step_seconds"],
        horizon_seconds=win_cfg["horizon_seconds"],
        max_seq_len=win_cfg["max_seq_len"],
        min_events_per_window=win_cfg["min_events_per_window"],
        vocab_size=vocab_size,
    )
    print(f"  {X.shape[0]} windows, {X.shape[1]} features")
    print(f"  positive labels: {y.sum()} ({y.mean()*100:.2f}%)")

    # temporal split (80/20)
    split = int(len(X) * 0.8)
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]
    print(f"  train: {len(Xtr)}, test: {len(Xte)}")

    # scale features
    print("[model] training Isolation Forest...")
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    contamination = args.contamination
    if contamination is None:
        contamination = 0.06#ytr.mean() if ytr.mean() > 0 else 0.05
        print(f"  auto contamination: {contamination:.4f}")

    iso = IsolationForest(
        n_estimators=args.n_trees,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(Xtr_s)

    # predictions: -1 = anomaly, 1 = normal
    pred_raw = iso.predict(Xte_s)
    scores = iso.score_samples(Xte_s)

    # convert to 0/1 (1 = anomaly)
    pred_anomaly = (pred_raw == -1).astype(int)

    # evaluation against known labels
    print("\n=== Test metrics (anomaly detection) ===")
    print(f"  accuracy : {accuracy_score(yte, pred_anomaly):.4f}")
    p, r, f1, _ = precision_recall_fscore_support(yte, pred_anomaly, zero_division=0)
    if p.shape[0] > 1:
        print(f"  precision: {p[1]:.4f}")
        print(f"  recall   : {r[1]:.4f}")
        print(f"  F1       : {f1[1]:.4f}")
    else:
        print(f"  precision: {p[0]:.4f}")
        print(f"  recall   : {r[0]:.4f}")
        print(f"  F1       : {f1[0]:.4f}")

    try:
        print(f"  ROC-AUC  : {roc_auc_score(yte, -scores):.4f}")
    except ValueError as e:
        print(f"  ROC-AUC  : n/a ({e})")

    print(f"\n  predicted anomalies: {pred_anomaly.sum()} / {len(pred_anomaly)}")
    print(f"  actual anomalies:    {yte.sum()} / {len(yte)}")

    # threshold analysis
    print("\n=== Threshold analysis (anomaly score) ===")
    print("%-12s %-10s %-10s %-10s" % ("Threshold", "Precision", "Recall", "F1"))
    for t in np.percentile(-scores, [1, 5, 10, 15, 20, 25, 50]):
        preds_t = (-scores >= t).astype(int)
        p2, r2, f2, _ = precision_recall_fscore_support(yte, preds_t, zero_division=0)
        if p2.shape[0] > 1:
            print("%-12.4f %-10.4f %-10.4f %-10.4f" % (t, p2[1], r2[1], f2[1]))

    print("\n" + classification_report(yte, pred_anomaly, target_names=["normal", "anomaly"], zero_division=0))

    # save model
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    artifact = {
        "model": iso,
        "scaler": scaler,
        "config": cfg,
        "vocab_size": vocab_size,
        "contamination": contamination,
    }
    with open(args.output, "wb") as f:
        pickle.dump(artifact, f)
    print(f"[done] model saved -> {args.output}")


if __name__ == "__main__":
    main()
