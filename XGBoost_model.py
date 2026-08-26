"""Train an XGBoost classifier on BGL parsed logs (data/parsed.csv)."""

import argparse
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


RAW_META_CACHE = "data/raw_meta.csv.gz"


def ensure_raw_meta(raw_path, cache_path=RAW_META_CACHE):
    if os.path.exists(cache_path):
        meta = pd.read_csv(cache_path, dtype=str).fillna("")
        meta["timestamp"] = meta["timestamp"].astype(np.int64)
        return meta
    if not os.path.exists(raw_path):
        print(f"[meta] no cache and no raw log ({raw_path}); meta features skipped")
        return None
    from build_meta import extract_meta

    meta = extract_meta(raw_path)
    meta.to_csv(cache_path, index=False)
    print(f"[meta] built cache -> {cache_path}")
    return meta


def align_meta(sorted_parsed, meta):
    """Align raw-log meta rows to parsed rows by timestamp + within-group order."""
    raw_ts = meta["timestamp"].to_numpy()
    uniq, first_idx, counts = np.unique(raw_ts, return_index=True, return_counts=True)
    ts = sorted_parsed["timestamp"].to_numpy()
    grp = pd.Series(ts).groupby(ts).cumcount().to_numpy()
    pos = np.clip(np.searchsorted(uniq, ts), 0, len(uniq) - 1)
    ok = uniq[pos] == ts
    slot = np.where(ok, first_idx[pos] + np.minimum(grp, counts[pos] - 1), -1)
    last_ok = np.maximum.accumulate(np.where(ok, np.arange(len(ok)), -1))
    slot = np.where(ok, slot, np.where(last_ok >= 0, slot[np.maximum(last_ok, 0)], 0))
    n_bad = int((~ok).sum())
    if n_bad:
        print(f"[meta] {n_bad} rows without exact ts match (nearest used)")
    out = {}
    for col in ["label", "node", "subsystem", "severity"]:
        out[col] = meta[col].to_numpy(dtype=object)[slot]
    return out


TOP_TAGS = 15
TOP_SUBSYSTEMS = 10
TOP_SEVERITIES = 6


def compute_episodes(df_sorted, gap_ms=3600_000):
    """Group alert lines into episodes separated by > gap_ms of silence."""
    ts = df_sorted["timestamp"].to_numpy()
    alert_ts = ts[df_sorted["is_alert"].to_numpy() == 1]
    if len(alert_ts) == 0:
        return np.empty((0, 2), dtype=np.int64)
    breaks = np.flatnonzero(np.diff(alert_ts) > gap_ms)
    ep_starts = np.r_[alert_ts[0], alert_ts[breaks + 1]]
    ep_ends = np.r_[alert_ts[breaks], alert_ts[-1]]
    return np.column_stack([ep_starts, ep_ends])


def _window_in_episode(episodes, w_start, w_end):
    j = int(np.searchsorted(episodes[:, 0], w_start, side="right")) - 1
    if j >= 0 and episodes[j, 1] >= w_start:
        return True
    k = j + 1
    return k < len(episodes) and episodes[k, 0] <= w_end


def _onset_label(episodes, w_end, horizon_ms):
    e = int(np.searchsorted(episodes[:, 0], w_end, side="right"))
    return int(e < len(episodes) and episodes[e, 0] <= w_end + horizon_ms)


def _categorical_codes(values, top_index):
    mapping = {name: i for i, name in enumerate(top_index)}
    return (
        pd.Series(values).map(mapping).fillna(-1).to_numpy(dtype=np.int64),
        len(top_index),
    )


def _window_bincount(codes, left, right, n_cats):
    seg = codes[left:right]
    if len(seg) == 0:
        return np.zeros(n_cats, dtype=int)
    bc = np.bincount(seg[seg >= 0] + 1, minlength=n_cats + 1)
    return bc[1:]


def build_tabular_windows(
    parsed,
    cfg,
    meta=None,
    label_mode="onset",
    episode_gap_s=3600,
    keep_in_episode=False,
):
    win_cfg = cfg["window"]
    window_ms = win_cfg["window_seconds"] * 1000
    step_ms = win_cfg["step_seconds"] * 1000
    horizon_ms = win_cfg["horizon_seconds"] * 1000
    min_events = win_cfg.get("min_events_per_window", 1)

    df = parsed.sort_values("timestamp", kind="stable").reset_index(drop=True)
    ts = df["timestamp"].to_numpy()
    events = df["event_id"].to_numpy()
    alerts = df["is_alert"].to_numpy()

    tag_codes = sub_codes = sev_codes = None
    node_arr = None
    if meta is not None:
        aligned = align_meta(df, meta)
        label_s = pd.Series(aligned["label"])
        top_tags = label_s[label_s != "-"].value_counts().head(TOP_TAGS).index.to_numpy()
        top_subs = pd.Series(aligned["subsystem"]).value_counts().head(TOP_SUBSYSTEMS).index.to_numpy()
        top_sevs = pd.Series(aligned["severity"]).value_counts().head(TOP_SEVERITIES).index.to_numpy()
        tag_codes, n_tags = _categorical_codes(aligned["label"], top_tags)
        sub_codes, n_subs = _categorical_codes(aligned["subsystem"], top_subs)
        sev_codes, n_sevs = _categorical_codes(aligned["severity"], top_sevs)
        node_arr = aligned["node"]
        print(
            f"[features] meta: {n_tags} tags, {n_subs} subsystems, "
            f"{n_sevs} severities, {pd.Series(node_arr).nunique()} nodes"
        )

    total_lines = len(df)
    ev_counts = df["event_id"].value_counts()
    row_count = pd.Series(events).map(ev_counts).to_numpy(dtype=float)
    row_idf = np.log(total_lines / row_count)
    rare_mask = (row_count / total_lines) < 1e-4

    alert_ev_freq = df.loc[df["is_alert"] == 1, "event_id"].value_counts()
    top_alert_events = alert_ev_freq.head(50).index.to_numpy()
    ae_pos = pd.Series(np.arange(len(top_alert_events)), index=top_alert_events)
    row_ae_idx = (
        pd.Series(events).map(ae_pos).fillna(-1).to_numpy(dtype=np.int64)
    )
    print(f"[features] {len(top_alert_events)} alert-event counters added")

    t0 = ts[0]
    last_alert_ts = -1
    hist_totals = []
    hist_alerts = []
    node_last_alert = {}

    episodes = compute_episodes(df, gap_ms=episode_gap_s * 1000)
    print(
        f"[episodes] {len(episodes)} alert episodes "
        f"(gap>{episode_gap_s}s), mode={label_mode}, "
        f"{'keeping' if keep_in_episode else 'dropping'} in-episode windows"
    )
    if len(episodes) == 0:
        raise SystemExit("[error] no alert lines found; cannot build labels")

    n_dropped = 0
    feature_rows = []
    labels = []
    starts = []

    for w_start in range(int(t0), int(ts[-1]) + 1, step_ms):
        w_end = w_start + window_ms
        left = int(np.searchsorted(ts, w_start, side="left"))
        right = int(np.searchsorted(ts, w_end, side="left"))
        n_events = right - left
        if n_events < min_events:
            continue

        if not keep_in_episode and _window_in_episode(episodes, w_start, w_end):
            n_dropped += 1
            continue

        if label_mode == "onset":
            label = _onset_label(episodes, w_end, horizon_ms)
        else:
            h_left = int(np.searchsorted(ts, w_end, side="left"))
            h_right = int(np.searchsorted(ts, w_end + horizon_ms, side="right"))
            label = int(alerts[h_left:h_right].sum() > 0)
        labels.append(label)
        starts.append(w_start)

        w_events = events[left:right]
        w_alert_flags = alerts[left:right]
        n_alerts = int(w_alert_flags.sum())
        if n_alerts > 0:
            alert_rel = np.flatnonzero(w_alert_flags)
            last_alert_pos = left + int(alert_rel[-1])
            last_alert_ts = ts[last_alert_pos]
            if node_arr is not None:
                for nd in np.unique(node_arr[left:right][w_alert_flags == 1]):
                    node_last_alert[nd] = w_end

        vals, counts = np.unique(w_events, return_counts=True)
        probs = counts / n_events
        entropy = float(-(probs * np.log(probs)).sum())

        gaps = np.diff(ts[left:right]) / 1000.0
        if len(gaps) > 0:
            gap_mean, gap_std, gap_min, gap_max = (
                float(gaps.mean()),
                float(gaps.std()),
                float(gaps.min()),
                float(gaps.max()),
            )
        else:
            gap_mean = gap_std = gap_min = gap_max = 0.0

        dt = pd.to_datetime(w_start, unit="ms", utc=True)
        hour = dt.hour + dt.minute / 60.0
        dow = dt.dayofweek

        lag1_total = hist_totals[-1] if hist_totals else 0
        lag1_alerts = hist_alerts[-1] if hist_alerts else 0
        recent_totals = hist_totals[-4:]
        recent_alerts = hist_alerts[-4:]

        feat = {
            "total_events": n_events,
            "unique_events": len(vals),
            "alerts_in_window": n_alerts,
            "alert_ratio": n_alerts / n_events,
            "secs_since_last_alert": (
                (w_end - last_alert_ts) / 1000.0 if last_alert_ts >= 0 else -1.0
            ),
            "event_rate": n_events / win_cfg["window_seconds"],
            "event_entropy": entropy,
            "top_event_share": float(counts.max() / n_events),
            "surprise_score": float(row_idf[left:right].mean()),
            "rare_event_count": int(rare_mask[left:right].sum()),
            "gap_mean_s": gap_mean,
            "gap_std_s": gap_std,
            "gap_min_s": gap_min,
            "gap_max_s": gap_max,
            "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
            "dow_sin": float(np.sin(2 * np.pi * dow / 7)),
            "dow_cos": float(np.cos(2 * np.pi * dow / 7)),
            "lag1_total_events": lag1_total,
            "lag1_alerts": lag1_alerts,
            "delta_total_events": n_events - lag1_total,
            "roll4_total_mean": float(np.mean(recent_totals)) if recent_totals else 0.0,
            "roll4_total_std": float(np.std(recent_totals)) if len(recent_totals) > 1 else 0.0,
            "roll4_alerts_sum": int(np.sum(recent_alerts)),
            "roll4_alerts_max": int(np.max(recent_alerts)) if recent_alerts else 0,
        }
        feat.update({f"ev_{int(v)}": c for v, c in zip(vals, counts)})

        ae_idx = row_ae_idx[left:right]
        ae_idx = ae_idx[ae_idx >= 0]
        if len(ae_idx) > 0:
            ae_counts = np.bincount(ae_idx, minlength=len(top_alert_events))
        else:
            ae_counts = np.zeros(len(top_alert_events), dtype=int)
        feat.update(
            {
                f"alertev_{int(top_alert_events[j])}": int(c)
                for j, c in enumerate(ae_counts)
                if c > 0
            }
        )

        if meta is not None:
            for names, codes, n_cats, prefix in (
                (top_tags, tag_codes, n_tags, "tag"),
                (top_subs, sub_codes, n_subs, "sys"),
                (top_sevs, sev_codes, n_sevs, "sev"),
            ):
                cnts = _window_bincount(codes, left, right, n_cats)
                feat.update(
                    {f"{prefix}_{names[j]}": int(c) for j, c in enumerate(cnts) if c > 0}
                )

            w_nodes = node_arr[left:right]
            uq_nodes, node_cnts = np.unique(w_nodes, return_counts=True)
            hot_cutoff = w_start - 3600_000
            hot = [
                nd
                for nd in uq_nodes
                if node_last_alert.get(nd, -1) >= hot_cutoff
            ]
            hot_mask = np.isin(w_nodes, hot) if hot else np.zeros(n_events, bool)
            feat["n_unique_nodes"] = len(uq_nodes)
            feat["node_max_share"] = float(node_cnts.max() / n_events)
            feat["hot_nodes"] = len(hot)
            feat["hot_node_events"] = int(hot_mask.sum())

        feature_rows.append(feat)

        hist_totals.append(n_events)
        hist_alerts.append(n_alerts)

    X = pd.DataFrame(feature_rows).fillna(0)
    y = np.array(labels, dtype=int)
    print(
        f"[window] {len(y)} windows ({n_dropped} in-episode dropped), "
        f"{y.mean()*100:.2f}% positive, {X.shape[1]} features"
    )
    return X, y, starts


def temporal_split(X, y, val_ratio, test_ratio, purge_windows=0):
    n = len(y)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    p = max(int(purge_windows), 0)
    tr_end = n_train - p
    va_end = n_train + n_val - p
    return (
        (X.iloc[:tr_end], y[:tr_end]),
        (X.iloc[n_train:va_end], y[n_train:va_end]),
        (X.iloc[va_end:], y[va_end:]),
    )


def report(name, y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, probs)
        metrics["pr_auc"] = average_precision_score(y_true, probs)
    print(f"[{name}] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    print(confusion_matrix(y_true, preds))
    return metrics


def fit_model(Xtr, ytr, seed, Xva=None, yva=None, n_estimators=500):
    n_pos = max(int(np.sum(ytr)), 1)
    n_neg = len(ytr) - n_pos
    use_es = Xva is not None
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_lambda=1.0,
        tree_method="hist",
        eval_metric=["logloss", "aucpr"],
        early_stopping_rounds=50 if use_es else None,
        random_state=seed,
        n_jobs=-1,
        scale_pos_weight=n_neg / n_pos,
    )
    if use_es:
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    else:
        model.fit(Xtr, ytr, verbose=False)
    return model


def best_f1_threshold(y_true, probs):
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = [
        f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        for t in thresholds
    ]
    return float(thresholds[int(np.argmax(f1s))]), float(max(f1s))


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost on parsed.csv")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--csv", default=None, help="Override parsed.csv path")
    parser.add_argument("--window", type=int, default=None,
                        help="Override window_seconds from config")
    parser.add_argument("--step", type=int, default=None,
                        help="Override step_seconds from config")
    parser.add_argument("--horizon", type=int, default=None,
                        help="Override horizon_seconds from config")
    parser.add_argument("--raw", default=None, help="Override raw BGL.log path")
    parser.add_argument("--no-meta", action="store_true",
                        help="Skip raw-log meta features")
    parser.add_argument("--label-mode", choices=["onset", "any"], default="onset",
                        help="onset: predict next episode start; any: legacy alert label")
    parser.add_argument("--episode-gap", type=int, default=3600,
                        help="Silence (seconds) separating two alert episodes")
    parser.add_argument("--keep-in-episode", action="store_true",
                        help="Keep windows that overlap an alert episode")
    parser.add_argument("--cv", type=int, default=0,
                        help="Rolling-origin CV folds (0 = single split)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.window is not None:
        cfg["window"]["window_seconds"] = args.window
    if args.step is not None:
        cfg["window"]["step_seconds"] = args.step
    if args.horizon is not None:
        cfg["window"]["horizon_seconds"] = args.horizon
    wcfg = cfg["window"]
    print(
        f"[window cfg] window={wcfg['window_seconds']}s step={wcfg['step_seconds']}s "
        f"horizon={wcfg['horizon_seconds']}s"
    )
    csv_path = args.csv or cfg["data"]["parsed_path"]
    if not os.path.exists(csv_path):
        raise SystemExit(f"[error] parsed file not found: {csv_path}")

    parsed = pd.read_csv(csv_path)
    if parsed.empty:
        raise SystemExit(f"[error] parsed file is empty: {csv_path}")
    print(f"[data] {len(parsed)} log lines")

    meta = None
    if not args.no_meta:
        raw_path = args.raw or cfg["data"].get("raw_path", "data/BGL.log")
        meta = ensure_raw_meta(raw_path)

    X, y, _ = build_tabular_windows(
        parsed,
        cfg,
        meta=meta,
        label_mode=args.label_mode,
        episode_gap_s=args.episode_gap,
        keep_in_episode=args.keep_in_episode,
    )

    seed = cfg["train"]["seed"]
    wcfg = cfg["window"]
    purge = int(np.ceil((wcfg["window_seconds"] + wcfg["horizon_seconds"])
                        / max(wcfg["step_seconds"], 1)))

    if args.cv > 0:
        n = len(y)
        fold_size = n // (args.cv + 1)
        roc_scores, pr_scores = [], []
        for k in range(1, args.cv + 1):
            tr_end = k * fold_size - purge
            te_start = k * fold_size
            te_end = (k + 1) * fold_size if k < args.cv else n
            Xtr, ytr = X.iloc[:tr_end], y[:tr_end]
            Xte, yte = X.iloc[te_start:te_end], y[te_start:te_end]
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                print(f"[cv {k}] skipped (single class)")
                continue
            model = fit_model(Xtr, ytr, seed, n_estimators=200)
            probs = model.predict_proba(Xte)[:, 1]
            roc = roc_auc_score(yte, probs)
            pr = average_precision_score(yte, probs)
            t, f1b = best_f1_threshold(yte, probs)
            preds = (probs >= t).astype(int)
            prec = precision_score(yte, preds, zero_division=0)
            rec = recall_score(yte, preds, zero_division=0)
            print(
                f"[cv {k}] train={len(ytr)} test={len(yte)} pos={int(yte.sum())} | "
                f"roc={roc:.4f} pr={pr:.4f} P={prec:.3f} R={rec:.3f} F1={f1b:.3f}"
            )
            roc_scores.append(roc)
            pr_scores.append(pr)
        print(
            f"[cv] mean roc={np.mean(roc_scores):.4f}±{np.std(roc_scores):.4f} "
            f"pr={np.mean(pr_scores):.4f}±{np.std(pr_scores):.4f}"
        )
        print("[done] cv mode; no model saved")
        return

    (Xtr, ytr), (Xva, yva), (Xte, yte) = temporal_split(
        X, y, cfg["train"]["val_ratio"], cfg["train"]["test_ratio"],
        purge_windows=purge,
    )
    print(
        f"[split] train={len(ytr)} val={len(yva)} test={len(yte)} "
        f"(purge={purge} windows between slices)"
    )
    if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
        raise SystemExit("[error] train/validation split has a single class")

    model = fit_model(Xtr, ytr, seed)
    best_it = getattr(model, "best_iteration", None)
    print(f"[train] best iteration={best_it}")

    val_probs = model.predict_proba(Xva)[:, 1]
    best_t, f1b = best_f1_threshold(yva, val_probs)
    print(f"[threshold] best F1={f1b:.4f} at t={best_t:.3f}")

    test_probs = model.predict_proba(Xte)[:, 1]
    report("test", yte, test_probs, threshold=best_t)

    os.makedirs("logs", exist_ok=True)
    model.save_model("logs/xgboost_model.json")
    importances = sorted(
        zip(model.feature_names_in_, model.feature_importances_),
        key=lambda kv: kv[1],
        reverse=True,
    )
    print("[importance] top 15:")
    for name, imp in importances[:15]:
        print(f"  {name}: {imp:.4f}")
    pd.DataFrame(importances, columns=["feature", "importance"]).to_csv(
        "logs/xgboost_importances.csv", index=False
    )
    print("[done] model -> logs/xgboost_model.json")


if __name__ == "__main__":
    main()
