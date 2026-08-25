import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ETH Balina XGBoost Dashboard", layout="wide")

# =============================================================================
# ORTAK FONKSİYONLAR
# =============================================================================

def load_and_prepare_eth(file):
    df = pd.read_csv(file)
    drop_cols = [c for c in ["close_time", "quote_asset_volume", "number_of_trades",
                              "taker_buy_base_volume", "taker_buy_quote_volume", "source"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.set_index('open_time')

    from ta.trend import EMAIndicator, MACD, CCIIndicator
    from ta.momentum import RSIIndicator
    from ta.volatility import BollingerBands
    from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator

    df['ema_21'] = EMAIndicator(df['close'], window=21).ema_indicator()
    df['rsi_14'] = RSIIndicator(df['close'], window=14).rsi()
    macd_ind = MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd_ind.macd()
    df['macd_signal'] = macd_ind.macd_signal()
    df['macd_hist'] = macd_ind.macd_diff()
    bbands = BollingerBands(df['close'], window=20, window_dev=2)
    df['bb_width'] = bbands.bollinger_wband()
    df['bb_percent'] = bbands.bollinger_pband()
    df['obv'] = OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df['cci_20'] = CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
    df['cmf_20'] = ChaikinMoneyFlowIndicator(df['high'], df['low'], df['close'], df['volume'], window=20).chaikin_money_flow()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    return df


def create_features(df):
    df['ema_dist'] = (df['close'] - df['ema_21']) / df['close']
    df['macd_norm'] = df['macd'] / df['close']
    df['macd_signal_norm'] = df['macd_signal'] / df['close']
    df['macd_hist_norm'] = df['macd_hist'] / df['close']
    df['obv_pct_change_1'] = df['obv'].pct_change(1)
    df['obv_pct_change_24'] = df['obv'].pct_change(24)
    df.drop(columns=['ema_21', 'macd', 'macd_signal', 'macd_hist', 'obv'], inplace=True)

    for lag in [1, 2, 3, 8, 24]:
        df[f'log_return_lag_{lag}'] = df['log_return'].shift(lag)

    ret_shifted = df['log_return'].shift(1)
    df['rolling_sum_6'] = ret_shifted.rolling(window=6).sum()
    df['rolling_sum_24'] = ret_shifted.rolling(window=24).sum()
    df['rolling_std_6'] = ret_shifted.rolling(window=6).std()

    lag_delta_cols = ['rsi_14', 'cci_20', 'macd_hist_norm', 'bb_percent']
    for col in lag_delta_cols:
        for lag in [1, 2, 3]:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        df[f'{col}_delta'] = df[f'{col}_lag_1'] - df[f'{col}_lag_2']

    for lag in [1, 6, 24]:
        df[f'bb_width_lag_{lag}'] = df['bb_width'].shift(lag)
    for lag in [1, 4, 8]:
        df[f'cmf_20_lag_{lag}'] = df['cmf_20'].shift(lag)

    feature_cols = (
        [f'log_return_lag_{lag}' for lag in [1, 2, 3, 8, 24]]
        + ['rolling_sum_6', 'rolling_sum_24', 'rolling_std_6']
        + [f'{col}_lag_{lag}' for col in lag_delta_cols for lag in [1, 2, 3]]
        + [f'{col}_delta' for col in lag_delta_cols]
        + [f'bb_width_lag_{lag}' for lag in [1, 6, 24]]
        + [f'cmf_20_lag_{lag}' for lag in [1, 4, 8]]
    )

    model_df = df[feature_cols + ['log_return', 'close']].copy()
    model_df['close_prev'] = df['close'].shift(1).loc[model_df.index]
    model_df.dropna(inplace=True)
    return model_df, feature_cols


def load_whale_features(file):
    whale_emb = pd.read_csv(file)
    whale_emb['hour'] = pd.to_datetime(whale_emb['hour'])
    whale_emb = whale_emb.set_index('hour').sort_index()
    return whale_emb


def create_whale_features(whale_emb, df_index):
    level_cols = ['pca_1', 'pca_2', 'pca_3', 'emb_l2_norm']
    diff_cols = ['cosine_dist_1h', 'l2_norm_delta_1h', 'active_delta']
    zscore_col = ['active_zscore_24h']
    count_col = ['wallet_active_count']

    whale_full = pd.DataFrame(index=whale_emb.index)
    for col in level_cols:
        for lag in [1, 6, 24]:
            whale_full[f'{col}_lag_{lag}'] = whale_emb[col].shift(lag)
    for col in diff_cols:
        for lag in [1, 3]:
            whale_full[f'{col}_lag_{lag}'] = whale_emb[col].shift(lag)
    for col in zscore_col:
        whale_full[f'{col}_lag_1'] = whale_emb[col].shift(1)
    for col in count_col:
        for lag in [1, 6]:
            whale_full[f'{col}_lag_{lag}'] = whale_emb[col].shift(lag)

    pca_1_shifted = whale_emb['pca_1'].shift(1)
    emb_norm_shifted = whale_emb['emb_l2_norm'].shift(1)
    active_delta_shifted = whale_emb['active_delta'].shift(1)
    cosine_shifted = whale_emb['cosine_dist_1h'].shift(1)
    active_count_shifted = whale_emb['wallet_active_count'].shift(1)

    whale_full['pca_1_momentum_4h'] = pca_1_shifted.diff(4)
    whale_full['emb_norm_momentum_6h'] = emb_norm_shifted.diff(6)
    whale_full['active_count_acceleration'] = active_delta_shifted.diff(1)
    whale_full['cosine_dist_rolling_std_6h'] = cosine_shifted.rolling(6).std()
    whale_full['pca_1_rolling_std_12h'] = pca_1_shifted.rolling(12).std()
    whale_full['active_count_rolling_mean_12h'] = active_count_shifted.rolling(12).mean()

    whale_lagged = whale_full.reindex(df_index)
    return whale_lagged


def directional_accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return (np.sign(y_pred) == np.sign(y_true)).mean()


def run_nested_cv(model_df, feature_cols, n_splits_outer=15, purge_gap=24,
                  max_train_size=3000, n_splits_inner=5, n_iter_inner=40):
    X = model_df[feature_cols]
    y = model_df['log_return']

    if max_train_size and max_train_size > 0:
        outer_cv = TimeSeriesSplit(n_splits=n_splits_outer, gap=purge_gap, max_train_size=max_train_size)
    else:
        outer_cv = TimeSeriesSplit(n_splits=n_splits_outer, gap=purge_gap)

    param_distributions = {
        "n_estimators": [300, 400, 500, 700, 900],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.05, 0.1, 0.3, 0.5],
        "reg_lambda": [0.3, 0.5, 0.8, 1.0, 1.5],
        "min_child_weight": [1, 2, 3, 5],
        "gamma": [0, 0.05, 0.1, 0.2],
    }

    directional_scorer = make_scorer(directional_accuracy_score, greater_is_better=True)

    rmse_list, mae_list, hit_ratio_list, price_rmse_list = [], [], [], []
    importances = np.zeros(len(feature_cols))
    fold_plot_data = []
    best_params_per_fold = []
    degenerate_fold_flags = []
    fold = 1

    progress_bar = st.progress(0)
    status_text = st.empty()

    for train_idx, test_idx in outer_cv.split(X):
        status_text.text(f"Fold {fold}/{n_splits_outer} egitiliyor...")
        progress_bar.progress(fold / n_splits_outer)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        inner_cv = TimeSeriesSplit(n_splits=n_splits_inner, gap=purge_gap)
        base_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)

        inner_search = RandomizedSearchCV(
            estimator=base_model, param_distributions=param_distributions,
            n_iter=n_iter_inner, scoring=directional_scorer, cv=inner_cv,
            n_jobs=-1, random_state=42, refit=True,
        )
        inner_search.fit(X_train, y_train)
        best_params_per_fold.append(inner_search.best_params_)

        final_model = inner_search.best_estimator_
        y_pred = final_model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        hit_ratio = (np.sign(y_pred) == np.sign(y_test)).mean() * 100

        close_prev_test = model_df['close_prev'].iloc[test_idx].values
        close_actual_test = model_df['close'].iloc[test_idx].values
        price_pred = close_prev_test * np.exp(y_pred)
        price_rmse = np.sqrt(mean_squared_error(close_actual_test, price_pred))

        unique_preds = len(np.unique(np.round(y_pred, 8)))
        pred_std = y_pred.std()
        is_degenerate = (unique_preds <= 2) or (pred_std < 1e-6)
        degenerate_fold_flags.append(is_degenerate)

        rmse_list.append(rmse); mae_list.append(mae)
        hit_ratio_list.append(hit_ratio); price_rmse_list.append(price_rmse)
        importances += final_model.feature_importances_

        fold_plot_data.append(dict(
            fold=fold, train_idx=train_idx, test_idx=test_idx,
            price_pred=price_pred, hit_ratio=hit_ratio, price_rmse=price_rmse,
        ))
        fold += 1

    progress_bar.progress(1.0)
    status_text.text("Egitim tamamlandi!")

    results = {
        'rmse_list': rmse_list, 'mae_list': mae_list,
        'hit_ratio_list': hit_ratio_list, 'price_rmse_list': price_rmse_list,
        'importances': importances, 'fold_plot_data': fold_plot_data,
        'best_params_per_fold': best_params_per_fold,
        'degenerate_fold_flags': degenerate_fold_flags,
        'feature_cols': feature_cols,
        'avg_rmse': np.mean(rmse_list), 'avg_mae': np.mean(mae_list),
        'avg_hit': np.mean(hit_ratio_list), 'avg_price_rmse': np.mean(price_rmse_list),
    }
    return results


# =============================================================================
# PLOT FONKSİYONLARI
# =============================================================================

def plot_fold_overview(results, model_df):
    fold_data = results['fold_plot_data']
    n_folds = len(fold_data)
    fig = make_subplots(rows=n_folds, cols=1, shared_xaxes=True, vertical_spacing=0.015,
                        subplot_titles=[f"Fold {d['fold']}: Hit={d['hit_ratio']:.1f}% | RMSE=${d['price_rmse']:.2f}" for d in fold_data])

    for i, data in enumerate(fold_data):
        train_idx, test_idx = data['train_idx'], data['test_idx']
        price_pred = data['price_pred']
        train_start = model_df.index[train_idx[0]]
        train_end = model_df.index[train_idx[-1]]
        full_prices = model_df['close'].loc[train_start:model_df.index[test_idx[-1]]]

        fig.add_trace(go.Scatter(x=full_prices.index, y=full_prices.values, mode='lines',
                                 name='Gercek', line=dict(color='black', width=0.8),
                                 showlegend=(i == 0)), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=model_df.index[test_idx], y=price_pred, mode='lines',
                                 name='Tahmin', line=dict(color='red', dash='dash', width=1),
                                 showlegend=(i == 0)), row=i + 1, col=1)

    fig.update_layout(height=280 * n_folds, title_text="Fold Bazli Fiyat Tahminleri",
                      template='plotly_white', showlegend=True)
    return fig


def plot_fold_detailed_grid(results, model_df, n_last_days=2):
    fold_data = results['fold_plot_data']
    n_bars = n_last_days * 24 + 1
    n_folds = len(fold_data)
    n_rows, n_cols = 5, 3
    fig = make_subplots(rows=n_rows * 2, cols=n_cols, shared_xaxes=True,
                        vertical_spacing=0.03, horizontal_spacing=0.05,
                        row_heights=[3, 1] * n_rows)

    for i, data in enumerate(fold_data):
        test_idx = data['test_idx']
        price_pred = data['price_pred']
        last_test_idx = test_idx[-n_bars:]
        last_price_pred = price_pred[-n_bars:]
        dates = model_df.index[last_test_idx]
        actual_price = model_df['close'].iloc[last_test_idx].values
        close_prev = model_df['close_prev'].iloc[last_test_idx].values

        row_group = i // n_cols
        col = i % n_cols + 1
        row_price = row_group * 2 + 1
        row_err = row_group * 2 + 2

        window_rmse = np.sqrt(mean_squared_error(actual_price, last_price_pred))
        pred_dir = np.sign(last_price_pred - close_prev)
        actual_dir = np.sign(actual_price - close_prev)
        window_hit = (pred_dir == actual_dir).mean() * 100

        fig.add_trace(go.Scatter(x=dates, y=actual_price, mode='lines+markers',
                                 name='Gercek', line=dict(color='black', width=0.8),
                                 marker=dict(size=2), showlegend=(i == 0)),
                      row=row_price, col=col)
        fig.add_trace(go.Scatter(x=dates, y=last_price_pred, mode='lines+markers',
                                 name='Tahmin', line=dict(color='red', dash='dash', width=0.8),
                                 marker=dict(size=3, symbol='x'), showlegend=(i == 0)),
                      row=row_price, col=col)

        signed_error = actual_price - last_price_pred
        colors = ['#ff7f0e' if e >= 0 else '#1f77b4' for e in signed_error]
        fig.add_trace(go.Bar(x=dates, y=signed_error, name='Hata',
                             marker_color=colors, showlegend=(i == 0)),
                      row=row_err, col=col)

    fig.update_layout(height=380 * n_rows, title_text=f"Son {n_last_days} Gun Detaysi (Fiyat + Hata)",
                      template='plotly_white', showlegend=True)
    return fig


def plot_feature_importance(results, top_n=30):
    feat_cols = results['feature_cols']
    importances = results['importances'] / len(results['fold_plot_data'])
    indices = np.argsort(importances)[-top_n:]

    fig = go.Figure(go.Bar(
        x=importances[indices],
        y=[feat_cols[i] for i in indices],
        orientation='h',
        marker_color='steelblue'
    ))
    fig.update_layout(title=f"En Onemli {top_n} Feature",
                      xaxis_title="Importance", height=max(400, top_n * 22),
                      template='plotly_white', yaxis=dict(autorange="reversed"))
    return fig


def plot_fold_metrics(results):
    n = len(results['fold_plot_data'])
    folds = [f"Fold {i+1}" for i in range(n)]

    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Hit Ratio (%)", "Fiyat RMSE ($)", "Log Return RMSE"])
    fig.add_trace(go.Bar(x=folds, y=results['hit_ratio_list'], name='Hit Ratio',
                         marker_color=['green' if h >= 50 else 'red' for h in results['hit_ratio_list']]),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=folds, y=results['price_rmse_list'], name='Fiyat RMSE',
                         marker_color='coral'), row=1, col=2)
    fig.add_trace(go.Bar(x=folds, y=results['rmse_list'], name='Log RMSE',
                         marker_color='mediumpurple'), row=1, col=3)
    fig.update_layout(height=350, title_text="Fold Bazli Metrikler", template='plotly_white', showlegend=False)
    return fig


# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.title("ETH Balina XGBoost Fiyat Tahmin Dashboard")
    st.markdown("XGBoost ile ETH/USDT saatlik fiyat tahmin modeli. Whale embedding ozellikleri dahil edilebilir.")

    # Sidebar
    st.sidebar.title("Ayarlar")
    eth_file = st.sidebar.file_uploader("ETH Veri CSV Yukle", type=["csv"])
    whale_file = st.sidebar.file_uploader("Whale Embedding CSV (Opsiyonel)", type=["csv"])

    st.sidebar.markdown("---")
    n_splits = st.sidebar.slider("Outer CV Fold Sayisi", 3, 30, 15)
    max_train_size_val = st.sidebar.slider("Max Train Size (0=expanding)", 0, 12000, 3000, step=100)
    max_train_size = max_train_size_val if max_train_size_val > 0 else None
    n_last_days = st.sidebar.slider("Son N Gun Detaysi", 1, 7, 2)

    if eth_file is None:
        st.info("Sidebar'dan ETH veri CSV dosyasini yukleyin.")
        st.stop()

    # Data processing
    with st.spinner("ETH verisi isleniyor..."):
        df = load_and_prepare_eth(eth_file)
    st.success(f"ETH verisi yuklendi: {len(df)} satir, {df.index.min()} -> {df.index.max()}")

    with st.spinner("Feature'lar olusturuluyor..."):
        model_df, feature_cols = create_features(df.copy())

    all_features = list(feature_cols)
    if whale_file is not None:
        try:
            with st.spinner("Whale embedding verisi isleniyor..."):
                whale_emb = load_whale_features(whale_file)
                whale_lagged = create_whale_features(whale_emb, df.index)
                whale_feature_cols = whale_lagged.columns.tolist()
                whale_in_model = whale_lagged.loc[model_df.index]
                for c in whale_feature_cols:
                    model_df[c] = whale_in_model[c]
                model_df.dropna(inplace=True)
                all_features = list(feature_cols) + whale_feature_cols
            st.success(f"Whale feature eklendi: +{len(whale_feature_cols)} ozellik (Toplam: {len(all_features)})")
        except Exception as e:
            st.error(f"Whale verisi yuklenemedi: {e}")

    if 'close_prev' not in model_df.columns:
        model_df['close_prev'] = df['close'].shift(1).loc[model_df.index]
    model_df = model_df.dropna(subset=['close_prev'])

    # Run model
    if st.sidebar.button("Modeli Egit", type="primary"):
        with st.spinner("Model egitimi basliyor... (Bu bir sure alabilir)"):
            results = run_nested_cv(model_df, all_features,
                                    n_splits_outer=n_splits, max_train_size=max_train_size)
        st.session_state['results'] = results
        st.session_state['model_df'] = model_df
        st.session_state['all_features'] = all_features
        st.success("Egitim tamamlandi!")

    if 'results' in st.session_state:
        results = st.session_state['results']
        model_df = st.session_state['model_df']

        # Summary metrics
        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ort. Hit Ratio", f"{results['avg_hit']:.2f}%")
        c2.metric("Ort. Fiyat RMSE", f"${results['avg_price_rmse']:.2f}")
        c3.metric("Ort. Log RMSE", f"{results['avg_rmse']:.6f}")
        c4.metric("Degenerate Fold", f"{sum(results['degenerate_fold_flags'])}/{len(results['fold_plot_data'])}")

        # Fold detail table
        st.markdown("### Fold Detaylari")
        fold_df = pd.DataFrame({
            'Fold': [d['fold'] for d in results['fold_plot_data']],
            'Train': [len(d['train_idx']) for d in results['fold_plot_data']],
            'Test': [len(d['test_idx']) for d in results['fold_plot_data']],
            'Hit Ratio (%)': [round(d['hit_ratio'], 2) for d in results['fold_plot_data']],
            'Fiyat RMSE ($)': [round(d['price_rmse'], 2) for d in results['fold_plot_data']],
            'Degenerate': results['degenerate_fold_flags'],
        })
        st.dataframe(fold_df, use_container_width=True)

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Fold Genel Bakis", "Son N Gun Detaysi", "Feature Onem Sirasi", "Fold Metrikleri", "Hyperparametreler"
        ])

        with tab1:
            fig = plot_fold_overview(results, model_df)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            fig = plot_fold_detailed_grid(results, model_df, n_last_days=n_last_days)
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            top_n = st.slider("Kac feature gosterilsin?", 10, 50, 30, key="feat_n")
            fig = plot_feature_importance(results, top_n=top_n)
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            fig = plot_fold_metrics(results)
            st.plotly_chart(fig, use_container_width=True)

        with tab5:
            params_df = pd.DataFrame(results['best_params_per_fold'])
            params_df.index = [f"Fold {i+1}" for i in range(len(params_df))]
            st.dataframe(params_df, use_container_width=True)


if __name__ == "__main__":
    main()
