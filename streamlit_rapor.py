import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="BGL Log Arıza Tespiti — Rapor Panosu",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RENK_NAVY = "#0F2B46"
RENK_TEAL = "#0FB5A6"
RENK_ORANGE = "#F59E0B"
RENK_RED = "#DD5454"

VERI_YOLU = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "parsed.csv")

DS = {
    "dosya_mb": 743,
    "satir": 3_017_161,
    "olay_tipi": 5_000,
    "alert_orani": 8.4,
    "pencere": 5_957,
    "pozitif": 267,
    "egitim": 4_765,
    "test": 1_192,
    "test_pozitif": 22,
}

IF_SONUC = {
    "Accuracy": 0.9706,
    "Precision": 0.24,
    "Recall": 0.2727,
    "F1": 0.2553,
    "ROC-AUC": 0.8133,
}

SINIF_RAPOR = pd.DataFrame(
    {
        "Sınıf": ["Normal", "Anomali"],
        "Precision": [0.99, 0.24],
        "Recall": [0.98, 0.27],
        "F1": [0.99, 0.26],
        "Destek": [1170, 22],
    }
)

ESIK_NOKTALARI = pd.DataFrame(
    {
        "Senaryo": [
            "Eşik 0,2938",
            "Eşik 0,2940",
            "Varsayılan (contamination = 0,06)",
        ],
        "Precision": [0.0185, 0.0318, 0.24],
        "Recall": [1.0000, 0.9091, 0.2727],
        "F1": [0.0362, 0.0615, 0.2553],
        "Alarm Oranı (%)": [100.0, 52.7, 25 / DS["test"] * 100],
        "Alarm Sayısı": [1192, 662, 25],
    }
)

MODELLER = pd.DataFrame(
    {
        "Model": [
            "BiLSTM (CrossEntropy)",
            "BiLSTM (BCEWithLogitsLoss)",
            "BiLSTM (Focal Loss)",
            "Isolation Forest (gözetimsiz)",
        ],
        "ROC-AUC": [0.44, 0.71, 0.37, 0.81],
        "Precision": [0.00, 0.00, 0.00, 0.24],
        "F1": [0.00, 0.00, 0.00, 0.26],
        "Tür": ["Denetimli", "Denetimli", "Denetimli", "Gözetimsiz"],
    }
)

PENCERE_KARSILASTIRMA = pd.DataFrame(
    {
        "Pencere Genişliği": ["30 dakika", "15 dakika"],
        "ROC-AUC": [0.80, 0.81],
        "Recall (%)": [57.7, 27.3],
    }
)


@st.cache_data(show_spinner="parsed.csv yükleniyor...")
def veri_onizleme(n_satir: int) -> pd.DataFrame:
    return pd.read_csv(VERI_YOLU, nrows=n_satir)


BOYUTLAR = [300, 450, 600, 750, 900, 1050, 1200, 1350, 1500, 1650, 1800]

CAPAS = {
    900: {"auc": 0.8133, "recall": 0.2727},
    1800: {"auc": 0.800, "recall": 0.577},
}


def sayi(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def tahmini_performans(window_s: int) -> dict:
    if window_s in CAPAS:
        return {**CAPAS[window_s], "olculen": True}
    xs = sorted(CAPAS)
    return {
        "auc": float(np.interp(window_s, xs, [CAPAS[x]["auc"] for x in xs])),
        "recall": float(np.interp(window_s, xs, [CAPAS[x]["recall"] for x in xs])),
        "olculen": False,
    }


@st.cache_data(show_spinner="parsed.csv okunuyor...")
def _ham_zaman():
    df = pd.read_csv(VERI_YOLU, usecols=["timestamp", "is_alert"])
    return (
        df["timestamp"].to_numpy(dtype=np.int64),
        df["is_alert"].to_numpy(dtype=np.int64),
    )


@st.cache_data(show_spinner=f"Pencere istatistikleri hesaplanıyor...")
def pencere_istatistikleri(window_s: int, min_events: int = 3):
    """isolation_model.py'deki build_features döngüsünün birebir kopyası (yalnızca sayım)."""
    try:
        ts, alerts = _ham_zaman()
    except Exception:
        return None
    ms = int(window_s) * 1000
    start, end = int(ts[0]), int(ts[-1])
    prev = None
    labels = []
    for w_start in range(start, end + 1, ms):
        left = int(np.searchsorted(ts, w_start, side="left"))
        if prev is not None and left > prev:
            prev = left
        right = int(np.searchsorted(ts, w_start + ms, side="left"))
        if prev is None:
            prev = left
        if right <= prev:
            continue
        if right - prev < min_events:
            prev = right
            continue
        h_left = right
        h_right = int(np.searchsorted(ts, w_start + 2 * ms, side="right"))
        labels.append(int(alerts[h_left:h_right].sum() > 0) if h_left < h_right else 0)
        prev = right
    y = np.array(labels, dtype=np.int64)
    n_egitim = int(len(y) * 0.8)
    return {
        "toplam": len(y),
        "pozitif": int(y.sum()),
        "egitim": n_egitim,
        "test": len(y) - n_egitim,
        "test_pozitif": int(y[n_egitim:].sum()),
    }


@st.cache_data(show_spinner="Farklı pencere boyutları hesaplanıyor...")
def pencere_tablosu() -> pd.DataFrame:
    kayit = []
    for s in BOYUTLAR:
        i = pencere_istatistikleri(s)
        if i is None:
            continue
        p = tahmini_performans(s)
        kayit.append(
            {
                "Pencere (dk)": s // 60,
                "Toplam Pencere": i["toplam"],
                "Pozitif": i["pozitif"],
                "Pozitif Oran (%)": round(i["pozitif"] / i["toplam"] * 100, 2),
                "Test Pozitifi": i["test_pozitif"],
                "ROC-AUC ≈": round(p["auc"], 3),
                "Recall % ≈": round(p["recall"] * 100, 1),
            }
        )
    return pd.DataFrame(kayit)


def metrik_kart(col, baslik, deger, alt_metin="", renk=RENK_TEAL):
    with col:
        st.markdown(
            f"""
            <div style="
                background:#F3F6FA; border-left:6px solid {renk};
                border-radius:10px; padding:14px 18px;">
                <div style="font-size:13px;color:#4E5A68;">{baslik}</div>
                <div style="font-size:30px;font-weight:700;color:{RENK_NAVY};">{deger}</div>
                <div style="font-size:11px;color:#7A8694;">{alt_metin}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


st.sidebar.markdown("## 🖥️ BGL Log Arıza Tespiti")
st.sidebar.caption("Isolation Forest ile Gözetimsiz Anomali Tespiti\n\nSelçuk Sezer · softITo · Ağustos 2026")
sayfa = st.sidebar.radio(
    "Bölümler",
    [
        "🏠 Genel Bakış",
        "📊 Veri Kümesi",
        "🤖 Isolation Forest Sonuçları",
        "⚖️ Model Karşılaştırma",
        "🎚️ Eşik ve Pencere Analizi",
        "📝 Sonuç ve Gelecek Çalışmalar",
    ],
)
st.sidebar.divider()
with st.sidebar.expander("⚙️ Yapılandırma (config.yaml)"):
    st.markdown(
        """
| Parametre | Değer |
|---|---|
| `similarity_threshold` | 0.5 |
| `depth` | 4 |
| `window_seconds` | 900 (15 dk) |
| `horizon_seconds` | 900 (15 dk) |
| `n_estimators` | 200 |
| `contamination` | 0.06 |
| `random_state` | 42 |
"""
    )

st.sidebar.divider()
PENCERE_SN = st.sidebar.select_slider(
    "⏱️ Pencere Genişliği (window_seconds)",
    options=BOYUTLAR,
    value=900,
    format_func=lambda s: f"{s // 60} dk ({s} sn)",
)
st.sidebar.caption(
    "🟢 **15 dk & 30 dk:** rapordaki ölçülmüş performans\n\n"
    "🔵 **diğerleri:** pencere sayıları `parsed.csv`'den birebir hesaplanır; "
    "ROC-AUC / Recall değerleri ölçümler arası doğrusal tahmindir"
)

if sayfa == "🏠 Genel Bakış":
    st.title("Sunucu Log Kayıtlarında Sistem Arızası Tespiti")
    st.markdown(
        "**Amaç:** Blue Gene/L (BGL) süper bilgisayarının ham loglarından hareketle, "
        "gözetimsiz öğrenme (Isolation Forest) ile anormal davranışları tespit etmek "
        "ve arızaları ileriye dönük öngörmek."
    )
    st.info(
        "💡 **Temel fark:** Isolation Forest hiçbir arıza etiketi görmeden yalnızca "
        "\u201cnormal davranışı\u201d öğrenir. Alert etiketleri yalnızca sonucun doğrulanmasında kullanılır.",
        icon="✅",
    )

    st.subheader("Pipeline")
    adimlar = [
        ("HAM LOGLAR", "BGL.log\n743 MB"),
        ("DRAIN PARSE", "parsed.csv\n5.000 olay tipi"),
        ("KAYAN PENCERE", f"{PENCERE_SN // 60} dk\npencereler"),
        ("ÖZNİTELİKLER", "5.008 boyut\n+ StandardScaler"),
        ("ISOLATION FOREST", "200 ağaç\ncont. 0,06"),
        ("DEĞERLENDİRME", "alert etiketiyle\nsanity check"),
    ]
    cols = st.columns(len(adimlar), gap="small")
    for i, (c, (baslik, alt)) in enumerate(zip(cols, adimlar)):
        renk = RENK_NAVY if i in (0, 4) else RENK_TEAL
        c.markdown(
            f"""
            <div style="background:{renk};color:white;border-radius:10px;
                 padding:12px 8px;text-align:center;margin-bottom:4px;">
                <div style="font-weight:700;font-size:12px;">{baslik}</div>
            </div>
            <div style="font-size:11px;color:#4E5A68;text-align:center;">{alt.replace(chr(10), '<br>')}</div>
            """,
            unsafe_allow_html=True,
        )

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Öne Çıkan Bulgular")
        st.markdown(
            """
- Etiket görmeden **ROC-AUC 0,81** — gerçek alert pencerelerinin ayrıştırılmasında güçlü sinyal
- Aynı veride denetimli BiLSTM modelleri **P = F1 = 0,00** ile başarısız oldu
- Eşik ayarıyla recall **%91–100** bandına çıkarılabiliyor (operasyonel tercih)
- Kısa pencere ayırt ediciliği artırıyor, duyarlılığı azaltıyor (**trade-off**)
"""
        )
    with c2:
        st.subheader("Tahmin Görevi")
        st.success(f"**\u201cÖnümüzdeki {PENCERE_SN // 60} dakika içinde bir arıza meydana gelecek mi?\u201d**", icon="🎯")
        st.caption(f"Pencere kapanışını izleyen horizon_seconds ({PENCERE_SN} sn) içinde ≥ 1 alert → etiket 1.")

elif sayfa == "📊 Veri Kümesi":
    st.title("📊 BGL Veri Kümesi")
    st.markdown(
        "Los Alamos National Laboratory'nin **131.072 işlemcili** Blue Gene/L süper bilgisayarının "
        "konsol logları — log tabanlı arıza tahmini araştırmalarında standart kıyaslama kümesi."
    )
    c1, c2, c3 = st.columns(3)
    metrik_kart(c1, "Ham Dosya", f"{DS['dosya_mb']} MB", "data/BGL.log", RENK_ORANGE)
    metrik_kart(c2, "Log Satırı", f"{DS['satir']:,}".replace(",", "."), "ayrıştırma sonrası")
    metrik_kart(c3, "Olay Tipi", f"{DS['olay_tipi']:,}".replace(",", "."), "benzersiz event_id", RENK_ORANGE)

    st.divider()
    st.subheader("Sınıf Dengesizliği")
    col_chart, col_stats = st.columns([3, 2])
    with col_chart:
        fig = go.Figure(
            go.Pie(
                labels=["Normal", "Anomali (alert)"],
                values=[100 - DS["alert_orani"], DS["alert_orani"]],
                hole=0.55,
                marker=dict(colors=[RENK_NAVY, RENK_RED]),
            )
        )
        fig.update_traces(textinfo="percent", textfont_size=16)
        fig.update_layout(title="Satır bazlı alert oranı (~%8,4)", height=360, margin=dict(t=60, b=20))
        st.plotly_chart(fig, width="stretch")
    with col_stats:
        ist = pencere_istatistikleri(PENCERE_SN)
        dk = PENCERE_SN // 60
        if ist is None:
            st.warning("`data/parsed.csv` okunamadı — kartlar rapordaki 15 dk değerlerini gösteriyor.", icon="⚠️")
            ist = {"toplam": DS["pencere"], "pozitif": DS["pozitif"], "egitim": DS["egitim"],
                   "test": DS["test"], "test_pozitif": DS["test_pozitif"]}
        st.markdown(f"###### Pencere düzeyinde ({dk} dk)")
        m1, m2 = st.columns(2)
        metrik_kart(m1, "Toplam Pencere", sayi(ist["toplam"]), f"{dk} dk, örtüşmesiz")
        metrik_kart(m2, "Pozitif", sayi(ist["pozitif"]), f"%{ist['pozitif'] / ist['toplam'] * 100:.2f}", RENK_RED)
        m3, m4 = st.columns(2)
        metrik_kart(m3, "Test Pozitifi", f"{ist['test_pozitif']} / {sayi(ist['test'])}", f"%{ist['test_pozitif'] / ist['test'] * 100:.2f}", RENK_RED)
        metrik_kart(m4, "Eğitim / Test", f"{sayi(ist['egitim'])} / {sayi(ist['test'])}", "%80 / %20 kronolojik")

    st.warning(
        "**Etiket tanımı:** `KERN`, `APP`, `RAS` önekli satırlar \u201calert\u201d (gerçek arıza/uyarı) olarak işaretlenir. "
        "Pozitiflerin nadirliği hem gözetimsiz skorların hem de denetimli modellerin davranışını belirler.",
        icon="⚠️",
    )

    if os.path.exists(VERI_YOLU):
        st.subheader("Pencere Genişliğine Göre Dengesizlik")
        tablo = pencere_tablosu()
        c_line, c_tab = st.columns([3, 2])
        with c_line:
            fig = px.line(tablo, x="Pencere (dk)", y="Pozitif Oran (%)", markers=True)
            fig.update_traces(line_color=RENK_NAVY, marker_color=RENK_TEAL)
            fig.add_vline(x=PENCERE_SN // 60, line_dash="dash", line_color=RENK_ORANGE,
                          annotation_text="seçili", annotation_position="top left")
            fig.update_layout(height=320, margin=dict(t=40),
                              xaxis_title="Pencere Genişliği (dk)",
                              yaxis_title="Pozitif Pencere Oranı (%)")
            st.plotly_chart(fig, width="stretch")
        with c_tab:
            st.dataframe(tablo, hide_index=True, width="stretch", height=320)
            st.caption("Pencere sayıları `parsed.csv`'den birebir hesaplanır; ROC-AUC / Recall sütunları "
                       "15 dk ↔ 30 dk ölçümleri arasındaki doğrusal tahmindir.")

    st.divider()
    if os.path.exists(VERI_YOLU):
        st.subheader("Ham Veri Önizleme — data/parsed.csv")
        n = st.slider("Yüklenecek satır sayısı", 1_000, 100_000, 10_000, step=1_000)
        try:
            df_onizleme = veri_onizleme(n)
            st.dataframe(df_onizleme, width="stretch")
            st.caption(f"Gösterilen: ilk {len(df_onizleme):,} satır (dosya toplamda ~3 M satır içerir)".replace(",", "."))
        except Exception as e:
            st.error(f"Dosya okunamadı: {e}")
    else:
        st.info("`data/parsed.csv` bulunamadı — önizleme için önce parse işlemini çalıştırın (`python isolation_model.py`).")

elif sayfa == "🤖 Isolation Forest Sonuçları":
    st.title("🤖 Isolation Forest — Deneysel Sonuçlar")
    st.caption("Test bölmesi: 1.192 pencere · 22 pozitif · model hiçbir etiket görmeden eğitildi")

    c1, c2, c3, c4, c5 = st.columns(5)
    perf = tahmini_performans(PENCERE_SN)
    ist = pencere_istatistikleri(PENCERE_SN) or {"test": DS["test"], "test_pozitif": DS["test_pozitif"]}
    tr_fmt = lambda v: f"{v:.2f}".replace(".", ",")
    if perf["olculen"]:
        acc_s, pre_s, rec_s, f1_s = "%97,06", "0,24", tr_fmt(perf["recall"]), "0,26"
        auc_s, auc_alt = tr_fmt(perf["auc"]), "ana başarı göstergesi"
    else:
        acc_s = pre_s = f1_s = "—"
        rec_s, auc_s = "≈ " + tr_fmt(perf["recall"]), "≈ " + tr_fmt(perf["auc"])
        auc_alt = f"{PENCERE_SN // 60} dk — doğrusal tahmin"
    metrik_kart(c1, "Accuracy", acc_s, "" if perf["olculen"] else "yalnızca 15 dk ölçümü")
    metrik_kart(c2, "Precision", pre_s, "anomali sınıfı", RENK_ORANGE)
    metrik_kart(c3, "Recall", rec_s,
                "anomali sınıfı" if perf["olculen"] else "15 ↔ 30 dk interpolasyonu", RENK_ORANGE)
    metrik_kart(c4, "F1", f1_s, "anomali sınıfı", RENK_ORANGE)
    metrik_kart(c5, "ROC-AUC", auc_s, auc_alt, RENK_TEAL)

    if not perf["olculen"]:
        st.info(
            f"ℹ️ **{PENCERE_SN // 60} dk** için eğitilmiş model yok. Accuracy / Precision / F1 yalnızca "
            "rapordaki 15 dk ölçümündedir; ROC-AUC ve Recall, 15 dk ↔ 30 dk ölçümleri arasında "
            "doğrusal tahminle verilmiştir (≈). Pencere sayıları ise `parsed.csv`'den gerçekten hesaplanmıştır.",
            icon="🧮",
        )

    col_gauge, col_class, col_yorum = st.columns([1.2, 1.4, 1.6])
    with col_gauge:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=perf["auc"],
            number={"valueformat": ".2f"},
            title={"text": "ROC-AUC" + ("" if perf["olculen"] else " (≈ tahmin)")},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": RENK_TEAL},
                "steps": [
                    {"range": [0, 0.5], "color": "#FBEAEA"},
                    {"range": [0.5, 0.75], "color": "#FFF4DC"},
                    {"range": [0.75, 1], "color": "#E0F5F2"},
                ],
                "threshold": {"line": {"color": RENK_NAVY, "width": 3}, "value": perf["auc"]},
            },
        ))
        fig.update_layout(height=280, margin=dict(t=40, b=20))
        st.plotly_chart(fig, width="stretch")
    with col_class:
        if perf["olculen"]:
            st.markdown("##### Sınıf Bazlı Performans")
            st.dataframe(SINIF_RAPOR, hide_index=True, width="stretch")
            st.caption("Tahmin edilen anomali: 25 / 1.192 — gerçek: 22 / 1.192")
        else:
            st.markdown("##### Sınıf Bazlı Performans")
            st.info("Sınıf bazlı rapor yalnızca ölçülmüş konfigürasyonda (15 dk) görüntülenir.", icon="📊")
    with col_yorum:
        test_oran = ist["test_pozitif"] / ist["test"] * 100
        st.markdown("##### Yorumlama")
        st.markdown(
            f"""
- **AUC {auc_s}** → anomali skorları pozitif pencereleri normalden ayırt ediyor
  ({'ölçülmüş' if perf['olculen'] else 'tahminî'} değer).
- Düşük precision (%{test_oran:.2f}'lik test pozitif oranı + gözetimsiz paradigmanın doğal sonucu): model
  \u201carıza gerçekleşecek pencere\u201dyi değil, \u201cistatistiksel olarak sıra dışı pencere\u201dyi işaretler.
- Erken uyarı senaryosunda kritik metrik **recall** → bkz. *Eşik ve Pencere Analizi* sekmesi.
"""
        )

elif sayfa == "⚖️ Model Karşılaştırma":
    st.title("⚖️ Denetimli vs Gözetimsiz")
    st.caption("LogRobust mimarisi (BiLSTM + Multi-Head Attention) farklı kayıp fonksiyonlarıyla, aynı veri üzerinde")
    secilen = st.radio("Metrik seçin", ["ROC-AUC", "Precision", "F1"], horizontal=True)
    fig = px.bar(
        MODELLER,
        x="Model",
        y=secilen,
        color="Tür",
        color_discrete_map={"Denetimli": RENK_RED, "Gözetimsiz": RENK_TEAL},
        text_auto=".2f",
        range_y=[0, 1] if secilen == "ROC-AUC" else None,
    )
    fig.update_layout(height=430, showlegend=False, margin=dict(t=30))
    fig.update_traces(textposition="outside", textfont_size=14)
    st.plotly_chart(fig, width="stretch")

    st.dataframe(MODELLER, hide_index=True, width="stretch")
    st.error(
        "**Sonuç:** Hiçbir BiLSTM konfigürasyonu karar eşiğinde tek bir pozitif örneği bile isabetle "
        "sınıflandıramadı (P = F1 = 0,00). En iyi sıralama bile (BCE, 0,71) gözetimsiz IF'in gerisinde kaldı.",
        icon="🔻",
    )
    with st.expander("Başarısızlığın olası nedenleri"):
        st.markdown(
            """
1. **Aşırı sınıf dengesizliği** — test bölmesinde pozitif oran %1,85; derin mimariler çoğunluk sınıfını tahmin etmeyi öğreniyor.
2. **Parametre hacmi >> veri hacmi** — attention'lı BiLSTM'in parametreleri 4.765 pencerelik eğitim kümesini aşıyor → overfitting/underfitting riski.
3. **Kayıp fonksiyonu kalibrasyonu** — BCE'de gradyan çoğunluk sınıfınca baskılanıyor; Focal Loss'un gamma/alpha parametreleri kalibre edilemedi (AUC 0,37 < rastgele 0,50).
"""
        )
        st.success(
            "Nüans: Sorun denetimli yöntemin doğasında değil; veri hacmi, dengesizlik ve kalibrasyon koşullarında. "
            "SMOTE / ağırlıklı örnekleme + eşik kalibrasyonuyla başarım artırılabilir.",
            icon="💡",
        )

elif sayfa == "🎚️ Eşik ve Pencere Analizi":
    st.title("🎚️ Operasyonel Ayarlar")
    perf = tahmini_performans(PENCERE_SN)
    perf_a, perf_r = perf["auc"], perf["recall"] * 100

    st.subheader("Karar Eşiği Süpürme")
    if PENCERE_SN != 900:
        st.caption("ⓘ Aşağıdaki eşik değerleri rapordaki **15 dk ölçülmüş** konfigürasyona aittir.")
    senaryo = st.selectbox("Senaryo seçin", ESIK_NOKTALARI["Senaryo"])
    satir = ESIK_NOKTALARI[ESIK_NOKTALARI["Senaryo"] == senaryo].iloc[0]
    k1, k2, k3, k4 = st.columns(4)
    metrik_kart(k1, "Precision", f"{satir['Precision']:.4f}", RENK_ORANGE)
    metrik_kart(k2, "Recall", f"%{satir['Recall']*100:.1f}", RENK_TEAL)
    metrik_kart(k3, "F1", f"{satir['F1']:.4f}", RENK_ORANGE)
    metrik_kart(k4, "Üretilen Alarm", f"{int(satir['Alarm Sayısı'])} / {DS['test']}", f"%{satir['Alarm Oranı (%)']:.1f} oran", RENK_RED)

    col_pr, col_bar = st.columns(2)
    with col_pr:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ESIK_NOKTALARI["Recall"], y=ESIK_NOKTALARI["Precision"],
            mode="markers+text",
            text=ESIK_NOKTALARI["Senaryo"],
            textposition="top center",
            marker=dict(size=14, color=[RENK_RED, "#FF9800", RENK_TEAL]),
        ))
        fig.update_layout(
            title="Çalışma Noktaları: Precision–Recall Ödünleşimi",
            xaxis_title="Recall", yaxis_title="Precision",
            height=380, margin=dict(t=50),
        )
        st.plotly_chart(fig, width="stretch")
    with col_bar:
        fig = px.bar(
            ESIK_NOKTALARI, x="Senaryo", y="Alarm Sayısı",
            color="Senaryo",
            color_discrete_sequence=[RENK_RED, "#FF9800", RENK_TEAL],
            text_auto=True,
        )
        fig.update_layout(
            title=f"Üretilen Alarm Sayısı (test: {DS['test']} pencere)",
            height=380, showlegend=False, margin=dict(t=50),
        )
        st.plotly_chart(fig, width="stretch")
    st.info(
        "Eşiğin hafif yükseltilmesi alarm sayısını **yarıya indirirken** recall'u %91'te tutuyor. "
        "Kaçırma maliyeti > yanlış alarm maliyeti olan erken uyarı sistemlerinde yüksek recall tercih edilir.",
        icon="🎛️",
    )

    st.divider()
    st.subheader("Pencere Genişliği Ödünleşimi")
    if os.path.exists(VERI_YOLU):
        tablo = pencere_tablosu()
    else:
        st.info("Dinamik eğri için `data/parsed.csv` gerekli; grafikte yalnızca rapordaki iki ölçülmüş nokta var.", icon="⚠️")
        tablo = pd.DataFrame({
            "Pencere (dk)": [15, 30],
            "ROC-AUC ≈": [0.813, 0.800],
            "Recall % ≈": [27.3, 57.7],
            "Toplam Pencere": [5957, None],
        })
    c_a, c_b = st.columns([3, 2])
    with c_a:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tablo["Pencere (dk)"], y=tablo["Recall % ≈"], name="Recall (%)",
            mode="lines+markers", line=dict(color=RENK_ORANGE, shape="spline"),
            hovertemplate="Pencere: %{x} dk<br>Recall ≈ %{y:.1f}%<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=tablo["Pencere (dk)"], y=tablo["ROC-AUC ≈"], name="ROC-AUC",
            mode="lines+markers", yaxis="y2",
            line=dict(color=RENK_TEAL, dash="dot", shape="spline"),
            hovertemplate="Pencere: %{x} dk<br>AUC ≈ %{y:.3f}<extra></extra>",
        ))
        olc = tablo[tablo["Pencere (dk)"].isin([15, 30])]
        fig.add_trace(go.Scatter(
            x=olc["Pencere (dk)"], y=olc["Recall % ≈"], name="ölçülen (rapor)",
            mode="markers", marker=dict(size=13, color=RENK_RED, symbol="diamond"),
            hovertemplate="Ölçülen nokta<extra></extra>",
        ))
        fig.add_vline(x=PENCERE_SN // 60, line_dash="dash", line_color=RENK_NAVY,
                      annotation_text=f"seçili {PENCERE_SN // 60} dk", annotation_position="top left")
        fig.update_layout(
            title=f"Seçim: {PENCERE_SN // 60} dk → Recall ≈ %{perf_r:.1f} · AUC ≈ {perf_a:.2f}",
            xaxis=dict(title="Pencere Genişliği (dk)", tickmode="array",
                       tickvals=sorted(tablo["Pencere (dk)"].unique())),
            yaxis=dict(title="Recall (%)"),
            yaxis2=dict(title="ROC-AUC", overlaying="y", side="right", range=[0.75, 0.85]),
            height=400, legend=dict(orientation="h", y=-0.25), margin=dict(t=60),
        )
        st.plotly_chart(fig, width="stretch")
    with c_b:
        ist = pencere_istatistikleri(PENCERE_SN)
        ekstra = ""
        if ist:
            ekstra = (f"- Seçili boyutta toplam pencere: **{sayi(ist['toplam'])}**, "
                      f"pozitif: **{sayi(ist['pozitif'])}** (%{ist['pozitif'] / ist['toplam'] * 100:.2f}).\n")
        st.markdown(
            f"""
**Seçili {PENCERE_SN // 60} dakika →** ROC-AUC ≈ {perf_a:.2f} · Recall ≈ %{perf_r:.1f}
<br>**Ölçümler:** 15 dk → AUC 0,81 · R %27,3 &nbsp;·&nbsp; 30 dk → AUC 0,80 · R %57,7

- Kısa pencere, anomaliyi dar bir zaman ufkuna sıkıştırarak **skor ayırt ediciliğini** artırır…
- …ancak pozitif örnek sayısındaki azalma nedeniyle **duyarlılık geriler**.
{ekstra}- Pencere genişliği, uygulama hedefine göre yapılması gereken **kritik tasarım kararıdır**.
""",
            unsafe_allow_html=True,
        )

else:
    st.title("📝 Sonuç ve Gelecek Çalışmalar")
    st.subheader("Ana Çıkarımlar")
    bulgular = [
        ("Uçtan uca gözetimsiz hat", "Drain → 15 dk kayan pencere → 5.008 boyutlu öznitelik → Isolation Forest; tekrarlanabilir (seed 42)."),
        ("Etiketsiz öğrenme çalışıyor", "Hiçbir arıza etiketi görmeden ROC-AUC 0,81 — gerçek alert pencerelerinin önemli bölümü yakalanıyor."),
        ("Bu koşullarda gözetimsiz > denetimli", "Aynı veride BiLSTM tabanlı modeller kullanılabilir başarım gösteremedi; IF daha güvenilir temel (baseline)."),
        ("Operasyonel esneklik", "Eşik ayarıyla recall–precision tercihi kalibre edilebilir; pencere genişliği kritik tasarım kararı."),
    ]
    for i, (b, a) in enumerate(bulgular):
        st.markdown(f"**{i+1}. {b}** — {a}")

    st.divider()
    st.subheader("Gelecek Çalışmalar")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown(
            """
- 🔁 **Dinamik contamination:** parametrenin zaman içindeki anomali oranına göre otomatik belirlenmesi
- 📈 **Zamansal trend öznitelikleri:** pencereler arası eğilim bilgisinin meta özniteliklere eklenmesi
"""
        )
    with g2:
        st.markdown(
            """
- ⚔️ **Denetimli rövanş:** SMOTE / ağırlıklı örnekleme + eşik kalibrasyonuyla BiLSTM'lerin yeniden değerlendirilmesi
- 🧩 **Hibrit ensemble:** gözetimsiz IF + yeniden kalibre edilmiş denetimli modellerin birleşimi
"""
        )

    st.divider()
    kaynaklar = (
        "Liu vd. (2008) *Isolation Forest*, IEEE ICDM · He vd. (2017) *Drain*, IEEE ICWS · "
        "Zhang vd. (2019) *Robust Log-Based Anomaly Detection*, ESEC/FSE · "
        "Lin vd. (2017) *Focal Loss*, ICCV · LANL BGL veri kümesi (usenix.org/cfdr-data)"
    )
    st.caption("📚 Kaynaklar: " + kaynaklar)
