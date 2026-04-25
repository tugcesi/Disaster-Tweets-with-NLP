"""
app.py – Hugging Face Spaces / Streamlit
HF Spaces bu dosyayı otomatik olarak çalıştırır.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from utils import clean_text, lemmatize_tokens   # ← utils modülünden

# ── Sayfa ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚨 Disaster Tweet Classifier",
    page_icon="🚨",
    layout="wide"
)

# ── Artifact Yükleme (HF'de dosyalar app.py ile aynı dizinde) ────────────────
@st.cache_resource
def load_artifacts():
    base_dir = os.path.dirname(__file__)   # app.py'nin bulunduğu klasör
    mp = os.path.join(base_dir, 'model.joblib')
    vp = os.path.join(base_dir, 'vectorizer.joblib')

    if not os.path.exists(mp) or not os.path.exists(vp):
        return None, None, f"Dosyalar bulunamadı: {mp}"

    return joblib.load(mp), joblib.load(vp), None

model, vect, err = load_artifacts()

if err:
    st.error(f"⚠️ {err}")
    st.info("Yerel ortamda `python save_model.py` çalıştırıp .joblib dosyalarını repoya ekle.")
    st.stop()

# ── Tahmin ────────────────────────────────────────────────────────────────────
def predict_tweet(tweet: str):
    cleaned    = clean_text(tweet)
    vectorized = vect.transform([cleaned])
    pred       = int(model.predict(vectorized)[0])
    proba      = model.predict_proba(vectorized)[0]
    return pred, proba

# ── Başlık ────────────────────────────────────────────────────────────────────
st.title("🚨 Disaster Tweet Classifier")
st.caption("Kaggle NLP Getting Started | Logistic Regression + CountVectorizer (ngram 1-2)")
st.divider()

# ── Layout ────────────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

DISASTER_EXAMPLES = [
    "Massive earthquake hits the city, buildings collapsing",
    "Wildfire spreading rapidly, thousands evacuated",
    "Flood warning issued, roads are closed",
]
NON_DISASTER_EXAMPLES = [
    "The traffic today was a complete disaster ugh",
    "This burger is so good it's like an explosion of flavor",
    "I'm on fire today! Finished all my tasks early",
]

# Session state ile örnek seçimi
if 'tweet_input' not in st.session_state:
    st.session_state['tweet_input'] = ''

with col_input:
    st.subheader("📝 Tweet Girin")

    tweet_input = st.text_area(
        "Tweet:",
        value=st.session_state['tweet_input'],
        placeholder="Örn: There is a fire in the forest, people are evacuating...",
        height=150,
        label_visibility="collapsed"
    )

    st.caption("💡 Örnek tweetler:")
    ex_col1, ex_col2 = st.columns(2)

    with ex_col1:
        st.markdown("🔴 **Felaket**")
        for ex in DISASTER_EXAMPLES:
            if st.button(f"📌 {ex[:35]}…", key=f"d_{ex}", use_container_width=True):
                st.session_state['tweet_input'] = ex
                st.rerun()

    with ex_col2:
        st.markdown("🟢 **Felaket Değil**")
        for ex in NON_DISASTER_EXAMPLES:
            if st.button(f"📌 {ex[:35]}…", key=f"n_{ex}", use_container_width=True):
                st.session_state['tweet_input'] = ex
                st.rerun()

    predict_btn = st.button(
        "🔍 Tahmin Et", type="primary", use_container_width=True,
        disabled=(len(tweet_input.strip()) == 0)
    )

# ── Sonuç ─────────────────────────────────────────────────────────────────────
with col_result:
    st.subheader("📊 Sonuç")

    if predict_btn and tweet_input.strip():
        try:
            pred, proba = predict_tweet(tweet_input)
            prob_disaster     = proba[1]
            prob_non_disaster = proba[0]

            if pred == 1:
                st.error("### 🚨 GERÇEK FELAKET")
                result_label = "Felaket"
                bar_color    = "#DC2626"
            else:
                st.success("### ✅ FELAKET DEĞİL")
                result_label = "Felaket Değil"
                bar_color    = "#16A34A"

            st.metric("Tahmin", result_label,
                      delta=f"Güven: %{max(prob_disaster, prob_non_disaster)*100:.1f}")

            # Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_disaster * 100,
                title={'text': "Felaket Olasılığı (%)"},
                number={'suffix': "%", 'valueformat': '.1f'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar':  {'color': bar_color},
                    'steps': [
                        {'range': [0,  40], 'color': '#D1FAE5'},
                        {'range': [40, 60], 'color': '#FEF9C3'},
                        {'range': [60, 100],'color': '#FFE4E6'},
                    ],
                    'threshold': {'line': {'color': 'black', 'width': 3},
                                  'thickness': 0.8, 'value': 50}
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(t=30, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Bar
            fig_bar = go.Figure(go.Bar(
                x=[prob_non_disaster * 100, prob_disaster * 100],
                y=['Felaket Değil', 'Felaket'],
                orientation='h',
                marker_color=['#16A34A', '#DC2626'],
                text=[f"%{prob_non_disaster*100:.1f}", f"%{prob_disaster*100:.1f}"],
                textposition='inside',
                textfont=dict(color='white', size=14)
            ))
            fig_bar.update_layout(
                xaxis=dict(range=[0, 100], title="Olasılık (%)"),
                height=160, margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ Hata: {e}")
            st.exception(e)

    else:
        st.info("👈 Sol tarafa bir tweet girin ve **Tahmin Et** butonuna tıklayın.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Model",      "Logistic Regression")
        m2.metric("Vektörizör", "CountVectorizer")
        m3.metric("N-gram",     "1–2")

# ── Detay ─────────────────────────────────────────────────────────────────────
if predict_btn and tweet_input.strip():
    st.divider()
    st.subheader("🔬 Metin Analiz Detayı")

    cleaned = clean_text(tweet_input)
    tokens  = lemmatize_tokens(cleaned)
    d1, d2  = st.columns(2)

    with d1:
        st.markdown("**Orijinal Tweet:**")
        st.text_area("", tweet_input, height=80, disabled=True, label_visibility="collapsed")
        st.markdown("**Temizlenmiş Metin:**")
        st.text_area("", cleaned,     height=80, disabled=True, label_visibility="collapsed")

    with d2:
        st.markdown(f"**Tokenlar ({len(tokens)} adet):**")
        if tokens:
            st.dataframe(
                pd.DataFrame({'Token': tokens, 'Uzunluk': [len(t) for t in tokens]}),
                hide_index=True, use_container_width=True, height=170
            )
        else:
            st.warning("Stopword temizleme sonrası token kalmadı.")

# ── Hakkında ──────────────────────────────────────────────────────────────────
with st.expander("ℹ️ Proje Hakkında"):
    st.markdown("""
    **Kaynak:** [Kaggle – NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

    | | |
    |---|---|
    | Train | 7,613 satır |
    | Hedef | 0 = Felaket Değil · 1 = Felaket |

    **Pipeline:** `clean_text` → `lemmatize_tokens` (TextBlob + NLTK) → `CountVectorizer(1-2)` → `LogisticRegression`
    """)