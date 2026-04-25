"""
app.py – Disaster Tweet Classifier (Streamlit)
Notebook pipeline ile birebir uyumlu.
Çalıştır: streamlit run app.py
"""

import warnings
warnings.filterwarnings('ignore')

import os
import re
import nltk
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from textblob import TextBlob
from nltk.corpus import stopwords

# ── NLTK ──────────────────────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
stop_words = set(stopwords.words('english'))

# ── Sayfa Ayarları ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚨 Disaster Tweet Classifier",
    page_icon="🚨",
    layout="wide"
)

# ── Artifact Yükleme ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    files = ['model.joblib', 'vectorizer.joblib']
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        return None, None, f"Eksik dosyalar: {missing}"
    model = joblib.load('model.joblib')
    vect  = joblib.load('vectorizer.joblib')
    return model, vect, None

model, vect, err = load_artifacts()

if err:
    st.error(f"⚠️ {err}")
    st.info("Önce şunu çalıştırın: `python save_model.py`")
    st.stop()

# ── Pipeline Fonksiyonları (notebook ile birebir) ────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+',     '', text)
    text = re.sub(r'\n',      '', text)
    text = re.sub(r'\r',      '', text)
    return text.strip()

def lemmatize_tokens(text: str):
    words = TextBlob(text).words
    return [word.lemmatize() for word in words if word.lower() not in stop_words]

def predict_tweet(tweet: str):
    cleaned   = clean_text(tweet)
    vectorized = vect.transform([cleaned])
    pred       = model.predict(vectorized)[0]
    proba      = model.predict_proba(vectorized)[0]
    return int(pred), proba

# ── Başlık ────────────────────────────────────────────────────────────────────
st.title("🚨 Disaster Tweet Classifier")
st.caption("Kaggle NLP Getting Started | Logistic Regression + CountVectorizer (ngram 1-2)")
st.divider()

# ── Ana Layout ────────────────────────────────────────────────────────────────
col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📝 Tweet Girin")

    tweet_input = st.text_area(
        "Tweet metni:",
        placeholder="Örn: There is a fire in the forest, people are evacuating...",
        height=150,
        label_visibility="collapsed"
    )

    # Örnek tweetler
    st.caption("💡 Örnek tweetler:")
    ex_col1, ex_col2 = st.columns(2)

    disaster_examples = [
        "Massive earthquake hits the city, buildings collapsing",
        "Wildfire spreading rapidly, thousands evacuated",
        "Flood warning issued, roads are closed",
    ]
    non_disaster_examples = [
        "The traffic today was a complete disaster ugh",
        "This burger is so good it's like an explosion of flavor",
        "I'm on fire today! Finished all my tasks early",
    ]

    with ex_col1:
        st.markdown("🔴 **Felaket Örnekleri**")
        for ex in disaster_examples:
            if st.button(f"📌 {ex[:40]}...", key=ex, use_container_width=True):
                tweet_input = ex

    with ex_col2:
        st.markdown("🟢 **Felaket Olmayan Örnekler**")
        for ex in non_disaster_examples:
            if st.button(f"📌 {ex[:40]}...", key=ex, use_container_width=True):
                tweet_input = ex

    predict_btn = st.button(
        "🔍 Tahmin Et", type="primary", use_container_width=True,
        disabled=(len(tweet_input.strip()) == 0)
    )

# ── Tahmin ────────────────────────────────────────────────────────────────────
with col_result:
    st.subheader("📊 Sonuç")

    if predict_btn and tweet_input.strip():
        try:
            pred, proba = predict_tweet(tweet_input)
            prob_disaster     = proba[1]
            prob_non_disaster = proba[0]

            if pred == 1:
                st.error("### 🚨 GERÇEK FELAKET")
                result_color = "#FCA5A5"
                result_emoji = "🚨"
                result_label = "Felaket"
            else:
                st.success("### ✅ FELAKET DEĞİL")
                result_color = "#86EFAC"
                result_emoji = "✅"
                result_label = "Felaket Değil"

            # Olasılık göstergesi
            st.metric(
                f"{result_emoji} Tahmin",
                result_label,
                delta=f"Güven: %{max(prob_disaster, prob_non_disaster)*100:.1f}"
            )

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob_disaster * 100,
                title={'text': "Felaket Olasılığı (%)"},
                number={'suffix': "%", 'valueformat': '.1f'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar':  {'color': '#DC2626' if pred == 1 else '#16A34A'},
                    'steps': [
                        {'range': [0,  40], 'color': '#D1FAE5'},
                        {'range': [40, 60], 'color': '#FEF9C3'},
                        {'range': [60, 100],'color': '#FFE4E6'},
                    ],
                    'threshold': {
                        'line': {'color': 'black', 'width': 3},
                        'thickness': 0.8,
                        'value': 50
                    }
                }
            ))
            fig_gauge.update_layout(height=260, margin=dict(t=30, b=10, l=20, r=20))
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Olasılık bar chart
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
                height=160,
                margin=dict(t=10, b=10, l=10, r=10),
                showlegend=False
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        except Exception as e:
            st.error(f"⚠️ Hata: {e}")

    elif not tweet_input.strip():
        st.info("👈 Sol tarafa bir tweet girin ve **Tahmin Et** butonuna tıklayın.")

        # Karşılama metrikleri
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Model",      "Logistic Regression")
        m2.metric("Vektörizör", "CountVectorizer")
        m3.metric("N-gram",     "1–2")

# ── Metin Analiz Detayı ───────────────────────────────────────────────────────
if predict_btn and tweet_input.strip():
    st.divider()
    st.subheader("🔬 Metin Analiz Detayı")

    cleaned = clean_text(tweet_input)
    tokens  = lemmatize_tokens(cleaned)

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        st.markdown("**Orijinal Tweet:**")
        st.text_area("", tweet_input, height=80, disabled=True, label_visibility="collapsed")

        st.markdown("**Temizlenmiş Metin:**")
        st.text_area("", cleaned, height=80, disabled=True, label_visibility="collapsed")

    with detail_col2:
        st.markdown(f"**Tokenlar ({len(tokens)} adet):**")
        if tokens:
            token_df = pd.DataFrame({'Token': tokens, 'Uzunluk': [len(t) for t in tokens]})
            st.dataframe(token_df, hide_index=True, use_container_width=True, height=170)
        else:
            st.warning("Stopword temizleme sonrası token kalmadı.")

# ── Hakkında ──────────────────────────────────────────────────────────────────
with st.expander("ℹ️ Proje Hakkında"):
    st.markdown("""
    **Veri Seti:** [Kaggle NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

    | | Değer |
    |---|---|
    | Train Satır | 7,613 |
    | Test Satır  | 3,263 |
    | Sütunlar    | id, keyword, location, text, target |
    | Hedef       | 0 = Felaket Değil, 1 = Gerçek Felaket |

    **Pipeline:**
    1. 📝 Metin temizleme (lowercase, noktalama/rakam/satır sonu kaldırma)
    2. 🔤 TextBlob lemmatization + NLTK İngilizce stopwords
    3. 🔢 CountVectorizer (ngram_range=(1,2), max_features=50,000)
    4. 🤖 Logistic Regression (C=1.0, max_iter=1000)
    """)