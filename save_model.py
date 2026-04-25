"""
save_model.py – Disaster Tweet Classifier
Notebook pipeline ile birebir uyumlu:
  1. train.csv yükle
  2. Metin temizleme (lowercase, noktalama, rakam, satır sonu)
  3. CountVectorizer (ngram 1-2, TextBlob lemmatize, NLTK stopwords)
  4. Logistic Regression eğit
  5. model.joblib + vectorizer.joblib kaydet

Çalıştır: python save_model.py
"""

import warnings
warnings.filterwarnings('ignore')

import re
import nltk
import numpy as np
import pandas as pd
import joblib

from textblob import TextBlob
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ── NLTK İndir ────────────────────────────────────────────────────────────────
nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
stop_words = set(stopwords.words('english'))

# ── 1. Metin Temizleme (notebook ile birebir) ─────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)   # noktalama işaretleri
    text = re.sub(r'\d+',     '', text)   # rakamlar
    text = re.sub(r'\n',      '', text)   # satır sonu
    text = re.sub(r'\r',      '', text)   # enter
    return text.strip()

# ── 2. Lemmatize + Stopword Analiz Fonksiyonu (notebook ile birebir) ──────────
def lemmatize_tokens(text: str):
    words = TextBlob(text).words
    return [word.lemmatize() for word in words if word.lower() not in stop_words]

# ── 3. Veri Yükle ─────────────────────────────────────────────────────────────
print("📂 Veri yükleniyor...")
train = pd.read_csv('train.csv')
print(f"   Train shape: {train.shape}")

# ── 4. Preprocessing ──────────────────────────────────────────────────────────
print("⚙️  Metin temizleniyor...")
train['text_clean'] = train['text'].apply(clean_text)

x = train['text_clean']
y = train['target']

# ── 5. Train / Val Split ──────────────────────────────────────────────────────
x_train, x_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# ── 6. Vektörizasyon ──────────────────────────────────────────────────────���───
print("🔢 Vektörizasyon (CountVectorizer ngram 1-2 + lemmatize)...")
vect = CountVectorizer(
    ngram_range=(1, 2),
    analyzer=lemmatize_tokens,
    stop_words='english',
    max_features=50000
)
x_train_vect = vect.fit_transform(x_train)
x_val_vect   = vect.transform(x_val)

# ── 7. Model Eğitimi ──────────────────────────────────────────────────────────
print("🚀 Model eğitiliyor (Logistic Regression)...")
model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
model.fit(x_train_vect, y_train)

# ── 8. Değerlendirme ──────────────────────────────────────────────────────────
preds = model.predict(x_val_vect)
acc   = accuracy_score(y_val, preds)

print(f"\n📊 Validation Sonuçları:")
print(f"   Accuracy : {acc:.4f}")
print()
print(classification_report(y_val, preds, target_names=['Non-Disaster', 'Disaster']))

# ── 9. Artifact Kaydet ────────────────────────────────────────────────────────
joblib.dump(model, 'model.joblib')
joblib.dump(vect,  'vectorizer.joblib')

print("✅ Kaydedilen dosyalar:")
print("   model.joblib")
print("   vectorizer.joblib")