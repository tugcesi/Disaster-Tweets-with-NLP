"""
save_model.py – Sadece yerel bilgisayarda çalıştırılır.
Üretilen model.joblib ve vectorizer.joblib dosyalarını HF'e yükle.

Çalıştır: python save_model.py
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from utils import clean_text, lemmatize_tokens   # ← utils modülünden

print("📂 Veri yükleniyor...")
train = pd.read_csv('train.csv')

print("⚙️  Temizleniyor...")
train['text_clean'] = train['text'].apply(clean_text)

x_train, x_val, y_train, y_val = train_test_split(
    train['text_clean'], train['target'],
    test_size=0.2, random_state=42, stratify=train['target']
)

print("🔢 Vektörizasyon...")
vect = CountVectorizer(
    ngram_range=(1, 2),
    analyzer=lemmatize_tokens,   # utils.lemmatize_tokens olarak kaydedilir ✓
    stop_words='english',
    max_features=50000
)
x_train_v = vect.fit_transform(x_train)
x_val_v   = vect.transform(x_val)

print("🚀 Model eğitiliyor...")
model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
model.fit(x_train_v, y_train)

preds = model.predict(x_val_v)
print(f"\n📊 Accuracy : {accuracy_score(y_val, preds):.4f}")
print(classification_report(y_val, preds, target_names=['Non-Disaster', 'Disaster']))

joblib.dump(model, 'model.joblib')
joblib.dump(vect,  'vectorizer.joblib')
print("\n✅ model.joblib ve vectorizer.joblib kaydedildi → HF'e yükle!")