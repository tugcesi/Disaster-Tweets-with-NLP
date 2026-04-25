# 🚨 NLP with Disaster Tweets

Kaggle **NLP Getting Started** yarışması için geliştirilmiş bir doğal dil işleme projesi.  
Bir tweet'in gerçek bir felaketi anlatıp anlatmadığını sınıflandırır: **1 = Felaket**, **0 = Felaket Değil**

---

## 📁 Proje Yapısı

```
├── NLPwithDisasterTweets.ipynb   # Keşifsel analiz & model denemeleri
├── save_model.py                 # Model eğitimi & artifact kaydetme
├── app.py                        # Streamlit web uygulaması
├── requirements.txt              # Bağımlılıklar
├── train.csv                     # Eğitim verisi (Kaggle'dan indirilir)
├── model.joblib                  # Eğitilmiş model (save_model.py ile oluşur)
└── vectorizer.joblib             # CountVectorizer (save_model.py ile oluşur)
```

---

## 📊 Veri Seti

**Kaynak:** [Kaggle – NLP with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

| Sütun      | Açıklama |
|---|---|
| `id`       | Tweet kimliği |
| `keyword`  | Tweetle ilgili anahtar kelime (bazıları eksik) |
| `location` | Kullanıcı konumu (çok eksik) |
| `text`     | Tweet metni |
| `target`   | 🎯 **1** = Gerçek felaket, **0** = Felaket değil |

- **Train:** 7,613 satır  
- **Test:** 3,263 satır

---

## ⚙️ Pipeline

### 1. Metin Temizleme
```python
text = text.lower()                         # küçük harfe çevir
text = re.sub(r'[^\w\s]', '', text)        # noktalama kaldır
text = re.sub(r'\d+',     '', text)        # rakamları kaldır
text = re.sub(r'\n|\r',   '', text)        # satır sonlarını kaldır
```

### 2. Tokenizasyon & Lemmatizasyon
```python
# TextBlob lemmatization + NLTK stopwords
def lemmatize_tokens(text):
    words = TextBlob(text).words
    return [word.lemmatize() for word in words if word not in stop_words]
```

### 3. Vektörizasyon
```python
CountVectorizer(
    ngram_range=(1, 2),        # unigram + bigram
    analyzer=lemmatize_tokens, # özel tokenizer
    stop_words='english',      # ek stopword filtresi
    max_features=50000
)
```

### 4. Model
```python
LogisticRegression(max_iter=1000, C=1.0, random_state=42)
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkları Kur

```bash
pip install -r requirements.txt
```

### 2. NLTK Verilerini İndir (ilk çalıştırmada otomatik)

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### 3. Veriyi İndir

[Kaggle](https://www.kaggle.com/competitions/nlp-getting-started/data) sayfasından `train.csv` dosyasını indirip proje klasörüne koy.

### 4. Modeli Eğit

```bash
python save_model.py
```

Bu komut şu dosyaları oluşturur:
- `model.joblib` — Logistic Regression modeli
- `vectorizer.joblib` — CountVectorizer (fit edilmiş)

### 5. Uygulamayı Başlat

```bash
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` adresini aç.

---

## 🖥️ Uygulama Özellikleri

- 📝 **Tweet giriş alanı** — serbest metin girişi
- 💡 **Örnek tweetler** — hazır felaket / felaket olmayan örnekler
- 📊 **Gauge chart** — felaket olasılığını görsel gösterir
- 📈 **Bar chart** — iki sınıf olasılıklarını karşılaştırır
- 🔬 **Metin analiz detayı** — temizlenmiş metin ve lemmatize tokenları gösterir

---

## 🔗 Bağlantılar

- 🏆 [Kaggle Yarışma Sayfası](https://www.kaggle.com/competitions/nlp-getting-started)

---

## 📄 Lisans

MIT License
