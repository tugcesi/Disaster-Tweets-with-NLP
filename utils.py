"""
utils.py – Paylaşılan NLP pipeline.
Çalıştırılmaz; save_model.py ve app.py tarafından import edilir.
"""

import re
import nltk
from textblob import TextBlob
from nltk.corpus import stopwords

nltk.download('stopwords',        quiet=True)
nltk.download('punkt',            quiet=True)
nltk.download('punkt_tab',        quiet=True)
nltk.download('wordnet',          quiet=True)   # ← TextBlob lemmatization için
nltk.download('omw-1.4',          quiet=True)   # ← wordnet dil desteği

stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+',     '', text)
    text = re.sub(r'\n|\r',   '', text)
    return text.strip()

def lemmatize_tokens(text: str):
    words = TextBlob(text).words
    return [word.lemmatize() for word in words if word.lower() not in stop_words]
