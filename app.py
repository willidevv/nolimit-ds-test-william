import streamlit as st
import pickle
import torch
from transformers import BertTokenizer, AutoModel

# Load model
@st.cache_resource
def load_components():
    tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
    model     = AutoModel.from_pretrained("indobenchmark/indobert-base-p2")
    model.eval()
    clf = pickle.load(open("clf.pkl", "rb"))
    le  = pickle.load(open("le.pkl",  "rb"))
    return tokenizer, model, clf, le

tokenizer, model, clf, le = load_components()

# Fungsi embedding
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

# UI
st.title("🎯 Sentiment Analysis")
st.caption("Analisis sentimen teks Bahasa Indonesia menggunakan IndoBERT")

text_input = st.text_area("Masukkan teks:", placeholder="Contoh: Produk ini sangat bagus!")

if st.button("Analisis"):
    if text_input.strip() == "":
        st.warning("Teks tidak boleh kosong!")
    else:
        with st.spinner("Menganalisis..."):
            emb   = get_embedding(text_input).reshape(1, -1)
            pred  = clf.predict(emb)
            proba = clf.predict_proba(emb)
            label = le.inverse_transform(pred)[0]
            conf  = proba.max() * 100

        emoji = {"positive": "✅", "negative": "❌", "neutral": "⚪"}
        indo  = {"positive": "POSITIF", "negative": "NEGATIF", "neutral": "NETRAL"}

        st.success(f"{emoji[label]} {indo[label]} ({conf:.1f}%)")