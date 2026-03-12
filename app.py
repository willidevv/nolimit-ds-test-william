import streamlit as st
import pickle
import numpy as np
import torch
from transformers import BertTokenizer, AutoModel

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Analysis — SmSA",
    page_icon="🎯",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.main { background-color: #F7F9FC; }

.title-block {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.title-block h1 {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1A1A2E;
    margin-bottom: 0.3rem;
}
.title-block p {
    color: #6B7280;
    font-size: 0.95rem;
}

.result-box {
    padding: 1.5rem;
    border-radius: 16px;
    text-align: center;
    margin-top: 1.5rem;
    font-size: 1.1rem;
    font-weight: 600;
}
.positive { background-color: #DCFCE7; color: #166534; border: 2px solid #86EFAC; }
.negative { background-color: #FEE2E2; color: #991B1B; border: 2px solid #FCA5A5; }
.neutral  { background-color: #F3F4F6; color: #374151; border: 2px solid #D1D5DB; }

.confidence-bar {
    margin-top: 0.8rem;
    font-size: 0.85rem;
    color: #6B7280;
}

.info-box {
    background: #EFF6FF;
    border-left: 4px solid #3B82F6;
    padding: 0.8rem 1rem;
    border-radius: 8px;
    font-size: 0.875rem;
    color: #1E40AF;
    margin-bottom: 1.5rem;
}

.footer {
    text-align: center;
    color: #9CA3AF;
    font-size: 0.8rem;
    margin-top: 3rem;
    padding-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load model & komponen ─────────────────────────────────────
@st.cache_resource
def load_components():
    tokenizer = BertTokenizer.from_pretrained("indobenchmark/indobert-base-p2")
    model     = AutoModel.from_pretrained("indobenchmark/indobert-base-p2")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    model.eval()
    clf = pickle.load(open("clf.pkl", "rb"))
    le  = pickle.load(open("le.pkl",  "rb"))
    return tokenizer, model, device, clf, le

tokenizer, model, device, clf, le = load_components()

# ── Fungsi embedding ──────────────────────────────────────────
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

# ── UI ────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🎯 Sentiment Analysis</h1>
    <p>Analisis sentimen teks Bahasa Indonesia menggunakan IndoBERT + Logistic Regression</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    📊 Model dilatih menggunakan dataset <b>SmSA IndoNLU</b> — 
    11.000 ulasan Bahasa Indonesia dengan akurasi <b>84% pada test set</b>.
</div>
""", unsafe_allow_html=True)

# ── Input teks ────────────────────────────────────────────────
text_input = st.text_area(
    "Masukkan teks yang ingin dianalisis:",
    placeholder="Contoh: Produk ini sangat bagus dan recommended!",
    height=130
)

# ── Tombol analisis ───────────────────────────────────────────
if st.button("🔍 Analisis Sentimen", use_container_width=True):
    if text_input.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong!")
    else:
        with st.spinner("Sedang menganalisis..."):
            emb   = get_embedding(text_input).reshape(1, -1)
            pred  = clf.predict(emb)
            proba = clf.predict_proba(emb)
            label = le.inverse_transform(pred)[0]
            conf  = proba.max() * 100

        emoji_map     = {"positive": "✅", "negative": "❌", "neutral": "⚪"}
        css_map       = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
        label_indo    = {"positive": "POSITIF", "negative": "NEGATIF", "neutral": "NETRAL"}

        st.markdown(f"""
        <div class="result-box {css_map[label]}">
            {emoji_map[label]} Sentimen: <span style="font-size:1.3rem">{label_indo[label]}</span>
            <div class="confidence-bar">Confidence: {conf:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Breakdown probabilitas
        st.markdown("#### 📊 Probabilitas per Label")
        label_indo_map = {"negative": "Negatif", "neutral": "Netral", "positive": "Positif"}
        for i, lbl in enumerate(le.classes_):
            st.progress(float(proba[0][i]),
                        text=f"{label_indo_map[lbl]}: {proba[0][i]*100:.1f}%")

# ── Contoh teks ───────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 💡 Coba Contoh Teks")

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("😊 Positif", use_container_width=True):
        st.info("📝 *Produk ini sangat bagus dan recommended!*")
with col2:
    if st.button("😞 Negatif", use_container_width=True):
        st.info("📝 *Pelayanan sangat lambat dan mengecewakan.*")
with col3:
    if st.button("😐 Netral", use_container_width=True):
        st.info("📝 *Nolimit adalah sebuah perusahaan teknologi.*")

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    nolimit-ds-test-william · Dataset: SmSA IndoNLU (MIT License) · Model: indobenchmark/indobert-base-p2
</div>
""", unsafe_allow_html=True)