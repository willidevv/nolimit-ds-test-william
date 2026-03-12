# 🎯 Sentiment Analysis — SmSA IndoNLU
**nolimit-ds-test-william**

Proyek ini merupakan submission test Data Science untuk **PT NoLimit Indonesia**, mengimplementasikan klasifikasi sentimen Bahasa Indonesia menggunakan model IndoBERT dari Hugging Face, embeddings, dan FAISS untuk similarity search.

---

## 📌 Task
**Task A — Classification**: Sentiment Analysis (positive / negative / neutral)

---

## 📊 Dataset

| Info | Detail |
|---|---|
| **Nama** | SmSA (Small-scale Multi-domain Sentiment Analysis) |
| **Sumber** | [IndoNLU Benchmark — GitHub](https://github.com/IndoNLP/indonlu) |
| **Lisensi** | [Apache License 2.0](https://github.com/IndoNLP/indonlu/blob/master/LICENSE) |
| **Bahasa** | Indonesia |
| **Label** | `positive`, `negative`, `neutral` |
| **Total Data** | Train: 11.000 | Valid: 1.260 | Test: 500 |

Dataset ini berisi komentar dan ulasan dari berbagai platform online Indonesia yang telah dianotasi secara manual oleh tim IndoNLU. Data bersumber dari domain kuliner, transportasi, produk, dan media sosial.

---

## ⚙️ Setup & Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/willidevv/nolimit-ds-test-william.git
cd nolimit-ds-test-william
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Notebook
Buka `NOLIMIT_TEST_WILLIAM.ipynb` di **Google Colab** atau Jupyter Notebook.

> ⚠️ **Disarankan menggunakan Google Colab dengan GPU (T4)** untuk mempercepat proses pembuatan embedding.
>
> Di Google Colab: Runtime → Change runtime type → T4 GPU

### 4. Dataset
Dataset akan otomatis didownload dari GitHub IndoNLU saat notebook dijalankan. Tidak perlu setup manual.

## 🤖 Model yang Digunakan

| Komponen | Model / Library |
|---|---|
| Tokenizer | `BertTokenizer` — `indobenchmark/indobert-base-p2` |
| Embedding | `AutoModel` — `indobenchmark/indobert-base-p2` (CLS token, 768 dim) |
| Similarity Search | `FAISS` — `IndexFlatL2` |
| Classifier | `Logistic Regression` — scikit-learn |
