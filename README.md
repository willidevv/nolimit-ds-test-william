# 🎯 Sentiment Analysis — SmSA IndoNLU
**nolimit-ds-test-william**

Proyek ini merupakan submission test Data Science untuk **PT Nolimit Indonesia**, mengimplementasikan klasifikasi sentimen Bahasa Indonesia menggunakan model IndoBERT dari Hugging Face, embeddings, dan FAISS untuk similarity search.

---

## 📌 Task
**Task A — Classification**: Sentiment Analysis (positive / negative / neutral)

---

## 📊 Dataset

| Info | Detail |
|---|---|
| **Nama** | SmSA (Small-scale Multi-domain Sentiment Analysis) |
| **Sumber** | [IndoNLU Benchmark — GitHub](https://github.com/IndoNLP/indonlu) |
| **Lisensi** | [MIT License](https://github.com/IndoNLP/indonlu/blob/master/LICENSE) |
| **Bahasa** | Indonesia |
| **Label** | `positive`, `negative`, `neutral` |
| **Total Data** | Train: 11.000 | Valid: 1.260 | Test: 500 |

Dataset ini berisi komentar dan ulasan dari berbagai platform online Indonesia yang telah dianotasi secara manual oleh tim IndoNLU. Data bersumber dari domain kuliner, transportasi, produk, dan media sosial.

---

## 🤖 Model yang Digunakan

| Komponen | Model / Library |
|---|---|
| Tokenizer | `BertTokenizer` — `indobenchmark/indobert-base-p2` |
| Embedding | `AutoModel` — `indobenchmark/indobert-base-p2` (CLS token, 768 dim) |
| Similarity Search | `FAISS` — `IndexFlatL2` |
| Classifier | `Logistic Regression` — scikit-learn |

---

## 🗺️ Pipeline (Flowchart)

Lihat file `flowchart.png` atau `flowchart.pdf` di repo ini.

```
Dataset SmSA IndoNLU
        ↓
Exploratory Data Analysis (EDA)
        ↓
Preprocessing (lowercase, hapus URL & simbol)
        ↓
Tokenisasi (BertTokenizer)
        ↓
Embedding IndoBERT → CLS token (768 dim)
        ↓
FAISS Index (similarity search)
        ↓
Encode Label (LabelEncoder)
        ↓
Training Logistic Regression
        ↓
Evaluasi (Accuracy, F1-Score, Classification Report)
        ↓
Contoh Prediksi + Confidence Score
```

---

## 📈 Hasil Evaluasi

### Validation Set
| Label | Precision | Recall | F1-Score |
|---|---|---|---|
| negative | 0.86 | 0.91 | 0.88 |
| neutral | 0.78 | 0.76 | 0.77 |
| positive | 0.95 | 0.93 | 0.94 |
| **accuracy** | | | **0.91** |

### Test Set
| Label | Precision | Recall | F1-Score |
|---|---|---|---|
| negative | 0.82 | 0.95 | 0.88 |
| neutral | 0.82 | 0.60 | 0.69 |
| positive | 0.88 | 0.85 | 0.86 |
| **accuracy** | | | **0.84** |

---

## 💡 Contoh Prediksi

| Teks Input | Prediksi |
|---|---|
| "Produk ini sangat bagus dan recommended!" | ✅ positive |
| "Pelayanan sangat lambat dan mengecewakan" | ✅ negative |
| "nolimit adalah sebuah perusahaan" | ✅ neutral |

---

## 📁 Struktur Repo

```
nolimit-ds-test-william/
│
├── NOLIMIT_TEST_WILLIAM.ipynb   ← notebook utama
├── README.md                    ← dokumentasi ini
├── requirements.txt             ← daftar library
├── flowchart.png                ← flowchart pipeline (PNG)
├── flowchart.pdf                ← flowchart pipeline (PDF)
└── data/
    └── sample_smsa.csv          ← sample 100 baris dataset
```

---

## ⚙️ Setup & Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/<username>/nolimit-ds-test-william.git
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

---

## 📦 Requirements

```
transformers
torch
faiss-cpu
scikit-learn
pandas
numpy
matplotlib
seaborn
reportlab
streamlit
```

---

## 👤 Author
**William**
Submission: nolimit-ds-test-william
Email: jobs@nolimit.id
