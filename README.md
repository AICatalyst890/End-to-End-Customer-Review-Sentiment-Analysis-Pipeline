# 🎬 End-to-End Customer Review Sentiment Analysis Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](#)
[![NLTK](https://img.shields.io/badge/NLTK-3776AB?style=flat&logo=python&logoColor=white)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](#)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Machine Learning & Deep Learning pipeline engineered to transform unstructured customer feedback into real-time sentiment analytics. Powered by **PyTorch** sequence modeling, **NLTK** textual preprocessing, and an interactive **Streamlit** dashboard for live inference and scoring.

---

## 📌 Business Context

In modern e-commerce architectures (e.g., Flipkart, eBay, Amazon), real-time customer review processing directly informs product quality metrics, seller performance, and customer satisfaction scores. Manually analyzing high-volume customer feedback is inefficient and costly.

This project delivers an automated **End-to-End Sentiment Engine** capable of:
* Classifying unstructured review text into binary sentiment categories (**Positive** vs. **Negative**).
* Generating fine-grained probability scores and confidence metrics for product analytics.
* Decoupling training workflows from inference services to ensure sub-second deployment overhead.

---

## ⚙️ Data Flow & Model Architecture

The operational pipeline transforms raw text inputs through custom NLTK normalization routines, maps tokens to high-dimensional TF-IDF matrices, reshapes feature matrices into 3D sequence tensors, and routes them through a Recurrent Neural Network (RNN) before outputting classification probabilities.

```text
               +----------------------------------+
               |     Raw Review Text Input        |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | Lowercasing & Regex URL Cleaning |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | NLTK Tokenization & Stemming     |
               | (word_tokenize, PorterStemmer)   |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |  TF-IDF Feature Extraction       |
               |  (scikit-learn TfidfVectorizer)  |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | Tensor Reshaping & Alignment     |
               | Shape: (Batch=1, Seq_Len=1, Dim) |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | PyTorch Recurrent Neural Network |
               | nn.RNN(hidden_size=128)          |
               +----------------------------------+
                                |  (Last Hidden State: out[:, -1, :])
                                v
               +----------------------------------+
               | Linear Layer & Sigmoid Threshold |
               | nn.Linear(128, 1) + torch.sigmoid|
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | Class Prediction & Confidence    |
               | Prob >= 0.5 -> POSITIVE          |
               | Prob <  0.5 -> NEGATIVE          |
               +----------------------------------+

```

---

## 🌟 Key Enterprise Features

* **Decoupled Training & Inference:** Model state weights (`.pth`) and feature vectorizers (`.joblib`) are serialized independently, enabling fast, lightweight inference services in production without retraining.
* **Dynamic Tensor Manipulation:** Preprocessed feature vectors are dynamically converted to PyTorch FloatTensors and reshaped using `unsqueeze(1)` to align sparse TF-IDF vectors with PyTorch 3D sequence expectations `(Batch, Seq_Len, Features)`.
* **Integrated NLTK Preprocessing Pipeline:** Features explicit text normalization incorporating regex link stripping, stopword filtering, and Porter Stemming for noise reduction before vectorization.
* **Interactive Web Analytics UI:** Streamlit interface provides live input text analysis, real-time positivity gauge progress bars, and percentage confidence scoring.

---

## 🛠️ Technology Stack

* **Core Language:** Python 3.8+
* **Deep Learning Framework:** PyTorch (`torch.nn`, `torch.optim`)
* **NLP & ML Frameworks:** NLTK (`tokenize`, `corpus`, `stem`), Scikit-Learn (`TfidfVectorizer`), Joblib
* **Data Processing:** Pandas, NumPy
* **Application / UI:** Streamlit

---

## 📁 Repository Structure

```text
.
├── RNN.ipynb                # Notebook: NLTK preprocessing, model training, evaluation & artifact export
├── app.py                   # Interactive Streamlit web application & inference engine
├── rnn_model.pth            # Serialized PyTorch trained model weights
├── tfidf_vectorizer.joblib  # Serialized fitted TF-IDF feature vectorizer
├── requirements.txt         # Project dependencies (PyTorch, NLTK, Streamlit, etc.)
├── IMDB Dataset.csv         # Customer review dataset
└── README.md                # Project documentation

```

---

## 🚀 Quick Start Guide

### 1. Prerequisites

Ensure **Python 3.8+** is installed on your environment.

### 2. Clone Repository & Setup Environment

```bash
# Clone repository
git clone https://github.com/AICatalyst890/End-to-End-Customer-Review-Sentiment-Analysis-Pipeline.git
cd End-to-End-Customer-Review-Sentiment-Analysis-Pipeline

# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### 3. Download Required NLTK Corpora

Run the following python snippet once to download required NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

```

### 4. Launch Streamlit Application

Run the following command to start the web application:

```bash
streamlit run app.py

```

> **Note for Windows Users:**
> If you encounter an OpenMP error (`libiomp5md.dll already initialized`), set the environment variable in your terminal prior to running Streamlit:

**PowerShell:**

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
streamlit run app.py

```

**Command Prompt (cmd):**

```cmd
set KMP_DUPLICATE_LIB_OK=TRUE
streamlit run app.py

```

---

## 📊 Sentiment Thresholding Logic

| Positivity Probability Score | Predicted Class | Action / Label |
| --- | --- | --- |
| **0.50 – 1.00 (≥ 50%)** | `POSITIVE` | Positive Customer Review |
| **0.00 – 0.49 (< 50%)** | `NEGATIVE` | Flagged Negative Review |

---

## 📄 License

This project is licensed under the [MIT License](https://www.google.com/search?q=MITLICENSE).
---
