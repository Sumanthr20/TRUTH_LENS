# 🔍 TruthLens — AI-Based Fake News Detection System

> **Academic Major Project**  
> **Student:**Sumanth

---

## 📌 Project Overview

TruthLens is an AI-powered web application that detects whether a given news article or headline is **REAL** or **FAKE** using Natural Language Processing (NLP) and Machine Learning.

The system uses a **TF-IDF vectoriser** to extract text features and a **Logistic Regression classifier** trained on labelled news data. A heuristic linguistic analyser adds an explainability layer, flagging conspiracy keywords, sensational language, and excessive capitalisation.

---

## 🗂️ Project Structure

```
fake-news-detector/
│
├── frontend/                  # Web UI (HTML + CSS + JS)
│   ├── index.html             # Main application page
│   ├── css/
│   │   └── style.css          # Complete stylesheet
│   └── js/
│       └── app.js             # Frontend logic & API calls
│
├── backend/                   # Python Flask API
│   ├── app.py                 # REST API server
│   ├── train_model.py         # ML training script
│   ├── requirements.txt       # Python dependencies
│   └── models/                # Saved model files (after training)
│       ├── fake_news_model.pkl
│       └── metrics.json
│
├── notebooks/
│   └── fake_news_analysis.ipynb   # Jupyter notebook (EDA + training)
│
├── docs/                      # Saved plots and documentation
│   ├── eda_analysis.png
│   ├── confusion_matrices.png
│   └── model_comparison.png
│
└── README.md
```

---

## ⚙️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Backend | Python 3, Flask, Flask-CORS |
| ML / NLP | scikit-learn, TF-IDF Vectorizer |
| Models | Logistic Regression, Random Forest, Naive Bayes |
| Data | Custom labelled dataset (real + fake news) |
| Notebook | Jupyter Notebook, Matplotlib, Seaborn |

---

## 🚀 How to Run

### Step 1 — Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### Step 2 — Train the Model
```bash
cd backend
python train_model.py
```
This generates `models/fake_news_model.pkl` and `models/metrics.json`.

### Step 3 — Start the Flask Server
```bash
python backend/app.py
```
Server runs at: `http://localhost:5000`

### Step 4 — Open the Web App
Open `frontend/index.html` in your browser.

> **Tip:** For the best experience, serve the frontend via Live Server (VS Code extension) or Python's built-in server:
> ```bash
> cd frontend
> python -m http.server 8080
> ```
> Then visit `http://localhost:8080`

---

## 🔌 API Endpoints

### `POST /api/predict`
Analyse a single news text.

**Request:**
```json
{ "text": "Government secretly puts microchips in vaccines!" }
```

**Response:**
```json
{
  "prediction": "FAKE",
  "is_fake": true,
  "confidence": 99.2,
  "confidence_label": "Very High",
  "probabilities": { "real": 0.8, "fake": 99.2 },
  "indicators": [...],
  "model": "Logistic Regression",
  "model_accuracy": 96.0
}
```

### `POST /api/batch`
Analyse up to 20 news texts at once.

**Request:**
```json
{ "texts": ["News headline 1", "News headline 2", ...] }
```

### `GET /api/metrics`
Get model performance metrics and dataset info.

---

## 🧠 Machine Learning Pipeline

```
Raw Text
   ↓
Preprocessing (lowercase, remove URLs, punctuation, extra spaces)
   ↓
TF-IDF Vectorisation (unigrams + bigrams, 10,000 features)
   ↓
Logistic Regression Classifier
   ↓
Prediction (REAL / FAKE) + Probability Score
   ↓
Linguistic Heuristics (explainability layer)
   ↓
Final Result with Confidence + Indicators
```

---

## 📊 Model Performance

| Model | Accuracy |
|-------|---------|
| **Logistic Regression** | **96–100%** |
| Naive Bayes | ~94% |
| Random Forest | ~83% |

*(Results on 25% test split of 72 training samples)*

---

## 🔮 Future Scope

- Integrate transformer-based models (BERT, RoBERTa, DistilBERT)
- Train on large public datasets: LIAR, FakeNewsNet, ISOT
- Add URL credibility checking
- Browser extension for real-time detection
- Multi-language support (Hindi, Kannada)
- Social media integration

---

## 📚 References

1. Ahmed, H., Traore, I., & Saad, S. (2017). *Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques.* ISDDC.
2. Shu, K., Sliva, A., Wang, S., Tang, J., & Liu, H. (2017). *Fake News Detection on Social Media: A Data Mining Perspective.* ACM SIGKDD.
3. Scikit-learn documentation: https://scikit-learn.org
4. Flask documentation: https://flask.palletsprojects.com

---

*© 2026 Sumanth*
