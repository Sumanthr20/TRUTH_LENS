"""
AI-Based Fake News Detection System - Flask API
Student: Sumanth 
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import json
import os
import logging
import urllib.request
import urllib.error
import time

app = Flask(__name__,
            static_folder=os.path.join(os.path.dirname(__file__), '..', 'frontend'),
            static_url_path='')
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Load Model ───────────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), 'models', 'fake_news_model.pkl')
METRICS_PATH = os.path.join(os.path.dirname(__file__), 'models', 'metrics.json')

model   = None
metrics = {}

def load_model():
    global model, metrics
    try:
        model = joblib.load(MODEL_PATH)
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
    try:
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        logger.info("✅ Metrics loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to load metrics: {e}")

load_model()

# ─── Text Preprocessing ───────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_confidence_label(prob):
    if prob >= 0.85:   return "Very High"
    elif prob >= 0.70: return "High"
    elif prob >= 0.55: return "Moderate"
    else:              return "Low"

def analyze_indicators(text):
    indicators = []
    text_lower = text.lower()
    sensational_words = ['shocking','breaking','exposed','secret','bombshell',
                         'urgent','confirmed','exclusive','they wont tell you',
                         'what they dont want','cover up','coverup','hidden truth']
    found = [w for w in sensational_words if w in text_lower]
    if found:
        indicators.append({"type":"warning","text":f"Sensational language detected: {', '.join(found[:3])}"})
    conspiracy_words = ['deep state','new world order','illuminati','microchip',
                        'mind control','chemtrail','big pharma','lizard','alien',
                        'crisis actor','false flag','reptilian']
    found_c = [w for w in conspiracy_words if w in text_lower]
    if found_c:
        indicators.append({"type":"danger","text":f"Conspiracy keywords found: {', '.join(found_c[:3])}"})
    source_words = ['study shows','according to','researchers found','published in',
                    'scientists say','experts say','per cent','percent','data shows']
    found_s = [w for w in source_words if w in text_lower]
    if found_s:
        indicators.append({"type":"positive","text":f"Factual language detected: {', '.join(found_s[:3])}"})
    if text.count('!') > 2:
        indicators.append({"type":"warning","text":"Excessive exclamation marks (emotional manipulation)"})
    caps_words = [w for w in text.split() if w.isupper() and len(w) > 2]
    if len(caps_words) > 3:
        indicators.append({"type":"warning","text":f"All-caps words detected: {' '.join(caps_words[:4])}"})
    word_count = len(text.split())
    if word_count < 15:
        indicators.append({"type":"info","text":"Short text — limited context for analysis"})
    elif word_count > 50:
        indicators.append({"type":"positive","text":"Sufficient text length for reliable analysis"})
    return indicators

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return app.send_static_file('index.html')

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "project": "AI-Based Fake News Detection System",
        "version": "1.0.0",
        "student": "Seema | R23DB036",
        "university": "Reva University",
        "guide": "Prof. Prabhu G.",
        "status": "running",
        "model_loaded": model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 500
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in request body"}), 400
    text = data['text'].strip()
    if len(text) < 10:
        return jsonify({"error": "Text too short. Please provide at least 10 characters."}), 400
    cleaned    = clean_text(text)
    prediction = int(model.predict([cleaned])[0])
    proba      = model.predict_proba([cleaned])[0]
    fake_prob  = float(proba[1])
    real_prob  = float(proba[0])
    confidence = max(fake_prob, real_prob)
    label      = "FAKE" if prediction == 1 else "REAL"
    indicators = analyze_indicators(text)
    return jsonify({
        "prediction":       label,
        "is_fake":          bool(prediction == 1),
        "confidence":       round(confidence * 100, 2),
        "confidence_label": get_confidence_label(confidence),
        "probabilities":    {"fake": round(fake_prob*100,2), "real": round(real_prob*100,2)},
        "indicators":       indicators,
        "word_count":       len(text.split()),
        "model":            metrics.get('best_model', 'Logistic Regression'),
        "model_accuracy":   round(metrics.get('best_accuracy', 0.95) * 100, 2)
    })

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify({
        "model_performance": metrics.get('all_results', {}),
        "best_model":        metrics.get('best_model', ''),
        "best_accuracy":     metrics.get('best_accuracy', 0),
        "dataset_size":      metrics.get('dataset_size', 0),
        "train_size":        metrics.get('train_size', 0),
        "test_size":         metrics.get('test_size', 0)
    })

@app.route('/api/batch', methods=['POST'])
def batch_predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({"error": "Missing 'texts' field in request body"}), 400
    texts = data['texts']
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "'texts' must be a non-empty list"}), 400
    if len(texts) > 20:
        return jsonify({"error": "Maximum 20 texts per batch request"}), 400
    results = []
    for text in texts:
        text       = str(text).strip()
        cleaned    = clean_text(text)
        pred       = int(model.predict([cleaned])[0])
        proba      = model.predict_proba([cleaned])[0]
        fake_prob  = float(proba[1])
        confidence = max(float(proba[0]), fake_prob)
        results.append({
            "text":       text[:100] + "..." if len(text) > 100 else text,
            "prediction": "FAKE" if pred == 1 else "REAL",
            "is_fake":    bool(pred == 1),
            "confidence": round(confidence * 100, 2)
        })
    fake_count = sum(1 for r in results if r['is_fake'])
    return jsonify({
        "results": results,
        "summary": {"total": len(results), "fake_count": fake_count, "real_count": len(results)-fake_count}
    })

# ─── Groq AI Proxy ────────────────────────────────────────────────────────────
@app.route('/api/gemini', methods=['POST'])
def gemini_proxy():
    GROQ_API_KEY = "YOUR_API_KEY_HERE"   # gsk_...
    GROQ_URL = 'API_URL_HERE'

    payload = request.get_json(force=True)
    # Extract the prompt text from Gemini-format request
    prompt_text = payload['contents'][0]['parts'][0]['text']

    groq_body = json.dumps({
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.2
    }).encode('utf-8')

    req = urllib.request.Request(
        GROQ_URL, data=groq_body,
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {GROQ_API_KEY}'
        },
        method='POST'
    )
    try:
        with urllib.request.urlopen(req) as resp:
            groq_data = json.loads(resp.read().decode('utf-8'))
            # Convert Groq response → Gemini-style so app.js needs no changes
            text = groq_data['choices'][0]['message']['content']
            gemini_style = {
                "candidates": [{"content": {"parts": [{"text": text}]}}]
            }
            return jsonify(gemini_style)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8')
        logger.error(f"Groq error: {e.code} — {error_body}")
        return app.response_class(response=error_body, status=e.code, mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)