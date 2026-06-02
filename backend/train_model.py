"""
AI-Based Fake News Detection System
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import re
import string
import os

# ─── Synthetic Training Data ──────────────────────────────────────────────────
REAL_NEWS = [
    "Scientists at NASA confirmed the launch of the Artemis mission scheduled for next year, citing successful tests.",
    "The Federal Reserve raised interest rates by 25 basis points in response to ongoing inflation pressures.",
    "Researchers at Johns Hopkins University published a peer-reviewed study on the efficacy of new cancer treatments.",
    "The United Nations Security Council convened an emergency session to address the escalating humanitarian crisis.",
    "Apple reported quarterly earnings that exceeded analyst expectations, with iPhone sales up 8 percent year over year.",
    "Climate scientists warn that global temperatures are on track to exceed the 1.5 degree Celsius threshold by 2035.",
    "The World Health Organization approved a new malaria vaccine showing 77 percent efficacy in clinical trials.",
    "Government officials released the annual budget report showing a modest decrease in national deficit figures.",
    "The Supreme Court ruled in a 6-3 decision on the landmark case concerning digital privacy rights.",
    "Economists predict moderate GDP growth of 2.3 percent for the upcoming fiscal year based on current trends.",
    "International trade negotiations concluded with a new agreement expected to boost exports significantly.",
    "Medical experts recommend updated COVID-19 booster shots ahead of the winter season.",
    "Tech giant Google announced layoffs affecting 12000 employees as part of restructuring efforts.",
    "The European Central Bank maintained its key interest rate amid concerns about economic slowdown.",
    "New archaeological findings in Egypt shed light on previously unknown pharaonic dynasty.",
    "A new electric vehicle battery technology promises to double current range on a single charge.",
    "Congressional leaders reached a bipartisan deal on infrastructure spending worth 1.2 trillion dollars.",
    "The International Monetary Fund revised its global growth forecast downward due to geopolitical tensions.",
    "Scientists successfully tested a quantum computing processor achieving record-breaking speeds.",
    "Public health officials confirmed the outbreak of a new flu strain and urge vaccination.",
    "India launched its first crewed lunar mission from the Satish Dhawan Space Centre in Sriharikota.",
    "The Reserve Bank of India held repo rate steady at 6.5 percent citing controlled inflation.",
    "ISRO successfully deployed the communication satellite GSAT-20 into geostationary orbit.",
    "Union Budget 2024 allocated increased funds toward education and rural development sectors.",
    "India's GDP grew at 7.2 percent in the last quarter, reaffirming its position as fastest growing economy.",
    "The Indian government launched PM Vishwakarma scheme to support traditional artisans nationwide.",
    "IIT researchers developed a low-cost water purification system suitable for rural communities.",
    "WHO designated new health emergency protocols following outbreak reports from Southeast Asia.",
    "SpaceX Starship completed its fourth test flight successfully, splashing down in the Indian Ocean.",
    "Microsoft announced integration of AI features across its Office 365 productivity suite.",
    "A study published in Nature found microplastics in 90 percent of tested ocean samples worldwide.",
    "Bengaluru metro expanded its network with three new stations opening on the Purple Line extension.",
    "The Paris Agreement signatories met to review progress on carbon emission reduction commitments.",
    "Global chip shortage begins to ease as TSMC increases semiconductor manufacturing capacity.",
    "New research links excessive social media use with increased anxiety among teenagers.",
    "Scientists discover a new species of deep sea fish near hydrothermal vents in Pacific Ocean.",
]

FAKE_NEWS = [
    "BREAKING: Government secretly puts microchips in vaccines to track citizens worldwide confirmed by insider.",
    "Scientists HIDE proof that the Earth is flat and NASA fakes all space missions with CGI technology.",
    "Miracle cure discovered: drinking bleach every morning eliminates all diseases doctors wont tell you.",
    "SHOCKING: Bill Gates admits he engineered COVID-19 to reduce world population by 90 percent.",
    "5G towers cause cancer and spread coronavirus this is what big telecom doesnt want you to know.",
    "Deep state elites drinking child blood to stay young exposed by anonymous whistleblower source.",
    "Chemtrails confirmed as mind control chemicals sprayed by government to make people obedient.",
    "Pope Francis secretly converts to Islam and announces end of Catholic Church next month.",
    "Hollywood celebrities are actually lizard people in disguise revealed by former NASA employee.",
    "Drinking urine cures cancer doctors and pharmaceutical companies suppress this natural remedy.",
    "URGENT: New world order plans to microchip all humans by 2025 resist the vaccine agenda now.",
    "Elvis Presley still alive spotted working at grocery store in Tennessee exclusive photos inside.",
    "Moon landing was completely faked Stanley Kubrick admitted it on his deathbed according to sources.",
    "Secret society controls all world governments and plans to eliminate 80 percent of population.",
    "Eating magnets gives you superpowers this is what the government does not want you to discover.",
    "EXPOSED: Dinosaurs never existed they were invented by scientists to disprove religion evidence found.",
    "WiFi radiation causes brain tumors in children tech companies pay doctors to hide the truth.",
    "Ancient aliens built the pyramids and are returning in 2024 to reclaim earth from humans.",
    "Sunscreen causes skin cancer and is a plot by pharmaceutical companies to keep you unhealthy.",
    "Obama born in Kenya new documents prove citizenship was faked insider leaks bombshell evidence.",
    "CONFIRMED: Fluoride in water is a government mind control program to make people docile and stupid.",
    "Famous actor dies in secret celebrity illuminati sacrifice mainstream media covers it up.",
    "Quantum physics proves heaven exists scientists discover parallel universe where dead people live.",
    "GMO foods contain hidden DNA altering chemicals to sterilize the population expose the agenda.",
    "Astrologer predicts massive earthquake will destroy California within 48 hours flee immediately.",
    "Natural doctors curing cancer with herbs being killed by big pharma to protect billion dollar industry.",
    "BOMBSHELL: Indian PM secretly working for foreign powers to sell national secrets leaked documents.",
    "Time travel machine invented in secret government lab photos leaked show future cities.",
    "Meditation app secretly records your thoughts using brain wave technology and sells to advertisers.",
    "Eating raw garlic every hour cures coronavirus doctors are being silenced for speaking the truth.",
    "Major Indian bank about to collapse government hiding financial crisis protect your savings now.",
    "Secret tunnel discovered under White House leading to underground city for elite survival.",
    "Chocolate milk comes from brown cows this fact has been suppressed by dairy industry for decades.",
    "Aliens have landed in Antarctica global governments covering up first contact since December.",
    "New phone update downloads your bank account information do not update your phone.",
    "Politician photographed with devil worshippers at secret ceremony exclusive footage surfaces.",
]

# ─── Build DataFrame ───────────────────────────────────────────────────────────
real_df = pd.DataFrame({'text': REAL_NEWS, 'label': 0})   # 0 = Real
fake_df = pd.DataFrame({'text': FAKE_NEWS, 'label': 1})   # 1 = Fake
df = pd.concat([real_df, fake_df], ignore_index=True).sample(frac=1, random_state=42)

# ─── Text Preprocessing ────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

X = df['clean_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# ─── Models ────────────────────────────────────────────────────────────────────
models = {
    'Logistic Regression': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
        ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))
    ]),
    'Random Forest': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
    'Naive Bayes': Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 1), max_features=5000)),
        ('clf', MultinomialNB(alpha=0.1))
    ]),
}

results = {}
print("=" * 60)
print("AI-BASED FAKE NEWS DETECTION SYSTEM - MODEL TRAINING")
print("Student: Seema | R23DB036 | Reva University")
print("=" * 60)

best_model = None
best_acc = 0
best_name = ""

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n{name}: Accuracy = {acc:.4f}")
    print(classification_report(y_test, preds, target_names=["Real", "Fake"]))
    if acc > best_acc:
        best_acc = acc
        best_model = pipeline
        best_name = name

print(f"\n✅ Best Model: {best_name} ({best_acc:.4f})")

# Save best model
os.makedirs('models', exist_ok=True)
joblib.dump(best_model, 'models/fake_news_model.pkl')
joblib.dump(results, 'models/model_results.pkl')
print("✅ Model saved to models/fake_news_model.pkl")

# Save metrics
metrics_data = {
    'best_model': best_name,
    'best_accuracy': float(best_acc),
    'all_results': {k: float(v) for k, v in results.items()},
    'dataset_size': len(df),
    'train_size': len(X_train),
    'test_size': len(X_test)
}
import json
with open('models/metrics.json', 'w') as f:
    json.dump(metrics_data, f, indent=2)
print("✅ Metrics saved to models/metrics.json")
