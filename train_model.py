# train_model.py
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from utils import clean_text

DATA_PATH = os.path.join("dataset", "news.csv")
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECT_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# If dataset missing, create small sample so training still runs
if not os.path.exists(DATA_PATH):
    print("⚠️ dataset/news.csv not found — creating a small sample dataset for demo.")
    sample = [
        ("Government announces new education policy for students", "REAL"),
        ("Aliens found in Delhi market according to reports", "FAKE"),
        ("Scientists discover new treatment for cancer", "REAL"),
        ("Actor claims to time travel using mirror", "FAKE"),
        ("AI used in hospitals to improve patient care", "REAL"),
        ("Facebook shutting down permanently next month", "FAKE"),
    ]
    df_sample = pd.DataFrame(sample, columns=["text","label"])
    df_sample.to_csv(DATA_PATH, index=False)
    print(f"Sample dataset created at {DATA_PATH} ({len(df_sample)} rows)")

# Load dataset
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text","label"]).reset_index(drop=True)
print(f"✅ Loaded {len(df)} records.")

# Preprocess
print("🔄 Cleaning text (may take a moment)...")
df['clean_text'] = df['text'].apply(clean_text)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42, stratify=df['label'] if 'label' in df else None
)

# Vectorize
print("📐 Fitting TF-IDF vectorizer...")
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Model training
print("⚙️ Training Logistic Regression model...")
model = LogisticRegression(max_iter=300)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Training complete — Accuracy on test set: {acc*100:.2f}%")
print("\nClassification report:\n", classification_report(y_test, y_pred))

# Save model & vectorizer
joblib.dump(model, MODEL_PATH)
joblib.dump(tfidf, VECT_PATH)
print(f"🎯 Model saved to: {MODEL_PATH}")
print(f"🎯 Vectorizer saved to: {VECT_PATH}")
