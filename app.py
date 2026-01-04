# app.py
from flask import Flask, render_template, request, jsonify
import os, joblib
from utils import clean_text

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "fake_news_model.pkl")
VECT_PATH = os.path.join("model", "tfidf_vectorizer.pkl")

# If model missing, run training (this imports and executes train_model.py)
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    print("⚙️ Model or vectorizer not found — training a new model now...")
    import train_model  # careful: train_model.py runs training and saves files

# Load model & vectorizer
model = joblib.load(MODEL_PATH)
tfidf = joblib.load(VECT_PATH)
print("✅ Model and vectorizer loaded.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form.get("text", "")
    if not text:
        return jsonify({"success": False, "message": "No text provided"}), 400
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    pred = model.predict(vector)[0]  # "REAL" or "FAKE"
    return jsonify({"success": True, "result": pred})

if __name__ == "__main__":
    app.run(debug=True)
