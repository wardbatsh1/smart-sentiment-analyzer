import os
import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import mlflow
import mlflow.sklearn

# === TEMPORARY NOTE ===
# Logging locally due to DagsHub MLflow outage
mlflow.set_tracking_uri("file:./mlruns")  # Logs will be saved in ./mlruns folder
mlflow.set_experiment("Sentiment-Analysis")

# === Download NLTK resources ===
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# === Preprocessing ===
stop_words = set(stopwords.words("english"))
critical_words = {
    "not", "never", "no", "cannot", "isn't", "wasn't", "aren't", "weren't",
    "didn't", "doesn't", "don't", "won't", "wouldn't", "can't", "couldn't",
    "shouldn't", "mustn't", "very", "really", "extremely", "too", "so",
    "absolutely", "highly", "but", "however", "recommend", "avoid", "love",
    "hate", "like", "dislike", "good", "bad", "great", "poor"
}
stop_words -= critical_words
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(
        r"\b(not|no|never|cannot|isnt|wasnt|arent|werent|didnt|dont|doesnt|cant|couldnt|shouldnt|wont|wouldnt|havent|hasnt|hadnt|mustnt|mightnt)\s+(\w+)",
        r"\1_\2",
        text,
    )
    words = [word for word in text.split() if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)

# === Load Dataset ===
data = pd.read_csv("train/train.csv", nrows=100000)
data.columns = ["sentiment", "text"]
data.dropna(subset=["text"], inplace=True)
data["text"] = data["text"].apply(preprocess_text)

def map_sentiment(value):
    if value in [1, 2]:
        return "Negative"
    elif value == 3:
        return "Neutral"
    else:
        return "Positive"

data["sentiment_category"] = data["sentiment"].apply(map_sentiment)

# === Encode Labels ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data["sentiment_category"])
X = data["text"]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Define Models ===
models = {
    "LogisticRegression": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ]),
    "NaiveBayes": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", MultinomialNB())
    ]),
    "RandomForest": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ]),
}

# === Train & Log ===
for model_name, model_pipeline in models.items():
    print(f"\nTraining {model_name}...")

    with mlflow.start_run(run_name=model_name):
        model_pipeline.fit(X_train, y_train)

        y_pred = model_pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_param("max_features", 5000)
        mlflow.log_metric("accuracy", acc)

        model_path = f"models/{model_name}_model.joblib"
        dump(model_pipeline, model_path)
        mlflow.sklearn.log_model(model_pipeline, model_name)

        print(f"Model saved to: {model_path}")
