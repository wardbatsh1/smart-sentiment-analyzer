import os
import re
import string
import pandas as pd
import nltk
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from joblib import dump
from sklearn.naive_bayes import MultinomialNB
from mlflow.tracking import MlflowClient

# ===============================
# 1. Configure MLflow with DagsHub
# ===============================
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/ward.batsh2/smart-sentiment-analyzer.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "ward.batsh2"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2ff1f1a5e81f1d485b2a5ed5f303908c613b1d70"
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
mlflow.set_experiment("sentiment-analysis")
client = MlflowClient()

# ===============================
# 2. Download NLTK resources
# ===============================
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# ===============================
# 3. Preprocessing setup
# ===============================
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
   text = re.sub(r"\b(not|no|never|cannot|isnt|wasnt|arent|werent|didnt|dont|doesnt|cant|couldnt|shouldnt|wont|wouldnt|havent|hasnt|hadnt|mustnt|mightnt)\s+(\w+)", r"\1_\2", text)
   words = [word for word in text.split() if word not in stop_words]
   words = [lemmatizer.lemmatize(word) for word in words]
   return " ".join(words)

# ===============================
# 4. Load and prepare dataset
# ===============================
df = pd.read_csv("train/train.csv", nrows=50000, names=["sentiment", "text"])
df.dropna(inplace=True)
df["text"] = df["text"].apply(preprocess_text)

def map_sentiment(value):
   if value in [1, 2]: return "Negative"
   elif value == 3: return "Neutral"
   else: return "Positive"

df["label"] = df["sentiment"].apply(map_sentiment)
X = df["text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ===============================
# 5. Vectorizer and Models
# ===============================
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
models = {
   "LogisticRegression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
   "NaiveBayes": MultinomialNB(),
   "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# ===============================
# 6. Train, Log, and Register Models
# ===============================
best_accuracy = 0
best_model_name = None
os.makedirs("models", exist_ok=True)

for model_name, model in models.items():
   print(f"\n🚀 Training {model_name}...")
   with mlflow.start_run(run_name=model_name) as run:
       run_id = run.info.run_id
       
       X_train_vec = vectorizer.fit_transform(X_train)
       X_test_vec = vectorizer.transform(X_test)

       model.fit(X_train_vec, y_train)
       y_pred = model.predict(X_test_vec)

       accuracy = accuracy_score(y_test, y_pred)
       report = classification_report(y_test, y_pred, output_dict=True)

       mlflow.log_param("model", model_name)
       mlflow.log_param("vectorizer", "Tfidf")
       mlflow.log_param("ngram_range", "(1,2)")
       mlflow.log_param("max_features", 5000)
       mlflow.log_metric("accuracy", accuracy)
       for label in report.keys():
           if label not in ["accuracy", "macro avg", "weighted avg"]:
               mlflow.log_metric(f"precision_{label}", report[label]["precision"])
               mlflow.log_metric(f"recall_{label}", report[label]["recall"])
               mlflow.log_metric(f"f1_score_{label}", report[label]["f1-score"])

       # Save locally and log with MLflow
       model_path = f"models/{model_name}_model.joblib"
       dump(model, model_path)
       mlflow.sklearn.log_model(model, artifact_path=f"{model_name}_model")

       # Register model in DagsHub registry
       model_uri = f"runs:/{run_id}/{model_name}_model"
       registered_name = f"{model_name}_Registered"
       result = mlflow.register_model(model_uri, registered_name)
       print(f"✅ Registered {registered_name} (v{result.version})")

       # Track best model
       if accuracy > best_accuracy:
           best_accuracy = accuracy
           best_model_name = model_name

       print(f"✅ {model_name} accuracy: {accuracy:.4f}")
       print(classification_report(y_test, y_pred))

print(f"\n🌟 Best model: {best_model_name} with accuracy: {best_accuracy:.4f}")
