from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
import mlflow
import mlflow.sklearn
import os
import logging
import sys
from joblib import load

# === Set up logging (so Koyeb captures it) ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)  # ✅ Send logs to stdout for Koyeb
    ]
)
logger = logging.getLogger(__name__)

# === Load the saved TfidfVectorizer ===
vectorizer = load("models/tfidf_vectorizer.joblib")

# === Configure MLflow ===
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/ward.batsh2/smart-sentiment-analyzer.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "ward.batsh2"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2ff1f1a5e81f1d485b2a5ed5f303908c613b1d70"
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# === Load the model from MLflow registry ===
logger.info("Loading model from MLflow model registry...")
try:
    model = mlflow.sklearn.load_model("models:/LogisticRegression_Registered/1")
    logger.info("✅ Model loaded successfully.")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    raise

# === FastAPI app ===
app = FastAPI(
    title="Smart Sentiment Analyzer",
    description="API to predict sentiment (Positive, Neutral, Negative)",
    version="1.0.0"
)

# === Enable Prometheus monitoring ===
Instrumentator().instrument(app).expose(app)

# === Enable CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Request/Response Models ===
class ReviewRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str

# === Root Endpoint ===
@app.get("/")
def root():
    logger.info("📡 Root endpoint accessed.")
    return {"message": "Welcome to the Sentiment Analyzer API!"}

# === Prediction Endpoint ===
@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(data: ReviewRequest):
    logger.info(f"📩 Received input: {data.text}")
    try:
        transformed_text = vectorizer.transform([data.text])
        prediction = model.predict(transformed_text)[0]
        sentiment = (
            "Negative" if prediction == 0 else
            "Neutral" if prediction == 1 else
            "Positive"
        )
        logger.info(f"✅ Predicted sentiment: {sentiment}")
        return {"sentiment": sentiment}
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return {"sentiment": "Prediction error"}

# === For local development only ===
if __name__ == "__main__":
    logger.info("🚀 Starting FastAPI app locally...")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
