from fastapi import FastAPI, Request
from pydantic import BaseModel
from joblib import load
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


# === Load the trained model ===
model = load("models/LogisticRegression_model.joblib")

# === FastAPI app ===
app = FastAPI(
    title="Smart Sentiment Analyzer",
    description="API to predict sentiment (positive, neutral, negative)",
    version="1.0.0"
)
# Allow CORS from React frontend (localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Request format ===
class ReviewRequest(BaseModel):
    text: str

# === Response format ===
class SentimentResponse(BaseModel):
    sentiment: str

# === Prediction endpoint ===
@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(data: ReviewRequest):
    prediction = model.predict([data.text])[0]
    return {"sentiment": "Negative" if prediction == 0 else "Neutral" if prediction == 1 else "Positive"}

# === Root ===
@app.get("/")
def root():
    return {"message": "Welcome to the Sentiment Analyzer API!"}

# === For local dev only ===
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
