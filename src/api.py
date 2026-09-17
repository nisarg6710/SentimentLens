import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference import SentimentInference
from src.logger import logger


app = FastAPI(
    title="IMDB Sentiment Classification API",
    description="Sentiment classification using a fine-tuned DistilBERT model.",
    version="1.0.0",
)


sentiment_model = None

def get_sentiment_model():
    global sentiment_model

    if sentiment_model is None:
        sentiment_model = SentimentInference()

    return sentiment_model


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Movie review text to classify.",
    )


@app.get("/")
def root():
    return {
        "message": "IMDB Sentiment Classification API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(request: PredictionRequest):

    logger.info("Prediction request received")

    start_time = time.perf_counter()

    try:
        model = get_sentiment_model()

        result = model.predict(request.text)

        elapsed_time = time.perf_counter() - start_time

        logger.info(
            "Prediction: %s | Confidence: %.4f | Latency: %.4fs",
            result["sentiment"],
            result["confidence"],
            elapsed_time,
        )

        return result

    except Exception:
        elapsed_time = time.perf_counter() - start_time

        logger.exception(
            "Prediction failed | Latency: %.4fs",
            elapsed_time,
        )

        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the prediction.",
        )