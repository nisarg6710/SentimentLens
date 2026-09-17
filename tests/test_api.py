from fastapi.testclient import TestClient

from src import api


client = TestClient(api.app)


class FakeSentimentModel:

    def predict(self, text):
        return {
            "sentiment": "positive",
            "confidence": 0.99,
            "positive_probability": 0.99,
            "negative_probability": 0.01,
        }


api.sentiment_model = FakeSentimentModel()


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    assert response.json()["message"] == (
        "IMDB Sentiment Classification API is running."
    )


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


def test_predict():
    response = client.post(
        "/predict",
        json={
            "text": "This movie was absolutely fantastic!"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["sentiment"] in ["positive", "negative"]

    assert 0.0 <= data["confidence"] <= 1.0

    assert 0.0 <= data["positive_probability"] <= 1.0

    assert 0.0 <= data["negative_probability"] <= 1.0


def test_empty_text():
    response = client.post(
        "/predict",
        json={"text": ""},
    )

    assert response.status_code == 422