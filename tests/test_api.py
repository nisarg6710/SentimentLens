from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


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
        json={
            "text": ""
        },
    )

    assert response.status_code == 422



## "python -m pytest tests/test_api.py -v" run this.