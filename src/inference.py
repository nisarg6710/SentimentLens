import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


MODEL_NAME = "nisarggccp0176/imdb-distilbert-sentiment"

MAX_LENGTH = 500


class SentimentInference:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, text):
        if not isinstance(text, str):
            raise TypeError("Input text must be a string.")

        if not text.strip():
            raise ValueError("Input text cannot be empty.")

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=1,
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1,
        ).item()

        confidence = probabilities[
            0, predicted_class
        ].item()

        sentiment = (
            "positive"
            if predicted_class == 1
            else "negative"
        )

        return {
            "sentiment": sentiment,
            "confidence": confidence,
            "positive_probability": probabilities[0, 1].item(),
            "negative_probability": probabilities[0, 0].item(),
        }