FROM python:3.12-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY src ./src
COPY configs ./configs
COPY notebooks/checkpoints/pretrained_transformer/distilbert_best \
     ./notebooks/checkpoints/pretrained_transformer/distilbert_best

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]


# docker build -t imdb-sentiment-api .

# docker run -d --name imdb-sentiment-api -p 8000:8000 --memory=3g --cpus=2 imdb-sentiment-api

# docker build --no-cache -t imdb-sentiment-api:test .
## docker command to check whether ignored files are excluded or not.

# docker rmi imdb-sentiment-api:test, docker builder prune -af
## removing this test since it keeps occupying extra space

# docker restart imdb-sentiment-api
## container restart test