# IMDB Sentiment Classification

A deep learning project for binary sentiment classification of IMDB movie reviews.

This project explores the progression from traditional recurrent neural networks to Transformer-based architectures, compares their performance on a consistent train/validation/test split, performs error analysis, and deploys a fine-tuned DistilBERT model as an inference service.

The final application combines **FastAPI, Docker, Docker Compose, and Streamlit** to provide an end-to-end sentiment classification system.

---

## Project Overview

The project has two main goals:

1. Compare different neural network architectures for sentiment classification.
2. Turn a fine-tuned pretrained Transformer into a usable ML application.

The modeling experiments include:

- Simple RNN
- LSTM
- BiLSTM
- Transformer Encoder implemented from scratch
- Fine-tuned DistilBERT

The scratch models were developed to understand the progression of sequence modeling architectures.

DistilBERT was then fine-tuned as a separate pretrained Transformer experiment and selected for the deployed application.

---

## End-to-End Architecture

```text
                         IMDB Dataset
                              |
                              v
                     Data Preprocessing
                              |
                              v
                  Train / Validation / Test
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
         Simple RNN          LSTM            BiLSTM
             |                |                |
             +----------------+----------------+
                              |
                              v
                       Model Comparison
                              |
                              v
                    Transformer Encoder
                              |
                              v
                         DistilBERT
                              |
                              v
                       Error Analysis
                              |
                              v
                      Inference Module
                              |
                              v
                           FastAPI
                              |
                              v
                    Docker / Docker Compose
                              |
                              v
                        Streamlit UI

Dataset

The project uses the IMDB movie review dataset from Stanford NLP through the Hugging Face Datasets library.

The dataset was divided into the following splits:

Split	Samples
Training	20,000
Validation	5,000
Test	25,000
Total	50,000
Split Usage
Training set: Used for model training and vocabulary construction for the scratch models.
Validation set: Used for model selection and checkpoint selection.
Test set: Kept separate for final model evaluation.

For the scratch models, the vocabulary was constructed using the training data only.

Modeling Approach
1. Simple RNN

A basic recurrent neural network was implemented as the starting point for sequence modeling.

2. LSTM

The LSTM model was introduced to better handle long-term dependencies and reduce the limitations of a basic RNN.

3. BiLSTM

A bidirectional LSTM was used to process the sequence in both forward and backward directions.

4. Transformer Encoder

A Transformer Encoder was implemented from scratch using PyTorch.

The model uses:

Token embeddings
Positional embeddings
Multi-head self-attention
Transformer encoder layers
Padding masks
Mean pooling
Binary classification head
5. DistilBERT

A pretrained distilbert-base-uncased model was fine-tuned for binary sentiment classification.

DistilBERT was treated as a separate pretrained-model experiment rather than as a direct continuation of the scratch architecture comparison.

Model Performance

All scratch models were evaluated on the same held-out IMDB test set.

Scratch Models
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Simple RNN	75.45%	76.68%	73.14%	74.87%	82.02%
LSTM	83.73%	81.91%	86.59%	84.18%	91.24%
BiLSTM	85.42%	86.41%	84.06%	85.22%	93.24%
Transformer Encoder	84.58%	84.70%	84.41%	84.55%	92.61%

Among the scratch architectures, the BiLSTM achieved the highest test accuracy, F1 score, and ROC-AUC.

Pretrained Transformer
Model	Accuracy	Precision	Recall	F1	ROC-AUC
DistilBERT	92.97%	92.16%	93.93%	93.03%	97.98%

DistilBERT was evaluated as a separate pretrained-model experiment.

Error Analysis

Error analysis was performed on the final DistilBERT predictions to understand difficult classification cases.

The analysis included:

False positive and false negative analysis
High-confidence incorrect predictions
Prediction confidence analysis
Error rates across review lengths
Brier score analysis
Qualitative inspection of difficult reviews
Overall Test Errors
Metric	Result
Test samples	25,000
Correct predictions	23,242
Incorrect predictions	1,758
Brier Score	0.0586
High-Confidence Errors
Confidence Threshold	False Positives	False Negatives	Total
≥ 90%	622	430	1,052
≥ 95%	517	328	845
≥ 99%	261	86	347

Common difficult cases included:

Mixed sentiment
Contradictory sentiment cues
Unusual or repetitive writing patterns
Local phrases conflicting with the overall review sentiment

Review length was also analyzed:

Review Length	Error Rate
≤ 500 tokens	6.54%
> 500 tokens	9.87%

The deployed DistilBERT model uses a maximum input length of 500 tokens, so longer reviews are truncated during inference.

ML Engineering

The project was structured beyond notebook-based experimentation to include reusable ML engineering components.

Configuration

Experiment and project settings are maintained in:

configs/config.yaml

This includes:

Random seed
Dataset split sizes
Maximum sequence length
Batch size
Training parameters
Model configuration
Project paths
Reproducibility

A dedicated reproducibility module provides deterministic seed configuration for:

Python
NumPy
PyTorch
CUDA
Experiment Tracking

Experiment results are recorded in:

experiments/experiments.csv
Model Registry

Model naming and model directory management are handled through:

src/model_registry.py
Error Analysis

Reusable error-analysis utilities are implemented in:

src/error_analysis.py
Inference

Model loading and prediction logic are separated from the API layer:

src/inference.py

This keeps the ML inference logic independent from the web service.

Deployment Architecture

The deployed application follows this architecture:

                +-------------------+
                |   Streamlit UI    |
                |    Port 8501      |
                +---------+---------+
                          |
                          | HTTP
                          v
                +-------------------+
                |      FastAPI      |
                |    Port 8000      |
                +---------+---------+
                          |
                          v
                +-------------------+
                |    DistilBERT     |
                |   CPU Inference   |
                +-------------------+

The FastAPI service is containerized using Docker and managed with Docker Compose.

FastAPI

The API provides the following endpoints:

Method	Endpoint	Description
GET	/	API status
GET	/health	Health check
POST	/predict	Sentiment prediction
Prediction Request
{
  "text": "This movie was absolutely fantastic! I loved every minute of it."
}
Prediction Response
{
  "sentiment": "positive",
  "confidence": 0.997,
  "positive_probability": 0.997,
  "negative_probability": 0.003
}

The API also validates incoming requests and rejects empty input.

Interactive API documentation is available through Swagger UI at:

http://127.0.0.1:8000/docs
Docker Deployment

The FastAPI inference service is packaged as a Docker image.

The Docker environment includes:

Python 3.12
PyTorch
Hugging Face Transformers
FastAPI
Uvicorn
SentencePiece
Fine-tuned DistilBERT

The deployment uses CPU-based inference because the application performs individual review predictions rather than large-scale batch inference.

Resource Configuration

The Docker container is configured with:

Resource	Limit
Memory	3 GB
CPU	2 cores
Restart policy	unless-stopped
Run with Docker Compose

Start the API:

docker compose up -d

Check the running service:

docker compose ps

Check the API health:

http://127.0.0.1:8000/health

Open the interactive API documentation:

http://127.0.0.1:8000/docs

Stop the service:

docker compose down

View container logs:

docker compose logs
Streamlit Application

The Streamlit frontend provides a simple interface for submitting movie reviews and viewing:

Predicted sentiment
Prediction confidence
Positive probability
Negative probability
API health status

Run the Streamlit application with:

python -m streamlit run app.py

The Streamlit application communicates with the FastAPI service running on port 8000.

The complete application flow is:

User
 |
 v
Streamlit
 |
 | HTTP request
 v
FastAPI
 |
 v
DistilBERT
 |
 v
Prediction
 |
 v
Streamlit
Testing

Automated API tests are implemented using Pytest.

The current test suite covers:

Root endpoint
Health endpoint
Prediction endpoint
Input validation

Run the tests with:

python -m pytest -v
Project Structure
IMDB-Sentiment-Classification/
│
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
├── README.md
├── PROJECT_DOCUMENTATION.md
│
├── configs/
│   └── config.yaml
│
├── src/
│   ├── api.py
│   ├── config.py
│   ├── error_analysis.py
│   ├── experiment_tracker.py
│   ├── inference.py
│   ├── logger.py
│   ├── model_registry.py
│   └── reproducibility.py
│
├── experiments/
│   └── experiments.csv
│
├── notebooks/
│   └── checkpoints/
│
├── tests/
│   └── test_api.py
│
├── test_notebooks/
│
└── data/

Generated files such as local datasets, logs, virtual environments, caches, and selected model artifacts are excluded from version control through .gitignore.

Technology Stack
Machine Learning
Python
PyTorch
NumPy
Pandas
Scikit-learn
NLP
Hugging Face Datasets
Hugging Face Transformers
DistilBERT
API & Application
FastAPI
Pydantic
Uvicorn
Streamlit
Deployment
Docker
Docker Compose
Development & Testing
PyYAML
Pytest
Git
Key Takeaways

This project demonstrates an end-to-end workflow for taking an NLP problem from experimentation to deployment:

Dataset
   ↓
Preprocessing
   ↓
Neural Network Experiments
   ↓
Model Evaluation
   ↓
Error Analysis
   ↓
Configuration & Reproducibility
   ↓
Inference Pipeline
   ↓
FastAPI
   ↓
Docker
   ↓
Streamlit

The project combines deep learning experimentation with practical ML engineering and deployment, rather than focusing only on model training.