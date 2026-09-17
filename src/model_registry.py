from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"


MODEL_NAMES = {
    "simple_rnn": "Simple RNN",
    "lstm": "LSTM",
    "bilstm": "BiLSTM",
    "transformer": "Transformer",
    "distilbert": "DistilBERT",
}


def get_model_path(model_key, model_dir=DEFAULT_MODEL_DIR):
    """
    Return the directory associated with a model.
    """
    if model_key not in MODEL_NAMES:
        raise ValueError(
            f"Unknown model key: {model_key}"
        )

    return Path(model_dir) / model_key


def create_model_directories(model_dir=DEFAULT_MODEL_DIR):
    """
    Create directories for all registered models.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    for model_key in MODEL_NAMES:
        path = get_model_path(model_key, model_dir)
        path.mkdir(parents=True, exist_ok=True)
        paths[model_key] = path

    return paths