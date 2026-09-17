from pathlib import Path
import csv
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"


class ExperimentTracker:
    def __init__(self, output_dir=EXPERIMENTS_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / "experiments.csv"

    def log_experiment(
        self,
        model_name,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        brier_score=None,
        parameters=None,
        notes=None,
    ):
        experiment = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "brier_score": brier_score,
            "parameters": parameters or "",
            "notes": notes or "",
        }

        file_exists = self.csv_path.exists()

        with open(
            self.csv_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=experiment.keys(),
            )

            if not file_exists:
                writer.writeheader()

            writer.writerow(experiment)

        return experiment