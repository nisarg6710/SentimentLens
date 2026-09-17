import torch
from pathlib import Path

class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        checkpoint_path,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.checkpoint_path = Path(
            checkpoint_path
        )

    def _forward(self, batch):
        """
        Handles the model-specific inputs contained in a batch.

        Current batch format:
            (input_ids, labels, lengths)

        Models:
            RNN    -> input_ids, lengths
            LSTM   -> input_ids, lengths
            BiLSTM -> input_ids, lengths
            Transformer -> input_ids, attention_mask (later)
        """

        input_ids = batch[0]
        labels = batch[1]

        # Everything after labels is considered
        # additional model-specific information.
        extra_inputs = batch[2:]

        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)

        extra_inputs = tuple(
            value.to(self.device)
            if torch.is_tensor(value)
            else value
            for value in extra_inputs
        )

        logits = self.model(
            input_ids,
            *extra_inputs,
        )

        return logits, labels 

    def train_epoch(self):

        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in self.train_loader:

            logits, labels = self._forward(batch)

            self.optimizer.zero_grad()

            loss = self.criterion(
                logits,
                labels,
            )

            loss.backward()

            self.optimizer.step()

            total_loss += (
                loss.item() * labels.size(0)
            )

            predictions = (
                torch.sigmoid(logits) >= 0.5
            )

            total_correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            total_samples += labels.size(0)

        epoch_loss = (
            total_loss / total_samples
        )

        epoch_accuracy = (
            total_correct / total_samples
        )

        return epoch_loss, epoch_accuracy

    @torch.no_grad()
    def validate(self):

        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in self.val_loader:

            logits, labels = self._forward(batch)

            loss = self.criterion(
                logits,
                labels,
            )

            total_loss += (
                loss.item() * labels.size(0)
            )

            predictions = (
                torch.sigmoid(logits) >= 0.5
            )

            total_correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            total_samples += labels.size(0)

        epoch_loss = (
            total_loss / total_samples
        )

        epoch_accuracy = (
            total_correct / total_samples
        )

        return epoch_loss, epoch_accuracy

    @torch.no_grad()
    def predict(self, data_loader):

        self.model.eval()

        all_labels = []
        all_predictions = []
        all_probabilities = []

        for batch in data_loader:

            logits, labels = self._forward(batch)

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= 0.5
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

        return (
            all_labels,
            all_predictions,
            all_probabilities,
        )

    def fit(self, epochs):

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        best_val_loss = float("inf")

        # checkpoint_path = Path(
        #     "checkpoints/rnn/best_model.pt"   ##.pt(pytorch) or .pth(PyTorch Hub or PyTorch Header) also .bin means binary
        # )

        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        for epoch in range(epochs):

            train_loss, train_accuracy = (
                self.train_epoch()
            )

            val_loss, val_accuracy = (
                self.validate()
            )

            history["train_loss"].append(
                train_loss
            )

            history["train_accuracy"].append(
                train_accuracy
            )

            history["val_loss"].append(
                val_loss
            )

            history["val_accuracy"].append(
                val_accuracy
            )

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                torch.save(
                    self.model.state_dict(),
                    self.checkpoint_path
                )

                print("  -> Best model saved.")

        

            print(
                f"Epoch {epoch + 1:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.4f}"
            )

        return history
