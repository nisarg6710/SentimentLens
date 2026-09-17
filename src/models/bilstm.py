import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=1,
        dropout=0.0,
        padding_idx=0,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 2 directions → 2 * hidden_dim
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden, _) = self.lstm(packed)

        # hidden shape:
        # [num_layers * 2, batch_size, hidden_dim]

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        final_hidden = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1,
        )

        logits = self.fc(final_hidden)

        return logits.squeeze(1)