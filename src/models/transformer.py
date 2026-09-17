import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        feedforward_dim=256,
        dropout=0.1,
        padding_idx=0,
        max_length=500,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.positional_embedding = nn.Embedding(
            num_embeddings=max_length,
            embedding_dim=embedding_dim,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            embedding_dim,
            1,
        )

    def forward(self, input_ids, attention_mask):
        batch_size, sequence_length = input_ids.shape

        positions = torch.arange(
            sequence_length,
            device=input_ids.device,
        ).unsqueeze(0).expand(batch_size, sequence_length)

        token_embeddings = self.embedding(input_ids)
        position_embeddings = self.positional_embedding(positions)

        x = token_embeddings + position_embeddings

        padding_mask = attention_mask == 0

        x = self.transformer_encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        attention_mask = attention_mask.unsqueeze(-1).float()

        x = x * attention_mask

        summed = x.sum(dim=1)

        counts = attention_mask.sum(dim=1).clamp(min=1)

        pooled = summed / counts

        pooled = self.dropout(pooled)

        logits = self.fc(pooled)

        return logits.squeeze(1)