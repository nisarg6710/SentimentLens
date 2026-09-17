import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SimpleRNNClassifier(nn.Module): ##nn.Module -> base class for all neural n/w modules
    def __init__(
            self,
            vocab_size,
            embedding_dim=128,
            hidden_dim=128,
            num_layers=1,
            dropout=0.0,
    ):
        super().__init__()  ## purpose --> call the initialization method of the parent class (nn.Module).

        self.embedding = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim = embedding_dim,
            padding_idx=0 ##ensure that padding tokens are ignored and do not learn gradients
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.fc = nn.Linear( ##fully connected (linear) layer. it maps the final hidden state of rnn (hidden_dim) down to a single output value (1).
            hidden_dim,
            1
        )

    ## forward function defines how input data passws through layers to generate a prediction.
    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids) ##list of indices to word embedding(dense word vectors.)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _ , hidden = self.rnn(packed)
        last_hidden = hidden[-1]
        logits = self.fc(last_hidden)
        return logits.squeeze(1)