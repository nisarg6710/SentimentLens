from torch.utils.data import DataLoader

from src.data.dataset import IMDBDataset


train_dataset = IMDBDataset(
    texts=train_df["text"].tolist(),
    labels=train_df["label"].tolist(),
    tokenizer=tokenizer,
    max_length=500
)

val_dataset = IMDBDataset(
    texts=val_df["text"].tolist(),
    labels=val_df["label"].tolist(),
    tokenizer=tokenizer,
    max_length=500
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)