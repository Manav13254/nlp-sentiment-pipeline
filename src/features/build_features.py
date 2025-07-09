import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset

# ==================== Path Setup ====================

curr_dir = Path(__file__).resolve()
project_root = curr_dir.parent.parent.parent  # from src/features
data_dir = project_root / "data" / "raw"

train_path = data_dir / "train.csv"
test_path = data_dir / "test.csv"

# ==================== Load Dataset ====================

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# ==================== Tokenization ====================

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_text(text, tokenizer, max_length=128):
    return tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )

train_encodings = tokenize_text(train_df["text"].tolist(), tokenizer)
test_encodings = tokenize_text(test_df["text"].tolist(), tokenizer)

# ==================== Custom Dataset ====================

class IMDbDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# ==================== Create DataLoaders ====================

train_dataset = IMDbDataset(train_encodings, train_df["label"].tolist())
test_dataset = IMDbDataset(test_encodings, test_df["label"].tolist())

batch_size = 16
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print("✅ DataLoaders are ready.")
