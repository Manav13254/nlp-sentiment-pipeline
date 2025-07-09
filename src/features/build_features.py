import pandas as pd
from pathlib import Path
import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import yaml

# ------------------------------
# Load params.yaml once globally
# ------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]
PARAMS_PATH = BASE_DIR / "params.yaml"

with open(PARAMS_PATH, "r") as file:
    params = yaml.safe_load(file)

BATCH_SIZE = params['preprocessing']['batch_size']
MAX_LENGTH = params['preprocessing']['max_length']


# ------------------------------
# Dataset Class
# ------------------------------
class IMDbDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels) if self.labels is not None else len(self.encodings["input_ids"])


# ------------------------------
# Tokenize Texts
# ------------------------------
def tokenize_texts(texts, tokenizer, max_length):
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
    )


# ------------------------------
# Main DataLoader function
# ------------------------------
def load_dataloaders(batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    train_path = BASE_DIR / "data" / "raw" / "train.csv"
    test_path = BASE_DIR / "data" / "raw" / "test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    train_encodings = tokenize_texts(train_df['text'].tolist(), tokenizer, max_length)
    test_encodings = tokenize_texts(test_df['text'].tolist(), tokenizer, max_length)

    train_dataset = IMDbDataset(train_encodings, train_df["label"].tolist())
    test_dataset = IMDbDataset(test_encodings, test_df["label"].tolist())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ------------------------------
# Debug Run
# ------------------------------
if __name__ == "__main__":
    train_loader, test_loader = load_dataloaders()
    print(" Dataloaders created using config from params.yaml")
