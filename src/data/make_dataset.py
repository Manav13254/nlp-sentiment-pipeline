import pandas as pd 
from pathlib import Path

from datasets import load_dataset

def get_dataset():
    dataset = load_dataset("imdb")
    train_df = dataset['train']
    test_df = dataset['test']
    return pd.DataFrame(train_df), pd.DataFrame(test_df)

def save_dataset(df, split='train'):
    
    curr_dir = Path().resolve()
    home_dir = curr_dir
    data_path = home_dir / "data" / "raw"

    data_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path / f"{split}.csv", index=False)

if __name__ == "__main__":
    
    train_df, test_df = get_dataset()
    save_dataset(train_df, split='train')
    save_dataset(test_df, split='test')
    print("Dataset saved successfully.")
    