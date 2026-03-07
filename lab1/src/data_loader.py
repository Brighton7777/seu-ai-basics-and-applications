from pathlib import Path
import pandas as pd

from preprocess import preprocess

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data"
PROCESSED_DIR = ROOT / "data/processed"

PROCESSED_DIR.mkdir(exist_ok=True)


def load_train():

    cache_file = PROCESSED_DIR / "train.parquet"

    # 如果缓存存在
    if cache_file.exists():
        print("Loading merged train data...")
        df = pd.read_parquet(cache_file)
        print(f"[LOAD DONE] shape={df.shape}")
        return df

    print("Merging raw train data...")

    train_trans = pd.read_csv(RAW_DIR / "train/train_transaction.csv")
    train_id = pd.read_csv(RAW_DIR / "train/train_identity.csv")

    train = train_trans.merge(train_id, on="TransactionID", how="left")
    print(f"[MERGE DONE] shape={train.shape}")

    train.to_parquet(cache_file)

    return train

def load_test():

    cache_file = PROCESSED_DIR / "test.parquet"

    if cache_file.exists():
        print("Loading merged test data...")
        df = pd.read_parquet(cache_file)

        print(f"[LOAD DONE] shape={df.shape}")
        return df

    print("Merging raw CSV files...")

    test_trans = pd.read_csv(RAW_DIR / "test/test_transaction.csv")
    test_id = pd.read_csv(RAW_DIR / "test/test_identity.csv")

    test = test_trans.merge(test_id, on="TransactionID", how="left")
    print(f"[MERGE DONE] shape={test.shape}")

    test.to_parquet(cache_file)

    return test