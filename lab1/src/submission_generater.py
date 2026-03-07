import pandas as pd
from pathlib import Path

from data_loader import load_test
from preprocess import preprocess

ROOT = Path(__file__).resolve().parent.parent


def generate_submission(model):

    print("[LOAD TEST DATA]")
    test = load_test()
    test_ids = test["TransactionID"]
    X_test = test.drop(columns=["TransactionID"])
    X_test = preprocess(X_test)

    print("Predicting test data...")
    pred = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({
        "TransactionID": test_ids,
        "isFraud": pred
    })

    save_path = ROOT / "submission.csv"

    submission.to_csv(save_path, index=False)

    print(f"[DONE] submission saved to {save_path}")