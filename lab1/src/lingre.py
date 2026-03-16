import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
from pathlib import Path
import gc

ROOT = Path(__file__).resolve().parent.parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    return load_train_data(), load_test_data()

def load_train_data():
    train_id_data = pd.read_csv(ROOT / "data/train/train_identity.csv")
    train_trans_data = pd.read_csv(ROOT / "data/train/train_transaction.csv")
    # test_id_data = pd.read_csv(ROOT / "data/test/test_identity.csv")
    # test_trans_data = pd.read_csv(ROOT / "data/test/test_transaction.csv")
    # print(train_id_data.shape)
    # print(train_trans_data.shape)
    train_data = pd.merge(train_trans_data, train_id_data, on='TransactionID', how='left')
    # test_data = pd.merge(test_trans_data, test_id_data, on='TransactionID', how='left')
    # print(train_data.shape)
    # print(test_data.shape)

    return train_data

def load_test_data():
    test_id_data = pd.read_csv(ROOT / "data/test/test_identity.csv")
    test_trans_data = pd.read_csv(ROOT / "data/test/test_transaction.csv")
    test_data = pd.merge(test_trans_data, test_id_data, on='TransactionID', how = 'left')
    return test_data

def to_tensor(X: pd.DataFrame) -> torch.Tensor:
    X_np = X.to_numpy(dtype=np.float32, copy=False)
    del X
    gc.collect()
    X_tensor = torch.from_numpy(X_np)
    return X_tensor

def data_preprocess(train_data: pd.DataFrame, test_data: pd.DataFrame): 
    train_data = train_data.iloc[:61200, :]
    train_labels = train_data['isFraud'] 
    train_data = train_data.drop(columns=['isFraud'])
    test_ids = test_data.iloc[:, 0]

    n_train = train_data.shape[0]

    all_features = pd.concat((train_data.iloc[:, 1:], test_data.iloc[:, 1:]))

    del train_data, test_data
    gc.collect()

    numeric_incides = all_features.select_dtypes(include='number').columns    
    # OOM's cause !!!
    # all_features[numeric_incides] = all_features[numeric_incides].apply(
    #     lambda x : (x - x.mean()) / x.std()
    # )
    num = all_features[numeric_incides]
    mean = num.mean()
    std = num.std()
    all_features[numeric_incides] = (num - mean) / std

    all_features[numeric_incides] = all_features[numeric_incides].fillna(0)

    all_features = pd.get_dummies(all_features, dummy_na=True)

    return all_features.iloc[:n_train, :], train_labels, all_features.iloc[n_train:, :], test_ids

# def test_data_preprocess(test_data: pd.DataFrame):
#     test_ids = test_data.iloc[:, 0]
#     test_features = test_data.iloc[:, 1:]
#     del test_data
#     gc.collect()
    
#     numeric_features_indices = test_features.dtypes[test_features.dtypes != 'object'].index
#     num = test_features[numeric_features_indices]
#     test_features[numeric_features_indices] = (num - num.mean()) / num.std()
#     del num
#     gc.collect()
#     test_features[numeric_features_indices] = test_features[numeric_features_indices].fillna(0)
#     del numeric_features_indices
#     gc.collect()
#     test_features = pd.get_dummies(test_features, dummy_na=True)
#     test_features = test_features.sort_index(axis=1)
#     gc.collect()

#     return test_features, test_ids


def weight_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
def Net(input_num, hidden1_num, hidden2_num, output_num, dropout1, dropout2) -> nn.Module:
    net = nn.Sequential(
        nn.Linear(input_num, hidden1_num),
        nn.ReLU(),
        nn.Dropout(dropout1),
        nn.Linear(hidden1_num, hidden2_num),
        nn.ReLU(),
        nn.Dropout(dropout2),
        nn.Linear(hidden2_num, output_num)
    )
    net.apply(weight_init)
    return net

def train(net, epochs_num, train_loader, loss, trainer):
    for epoch in range(epochs_num):
        total_loss = 0.0
        examples_num = 0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat.reshape(-1, 1), y.reshape(-1, 1))
            total_loss += l.detach() * len(X)
            examples_num += len(X)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        print(f'epoch {epoch + 1}, loss {total_loss / examples_num:.6f}')

def predicte(net, test_features):
    with torch.no_grad():
        test_features = test_features.to(device)
        y_hat = net(test_features)
        y_hat = y_hat.reshape(-1, 1)
        return y_hat 

def save_prediction(test_ids: torch.Tensor, test_labels: torch.Tensor):
    test_ids = test_ids.type(torch.int).numpy().reshape(-1)
    test_labels = test_labels.numpy().reshape(-1)
    submission = pd.DataFrame({
        'TransactionID': test_ids,
        'isFraud': test_labels
    })
    df = os.path.join(ROOT, "submission.csv")
    submission.to_csv(df, index=False, float_format="%.10f")

    print('prediction saved to submission.csv')

if __name__ == '__main__':
    
    train_data, test_data = load_data()
    train_features, train_labels, test_features, test_ids = data_preprocess(train_data, test_data)
    del train_data, test_data
    gc.collect()
    # assert(False)

    train_features = to_tensor(train_features)
    # print(train_features.shape)
    train_labels = to_tensor(train_labels)
    # test_features[:] = to_tensor(test_featrues)

    print(train_features.shape[-1])

    input_num, hidden1_num, hidden2_num, output_num = train_features.shape[-1], 16, 8, 1
    dropout1, dropout2 = 0.5, 0.5
    net = Net(input_num, hidden1_num, hidden2_num, output_num, dropout1, dropout2)
    net = net.to(device)

    loss = nn.MSELoss()
    lr = 0.00005
    trainer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

    batch_size = 128
    train_loader = DataLoader(TensorDataset(train_features, train_labels), batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    epochs_num = 200
    train(net, epochs_num, train_loader, loss, trainer)

    del train_features
    del train_labels
    gc.collect()

    test_features = to_tensor(test_features)
    test_ids = to_tensor(test_ids)

    print(test_features.shape)
    test_loader = DataLoader(test_features, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    y_hats = []
    for X in test_loader:
        y_hat = predicte(net, X)
        y_hats.append(y_hat.cpu().numpy())
    
    test_labels = np.concatenate(y_hats, axis=0)

    del y_hats
    gc.collect()

    test_labels = torch.from_numpy(test_labels).reshape(-1, 1)

    save_prediction(test_ids, test_labels)
