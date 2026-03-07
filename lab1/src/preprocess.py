import pandas as pd

def preprocess(df):
    # 替换字符，保证一致
    df.columns = df.columns.str.replace('_', '-', regex=False)

    # 删除ID
    if "TransactionID" in df.columns:
        df = df.drop(columns=["TransactionID"])

    # 填充缺失值
    df = df.fillna(-999)

    # 类别变量编码
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("category").cat.codes

    return df