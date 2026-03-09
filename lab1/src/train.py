from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from data_loader import load_train
from preprocess import preprocess
from submission_generater import generate_submission

# 读取数据
train = load_train()

# 标签
y = train["isFraud"]

# 特征
X = train.drop(columns=["isFraud"])

# 数据清理
X = preprocess(X)

# 划分数据
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练
model = LogisticRegression(max_iter=20000)

model.fit(X_train, y_train)

# 验证
pred = model.predict_proba(X_val)[:,1]

print("AUC:", roc_auc_score(y_val, pred))

print("[TRAIN DONE]")

generate_submission(model)