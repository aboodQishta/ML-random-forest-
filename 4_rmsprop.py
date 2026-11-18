import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("Not Confirmed 2025 clean.csv")
df = df.drop_duplicates("Order ID")
df = df[df["Call Result"].isin(["Answered", "Not Answered"])]

X = pd.get_dummies(df["Country"]).astype(float)
X["Revenue"] = df["Revenue &"].astype(str).str.replace(",", "").astype(float)
X["HasDiscount"] = df["Discount"].notna().astype(float)
y = (df["Call Result"] == "Answered").astype(int).values

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
rf.fit(Xtr, ytr)
pred = rf.predict(Xte)
print("ROWS:", len(df), "TRAIN:", len(Xtr), "TEST:", len(Xte))
print("\nRANDOM FOREST")
print("accuracy:", round(accuracy_score(yte, pred), 4), "f1:", round(f1_score(yte, pred), 4))
print(confusion_matrix(yte, pred))
print(classification_report(yte, pred, target_names=["Not Answered", "Answered"]))
print(pd.Series(rf.feature_importances_, X.columns).sort_values(ascending=False).round(4).head(5))

enc = OneHotEncoder(handle_unknown="ignore")
Ltr = enc.fit_transform(rf.apply(Xtr)).toarray()
Lte = enc.transform(rf.apply(Xte)).toarray()
Ltr = np.c_[np.ones(len(Ltr)), Ltr]
Lte = np.c_[np.ones(len(Lte)), Lte]


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss(w, A, t):
    p = np.clip(sigmoid(A @ w), 1e-9, 1 - 1e-9)
    return -np.mean(t * np.log(p) + (1 - t) * np.log(1 - p))


def train(opt, batch, lr, epochs=100):
    rng = np.random.default_rng(42)
    w = np.zeros(Ltr.shape[1])
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    step = 0
    for _ in range(epochs):
        idx = rng.permutation(len(Ltr))
        for i in range(0, len(idx), batch):
            b = idx[i:i + batch]
            g = Ltr[b].T @ (sigmoid(Ltr[b] @ w) - ytr[b]) / len(b)
            step += 1
            if opt == "rmsprop":
                v = 0.9 * v + 0.1 * g ** 2
                w -= lr * g / (np.sqrt(v) + 1e-8)
            elif opt == "adam":
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * g ** 2
                w -= lr * (m / (1 - 0.9 ** step)) / (np.sqrt(v / (1 - 0.999 ** step)) + 1e-8)
            else:
                w -= lr * g
    return w


runs = [
    ("RMSprop", "rmsprop", 32, 0.01),
]

print("\nRANDOM FOREST LEAVES + LOGISTIC LAYER, BY OPTIMIZER")
rows = []
for name, opt, batch, lr in runs:
    w = train(opt, batch, lr)
    p = (sigmoid(Lte @ w) > 0.5).astype(int)
    rows.append([name, round(loss(w, Ltr, ytr), 4), round(loss(w, Lte, yte), 4),
                 round(accuracy_score(yte, p), 4), round(f1_score(yte, p), 4)])
print(pd.DataFrame(rows, columns=["optimizer", "train_loss", "test_loss", "accuracy", "f1"]).to_string(index=False))
