import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# Load Dataset
# =========================
df = pd.read_csv("dataset.csv")

print(df.head())
print(df.info())
print(df.describe())

# =========================
# EDA
# =========================
sns.pairplot(df, hue=df.columns[-1])
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.show()

# =========================
# Feature Target Split
# =========================
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# =========================
# Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# Scaling
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# Model
# =========================
model = LogisticRegression()
model.fit(X_train, y_train)

# =========================
# Prediction
# =========================
y_pred = model.predict(X_test)

# =========================
# Evaluation
# =========================
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================
# Decision Boundary (ONLY IF 2 FEATURES)
# =========================
if X.shape[1] == 2:
    X_set, y_set = scaler.inverse_transform(X_train), y_train.values
    X1, X2 = np.meshgrid(
        np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
        np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01),
    )
    plt.contourf(
        X1,
        X2,
        model.predict(
            scaler.transform(np.array([X1.ravel(), X2.ravel()]).T)
        ).reshape(X1.shape),
        alpha=0.5,
    )
    plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set)
    plt.title("Decision Boundary")
    plt.show()
