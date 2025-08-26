import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
import joblib
# ==================== Task 4: Logistic Regression from scratch ====================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic_regression(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    for epoch in range(epochs):
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)
        error = y_hat - y
        dw = np.dot(X.T, error) / m
        db = np.sum(error) / m
        w -= lr * dw
        b -= lr * db
        if epoch % 200 == 0:
            loss = -np.mean(y * np.log(y_hat+1e-10) + (1-y) * np.log(1-y_hat+1e-10))
            print(f"Epoch {epoch}: Loss = {loss:.4f}")
    return w, b

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
w, b = train_logistic_regression(X_train, y_train)
y_pred_prob = sigmoid(np.dot(X_test, w) + b)
y_pred = (y_pred_prob >= 0.5).astype(int)
acc_scratch = accuracy_score(y_test, y_pred)
print(f" Logistic Regression from Scratch Accuracy: {acc_scratch*100:.2f}%")

# Compare with built-in
model = LogisticRegression()
model.fit(X_train, y_train)
y_builtin_pred = model.predict(X_test)
acc_builtin = accuracy_score(y_test, y_builtin_pred)
print(f" Built-in Logistic Regression Accuracy: {acc_builtin*100:.2f}%")
print(" Confusion Matrix (Scratch):")
print(confusion_matrix(y_test, y_pred))

