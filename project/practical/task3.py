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
# ==================== Task 3: Model training with cross-validation ====================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies_logreg = []
accuracies_tree = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Logistic Regression
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    preds_logreg = logreg.predict(X_test)
    accuracies_logreg.append(accuracy_score(y_test, preds_logreg))

    # Decision Tree
    tree = DecisionTreeClassifier(max_depth=3)
    tree.fit(X_train, y_train)
    preds_tree = tree.predict(X_test)
    accuracies_tree.append(accuracy_score(y_test, preds_tree))

print(f"Logistic Regression CV Accuracy: {np.mean(accuracies_logreg)*100:.2f}%")
print(f"Decision Tree CV Accuracy:      {np.mean(accuracies_tree)*100:.2f}%")

