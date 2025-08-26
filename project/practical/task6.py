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
#  ==================== Task 6: Explainability ====================
try:
    explainer = shap.Explainer(final_model.named_steps['clf'], final_model.named_steps['prep'].transform(X_train))
    shap_values = explainer(final_model.named_steps['prep'].transform(X_test[:50]))
    shap.plots.bar(shap_values, show=False)
    plt.title("SHAP Feature Importance")
    plt.savefig("task6_shap_importance.png")
    plt.show()
except Exception as e:
    print("SHAP failed, using permutation importance instead:", e)
    perm = permutation_importance(final_model, X_test, y_test, n_repeats=10, random_state=42)
    sorted_idx = perm.importances_mean.argsort()[::-1]
    plt.bar(range(len(sorted_idx)), perm.importances_mean[sorted_idx])
    plt.xticks(range(len(sorted_idx)), np.array(numeric_features+categorical_features)[sorted_idx], rotation=45)
    plt.title("Permutation Feature Importance")
    plt.savefig("task6_perm_importance.png")
    plt.show()
