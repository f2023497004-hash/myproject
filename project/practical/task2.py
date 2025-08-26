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
# ==================== Task 2: Preprocessing ====================
df = df[['Survived', 'Sex', 'Age']].dropna()
df['Sex'] = (df['Sex'] == 'male').astype(int)

X = df[['Sex', 'Age']].values
y = df['Survived'].values

scaler = StandardScaler()
X[:, 1] = scaler.fit_transform(X[:, 1].reshape(-1, 1)).flatten()
