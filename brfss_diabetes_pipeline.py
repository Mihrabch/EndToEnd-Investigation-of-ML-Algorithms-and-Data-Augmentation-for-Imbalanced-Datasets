"""
BRFSS Diabetes Prediction — ML Pipeline
========================================
Dataset : CDC Behavioral Risk Factor Surveillance System (BRFSS)
Target  : DIABETE4 (diabetes diagnosis)
Purpose : End-to-end pipeline showcasing preprocessing, class balancing,
          encoding, and hyperparameter tuning via GridSearchCV.

NOTE: This script is a pipeline demonstration sample using SMOTEN
      sampling_strategy=1.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTEN
from xgboost import XGBClassifier


# ── 1. Load & Preprocess ──────────────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    """
    Load BRFSS data and apply domain-driven filters:
      - Remove age groups 1–4 (under 35s, low T2D prevalence)
      - Drop _RFHYPE6 (redundant hypertension flag)
      - Cast all columns to category (required by SMOTEN)
    """
    df = pd.read_csv(path)
    df = df[~df['_AGEG5YR'].isin([1, 2, 3, 4])]
    df = df.drop(columns=['_RFHYPE6'])
    return df.astype('category')


# ── 2. Split ──────────────────────────────────────────────────────────────────
def split(df: pd.DataFrame, target: str = 'DIABETE4'):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=4)


# ── 3. Class Balancing — SMOTEN ───────────────────────────────────────────────
def resample(X_train, y_train, strategy: float = 1.0):
    """
    SMOTEN generates synthetic samples for categorical features.
    strategy=1.0 fully balances minority/majority classes.
    Other strategies (0.3, 0.5) explored in the full notebook.
    """
    sm = SMOTEN(sampling_strategy=strategy, random_state=42)
    return sm.fit_resample(X_train, y_train)


# ── 4. One-Hot Encoding ───────────────────────────────────────────────────────
def encode(X_train, X_test):
    """
    Fit encoder on resampled train set only — prevents data leakage.
    handle_unknown='ignore' ensures unseen test categories don't break inference.
    """
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(X_train)
    cols = ohe.get_feature_names_out(X_train.columns)
    return (
        pd.DataFrame(ohe.transform(X_train), columns=cols),
        pd.DataFrame(ohe.transform(X_test),  columns=cols)
    )


# ── 5. Label Fix ──────────────────────────────────────────────────────────────
def fix_labels(*series):
    """ Remap BRFSS label 3 (gestational/borderline) → 0 (no diabetes). """
    return [s.replace({3: 0}) for s in series]


# ── 6. Evaluation ─────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred) -> dict:
    return {
        'precision' : precision_score(y_true, y_pred, average='macro'),
        'recall'    : recall_score(y_true, y_pred, average='macro'),
        'accuracy'  : accuracy_score(y_true, y_pred),
        'f1'        : f1_score(y_true, y_pred, average='macro'),
        'auc'       : roc_auc_score(y_true, y_pred, average=None),
        'report'    : classification_report(y_true, y_pred)
    }


# ── 7. Grid Search Configs ────────────────────────────────────────────────────
# Scoring = recall — prioritising sensitivity (catching diabetic cases)
# over precision, given the clinical cost of false negatives.

GRIDS = {
    'Logistic Regression': (
        LogisticRegression(random_state=42),
        {
            'penalty'      : ['l1', 'l2', 'elasticnet', 'none'],
            'C'            : [0.1, 0.01, 0.001, 0.5],
            'solver'       : ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter'     : [100, 300, 500],
            'class_weight' : [{0: 1, 1: 10}],
            'fit_intercept': [True, False],
            'warm_start'   : [True, False]
        }
    ),
    'AdaBoost': (
        AdaBoostClassifier(estimator=LogisticRegression()),
        {
            'n_estimators' : [50, 100, 200, 1000],
            'learning_rate': [0.1, 0.5, 1.0],
            'algorithm'    : ['SAMME', 'SAMME.R']
        }
    ),
    'Decision Tree': (
        DecisionTreeClassifier(),
        {
            'criterion'        : ['gini', 'entropy'],
            'splitter'         : ['best', 'random'],
            'max_depth'        : [None, 2, 5, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf' : [1, 2, 5]
        }
    ),
    'XGBoost': (
        XGBClassifier(),
        {
            'n_estimators'    : np.round(np.linspace(50, 150, 25)).astype(int),
            'learning_rate'   : np.linspace(0.1, 1, 2),
            'max_depth'       : np.round(np.linspace(2, 12, 2)).astype(int),
            'gamma'           : np.linspace(0.1, 1, 2),
            'subsample'       : np.linspace(0.1, 1, 2),
            'colsample_bytree': np.linspace(0.1, 1, 2),
            'reg_alpha'       : np.linspace(0.1, 1, 2),
            'reg_lambda'      : np.linspace(0.1, 1, 2),
        }
    ),
    'Gradient Boosting': (
        GradientBoostingClassifier(random_state=42),
        {
            'n_estimators' : np.round(np.linspace(100, 1000, 10)).astype(int),
            'learning_rate': np.linspace(0.1, 1, 100),
            'loss'         : ['log_loss', 'exponential'],
            'max_features' : ['sqrt', 'log2', None]
        }
    ),
}


def run_grid_search(X_train, y_train):
    results = {}
    for name, (model, params) in GRIDS.items():
        print(f"\n── {name} ──")
        gs = GridSearchCV(model, params, cv=5, n_jobs=-1, scoring='recall')
        gs.fit(X_train, np.ravel(y_train))
        results[name] = {'best_params': gs.best_params_, 'best_recall': gs.best_score_}
        print(f"  Best params : {gs.best_params_}")
        print(f"  Best recall : {gs.best_score_:.4f}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    data                             = load_and_clean('Final_diabetes_data.csv')
    X_train, X_test, y_train, y_test = split(data)
    X_train, y_train                 = resample(X_train, y_train, strategy=1.0)
    X_train, X_test                  = encode(X_train, X_test)
    y_train, y_test                  = fix_labels(y_train, y_test)

    results = run_grid_search(X_train, y_train)
