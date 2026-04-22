import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    'norm_age', 'norm_size', 'access_ratio',
    'days_since_access', 'co2_estimate_g'
]

def train(df: pd.DataFrame, cfg: dict):
    X = df[FEATURES].fillna(0)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg['model']['test_size'],
        random_state=cfg['data']['random_seed'],
        stratify=y
    )

    # Logistic Regression
    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=cfg['data']['random_seed']))
    ])
    lr.fit(X_train, y_train)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=50,
        random_state=cfg['data']['random_seed']
    )
    rf.fit(X_train, y_train)

    # Kaydet
    joblib.dump(lr, 'outputs/models/lr_pipeline.pkl')
    joblib.dump(rf, 'outputs/models/random_forest.pkl')
    print("Modeller kaydedildi.")

    return lr, rf, X_test, y_test