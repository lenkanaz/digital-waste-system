import pandas as pd
import yaml
import numpy as np
from src.features import build_features
from src.labeling import create_labels, baseline_predict
from src.models import train, FEATURES
from src.evaluation import evaluate_all
from src.simulation import simulate_users

print("Veri yükleniyor...")
df = pd.read_csv('data/processed/merged.csv')
cfg = yaml.safe_load(open('config/settings.yaml'))

print("Feature engineering...")
df = build_features(df, cfg)

print("Labeling...")
df = create_labels(df)

print("Model eğitiliyor...")
lr, rf, X_test, y_test = train(df, cfg)

print("Evaluation...")
baseline = baseline_predict(
    pd.DataFrame({'days_since_access': X_test['days_since_access'].values})
)

models = {
    'Baseline': baseline,
    'Logistic Regression': lr,
    'Random Forest': rf
}

evaluate_all(models, X_test, y_test)

print("\nSimülasyon çalışıyor...")
sim_results = simulate_users(rf, cfg)
print(sim_results.to_string())
sim_results.to_csv('outputs/reports/simulation_results.csv', index=False)

print("\nTamamlandı!")