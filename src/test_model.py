import pandas as pd
import yaml
from src.features import build_features
from src.labeling import create_labels
from src.models import train

df = pd.read_csv('data/processed/merged.csv')
cfg = yaml.safe_load(open('config/settings.yaml'))
df = build_features(df, cfg)
df = create_labels(df)
lr, rf, X_test, y_test = train(df, cfg)
print('Bitti')