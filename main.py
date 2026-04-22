import pandas as pd
import yaml
import joblib
from src.features import build_features
from src.labeling import create_labels, baseline_predict
from src.models import train, FEATURES
from src.evaluation import evaluate_all
from src.simulation import simulate_users

def main():
    print("=== Digital Waste System ===\n")

    print("1/6 Config yükleniyor...")
    cfg = yaml.safe_load(open('config/settings.yaml'))

    print("2/6 Veri yükleniyor...")
    df = pd.read_csv('data/processed/merged.csv')
    print(f"    {len(df)} satır yüklendi.")

    print("3/6 Feature engineering...")
    df = build_features(df, cfg)

    print("4/6 Labeling...")
    df = create_labels(df)
    print(f"    Waste: {df['label'].sum()} / {len(df)}")

    print("5/6 Model eğitiliyor...")
    lr, rf, X_test, y_test = train(df, cfg)

    print("6/6 Evaluation + Simülasyon...")
    baseline = baseline_predict(
        pd.DataFrame({'days_since_access': X_test['days_since_access'].values})
    )
    models = {
        'Baseline': baseline,
        'Logistic Regression': lr,
        'Random Forest': rf
    }
    evaluate_all(models, X_test, y_test)

    sim = simulate_users(rf, cfg)
    print("\nSimülasyon sonuçları:")
    print(sim.to_string())
    sim.to_csv('outputs/reports/simulation_results.csv', index=False)

    print("\n=== Tamamlandı ===")
    print("Grafikler: outputs/figures/")
    print("Raporlar:  outputs/reports/")
    print("Modeller:  outputs/models/")

if __name__ == "__main__":
    main()