import pandas as pd
import yaml
import joblib
from src.features import build_features
from src.labeling import create_labels, baseline_predict
from src.models import train, FEATURES
from src.evaluation import evaluate_all, plot_feature_importance, failure_analysis
from src.simulation import simulate_users
import os

def main():
    print("=== Digital Waste System ===\n")

    cfg = yaml.safe_load(open('config/settings.yaml'))
    df = pd.read_csv('data/processed/merged.csv')
    print(f"    {len(df)} satır yüklendi.")

    df = build_features(df, cfg)
    df = create_labels(df)
    print(f"    Waste: {df['label'].sum()} / {len(df)}")

    if os.path.exists('outputs/models/random_forest.pkl'):
        print("Kaydedilmiş model yükleniyor...")
        rf = joblib.load('outputs/models/random_forest.pkl')
        lr = joblib.load('outputs/models/lr_pipeline.pkl')
        from sklearn.model_selection import train_test_split
        X = df[FEATURES].fillna(0)
        y = df['label']
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=cfg['model']['test_size'],
            random_state=cfg['data']['random_seed'], stratify=y
        )
    else:
        print("Model eğitiliyor...")
        lr, rf, X_test, y_test = train(df, cfg)

    baseline = baseline_predict(
        pd.DataFrame({'days_since_access': X_test['days_since_access'].values})
    )
    models = {
        'Baseline': baseline,
        'Logistic Regression': lr,
        'Random Forest': rf
    }
    evaluate_all(models, X_test, y_test)
    plot_feature_importance(rf, FEATURES)

    print("\nFailure analysis çalışıyor...")
    fp, fn = failure_analysis(rf, X_test, y_test, FEATURES)

    # Gerçek local dosyalar için reclaimable hesapla
    local_mask = df['source'] == 'local'
    if local_mask.sum() > 0:
        local_df_real = df[local_mask].copy()
        local_features = local_df_real[FEATURES].fillna(0)
        local_proba = rf.predict_proba(local_features)[:, 1]
        local_waste = local_df_real[local_proba > 0.5]
        reclaimable_mb = local_waste['size_mb'].sum()
        print(f"\n=== GERÇEK LOCAL DOSYALAR ===")
        print(f"Taranan: {local_mask.sum()} dosya")
        print(f"Waste flagged: {len(local_waste)} dosya")
        print(f"Reclaimable: {reclaimable_mb:.1f} MB ({reclaimable_mb/1000:.2f} GB)")

    sim = simulate_users(rf, cfg)
    print("\nSimülasyon sonuçları:")
    print(sim.to_string())
    sim.to_csv('outputs/reports/simulation_results.csv', index=False)

    print("\n=== Tamamlandı ===")

if __name__ == "__main__":
    main()