import pandas as pd
import numpy as np
from src.features import build_features
from src.labeling import baseline_predict

FEATURES = [
    'norm_age', 'norm_size', 'access_ratio',
    'days_since_access', 'co2_estimate_g'
]

def simulate_users(model, cfg, n=1000):
    scenarios = {
        'messy_hoarder': {
            'age_scale': 500, 'access_scale': 300, 'size_mean': 16
        },
        'active_user': {
            'age_scale': 60, 'access_scale': 10, 'size_mean': 13
        },
        'heavy_media': {
            'age_scale': 200, 'access_scale': 180, 'size_mean': 17
        }
    }

    summary = []
    for user_type, params in scenarios.items():
        rng = np.random.default_rng(cfg['data']['random_seed'])
        df_sim = pd.DataFrame({
            'file_id': [f'{user_type}_{i}' for i in range(n)],
            'extension': rng.choice(
                ['.jpg','.mp4','.zip','.pdf','.py'],
                p=[.25,.20,.20,.20,.15], size=n
            ),
            'size_bytes': rng.lognormal(params['size_mean'], 2.0, n).astype(int),
            'days_since_access': rng.exponential(params['access_scale'], n),
            'age_days': rng.exponential(params['age_scale'], n),
            'source': 'simulation'
        })

        df_sim = build_features(df_sim, cfg)
        df_sim = df_sim.fillna(0)

        waste_prob = model.predict_proba(df_sim[FEATURES])[:, 1]

        summary.append({
            'user_type': user_type,
            'waste_rate': round((waste_prob > 0.5).mean(), 3),
            'avg_waste_prob': round(waste_prob.mean(), 3),
            'reclaimable_gb': round(
                df_sim[waste_prob > 0.5]['size_mb'].sum() / 1000, 2
            ),
            'total_co2_g': round(df_sim['co2_estimate_g'].sum(), 4)
        })

    return pd.DataFrame(summary)