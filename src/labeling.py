import numpy as np
import pandas as pd

def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # 3 bağımsız koşul
    c1 = df['days_since_access'] > 180
    c2 = df['access_ratio'] > 0.9
    c3 = df['ext_category'].isin(['archive', 'media'])
    
    # En az 2 koşul sağlanıyorsa waste=1
    score = c1.astype(int) + c2.astype(int) + c3.astype(int)
    df['label'] = (score >= 2).astype(int)
    df['label_confidence'] = score / 3
    
    return df

def baseline_predict(df: pd.DataFrame) -> np.ndarray:
    # Sadece yaş kuralı — karşılaştırma için
    return (df['days_since_access'] > 180).astype(int).values