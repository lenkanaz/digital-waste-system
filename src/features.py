import numpy as np
import pandas as pd
import zlib

def compute_co2(size_bytes: float, cfg: dict) -> float:
    gb = size_bytes / 1e9
    return gb * cfg['co2']['kwh_per_gb'] * cfg['co2']['carbon_tr'] * 1000 * cfg['co2']['pue']

def compute_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = np.bincount(np.frombuffer(data, dtype=np.uint8), minlength=256)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -(probs * np.log2(probs)).sum()

def categorize_ext(ext: str) -> str:
    media = {'.jpg', '.png', '.mp4', '.mp3', '.avi', '.gif'}
    docs  = {'.pdf', '.docx', '.pptx', '.xlsx', '.txt'}
    arch  = {'.zip', '.tar', '.gz', '.rar'}
    code  = {'.py', '.js', '.ipynb', '.csv'}
    if ext in media: return 'media'
    if ext in docs:  return 'document'
    if ext in arch:  return 'archive'
    if ext in code:  return 'code'
    return 'other'

def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    
    df['size_mb'] = df['size_bytes'] / 1e6
    df['ext_category'] = df['extension'].map(categorize_ext)
    df['access_ratio'] = (
        df['days_since_access'] / (df['age_days'] + 1)
    ).clip(0, 1)
    df['co2_estimate_g'] = df['size_bytes'].apply(
        lambda x: compute_co2(x, cfg)
    )
    df['norm_age']  = df['age_days'] / (df['age_days'].mean() + 1e-9)
    df['norm_size'] = df['size_mb']  / (df['size_mb'].mean()  + 1e-9)
    
    return df