import os
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime

def scan_local(root: str) -> pd.DataFrame:
    rows = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            path = os.path.join(dirpath, f)
            try:
                s = os.stat(path)
                rows.append({
                    "file_id": hashlib.sha256(path.encode()).hexdigest()[:16],
                    "extension": os.path.splitext(f)[1].lower(),
                    "size_bytes": s.st_size,
                    "last_access": datetime.fromtimestamp(s.st_atime),
                    "created": datetime.fromtimestamp(s.st_ctime),
                    "source": "local"
                })
            except:
                pass
    return pd.DataFrame(rows)

def generate_synthetic(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    sizes = rng.lognormal(mean=14.5, sigma=2.8, size=n).astype(int)
    access = rng.exponential(scale=120, size=n)
    age = rng.exponential(scale=365, size=n)
    exts = rng.choice(
        ['.pdf','.jpg','.mp4','.docx','.zip','.py','.csv','.txt','.png','.xlsx'],
        p=[.15,.18,.10,.12,.08,.07,.08,.10,.07,.05],
        size=n
    )
    return pd.DataFrame({
        "file_id": [f"syn_{i:06d}" for i in range(n)],
        "extension": exts,
        "size_bytes": sizes,
        "days_since_access": access,
        "age_days": age,
        "source": "synthetic"
    })