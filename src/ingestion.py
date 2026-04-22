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
                    "days_since_access": (datetime.now() - datetime.fromtimestamp(s.st_atime)).days,
                    "age_days": (datetime.now() - datetime.fromtimestamp(s.st_ctime)).days,
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
def fetch_unsplash(client_id: str, n: int = 300) -> pd.DataFrame:
    import requests
    rows, page = [], 1
    while len(rows) < n:
        resp = requests.get(
            "https://api.unsplash.com/photos",
            params={"per_page": 30, "page": page},
            headers={"Authorization": f"Client-ID {client_id}"}
        )
        if resp.status_code != 200:
            break
        for p in resp.json():
            rows.append({
                "file_id": f"usp_{p['id']}",
                "extension": ".jpg",
                "size_bytes": p['width'] * p['height'] * 3,
                "days_since_access": 0,
                "age_days": (datetime.now() - datetime.fromisoformat(p['created_at'][:10])).days,
                "source": "open_data"
            })
        page += 1
    return pd.DataFrame(rows[:n])