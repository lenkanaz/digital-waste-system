import os
import hashlib
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