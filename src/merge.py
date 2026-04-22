import pandas as pd
import yaml
from src.ingestion import scan_local, generate_synthetic, fetch_unsplash

def load_config():
    with open("config/settings.yaml") as f:
        return yaml.safe_load(f)

def merge_all():
    cfg = load_config()
    
    print("Local dosyalar taranıyor...")
    local_df = scan_local("C:/Users/zehra/Downloads")
    
    print("Synthetic veri üretiliyor...")
    synth_df = generate_synthetic(
        cfg['data']['synthetic_n'],
        cfg['data']['random_seed']
    )
    
    print("Unsplash verisi çekiliyor...")
    unsplash_df = fetch_unsplash(
        cfg['unsplash']['client_id'],
        cfg['unsplash']['n_photos']
    )
    
    print("Birleştiriliyor...")
    df = pd.concat([local_df, synth_df, unsplash_df], ignore_index=True)
    df.to_csv("data/processed/merged.csv", index=False)
    print(f"Toplam {len(df)} satır kaydedildi.")
    return df

if __name__ == "__main__":
    merge_all()