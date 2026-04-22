import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def run_validation(df_local, df_synthetic):
    # KS-test
    ks_stat, p_value = stats.ks_2samp(
        df_local['size_bytes'].dropna(),
        df_synthetic['size_bytes'].dropna()
    )
    print(f"KS statistic: {ks_stat:.3f}")
    print(f"p-value: {p_value:.3f}")
    if p_value > 0.05:
        print("Sonuç: Dağılımlar anlamlı derecede farklı değil — sentetik veri gerçekçi.")
    else:
        print("Sonuç: Dağılımlar farklı.")

    # Histogram karşılaştırması
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(np.log10(df_local['size_bytes'] + 1), bins=40, alpha=0.6, label='Local (real)', color='steelblue')
    ax.hist(np.log10(df_synthetic['size_bytes'] + 1), bins=40, alpha=0.6, label='Synthetic', color='coral')
    ax.set_xlabel('File size (log10 bytes)')
    ax.set_ylabel('Count')
    ax.set_title('Real vs Synthetic — File Size Distribution')
    ax.legend()
    plt.tight_layout()
    plt.savefig('outputs/figures/validation.png', dpi=150)
    plt.close()
    print("Validation grafiği kaydedildi.")

if __name__ == "__main__":
    df = pd.read_csv('data/processed/merged.csv')
    df_local = df[df['source'] == 'local']
    df_synthetic = df[df['source'] == 'synthetic']
    run_validation(df_local, df_synthetic)