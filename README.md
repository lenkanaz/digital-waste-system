# Digital Waste System

My laptop kept running out of storage. I didn't know which files were useful and which weren't — so I built a system to figure that out automatically, and along the way started thinking about the environmental cost of storing data we never use.

## What This Does

Three-source data pipeline (local filesystem + synthetic generation + Unsplash open data) feeding into an ML classification system that labels files as waste or not-waste, estimates CO₂ storage cost, and simulates behavior across different user types.

## Architecture

```
Local Scanner → SHA-256 Anonymization     ↘
Synthetic Generator (100K files)          → Merge → Feature Engineering → Labeling → ML Models → Evaluation
Unsplash API (open data, real metadata)   ↗                                               ↓
                                                                               Simulation + Hypothesis Tests
```

## Data

| Source | Size | Notes |
|---|---|---|
| Local files | ~736 rows | SHA-256 anonymized — no personal data stored |
| Synthetic | 100,000 rows | Log-normal size, exponential age distribution |
| Unsplash API | 30 rows | Real photo metadata for file diversity |

**Total: 100,766 rows**

## Features Engineered

| Feature | Description | Why |
|---|---|---|
| `access_ratio` | days_since_access / age_days | Normalized access frequency |
| `co2_estimate_g` | size × energy × carbon intensity × PUE | Environmental cost of storage |
| `norm_age`, `norm_size` | User-normalized age and size | Context-aware features |
| `ext_category` | media / doc / archive / code / other | File type behavioral signal |

## Labeling Strategy

To avoid label leakage, waste labels use **three independent criteria** — not just age:

1. `days_since_access > 180`
2. `access_ratio > 0.9`  
3. `ext_category` in [archive, media]

A file is labeled waste if **at least 2 of 3** conditions are met. The baseline uses only rule 1 — making the ML comparison meaningful.

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Baseline (age > 180d) | 0.748 | 0.758 | 0.753 | 0.848 |
| Logistic Regression | 0.759 | 0.665 | 0.709 | 0.959 |
| **Random Forest** | **0.780** | **0.718** | **0.748** | **0.970** |

![ROC Curve](outputs/figures/roc_curve.png)

![Feature Importance](outputs/figures/feature_importance.png)

## Simulation

Tested the model against three synthetic user profiles (1,000 files each):

| User Type | Waste Rate | Reclaimable | CO₂ (g/month) |
|---|---|---|---|
| Messy hoarder | 44.8% | 24 GB | 33.2 |
| Active user | 1.1% | 0.02 GB | 1.7 |
| Heavy media | 36.8% | 46 GB | 90.3 |

The heavy media result was unexpected — I assumed the hoarder profile would produce more reclaimable storage, but large rarely-accessed video files turned out to have a bigger footprint.

![Data Exploration](outputs/figures/exploration.png)

## Hypothesis Tests

- **H2** — RF > Baseline: Supported — ROC-AUC 0.970 vs 0.848
- **H3** — Waste rate differs by user type: Supported — ANOVA p=0.0000, messy hoarder 44.8% vs active user 1.1%

## K-Means Clustering

Files clustered into 3 segments independently of the classifier:

| Cluster | Profile | Waste Rate |
|---|---|---|
| 0 | Small, old files | 19.6% |
| 1 | Very large files | 25.0% |
| 2 | Medium, high access ratio | 32.0% |

## Things That Went Wrong

- **Python 3.14 broke everything.** pip wouldn't install any packages. Downgraded to 3.11.
- **Label leakage.** My first instinct was age > 180 days for both labeling and baseline — that would have made the comparison circular. Switched to multi-criteria labeling.
- **KeyboardInterrupt in VS Code terminal.** Long one-liner commands kept getting cut off. Solved by writing script files instead.
- **KS-test p=0.000.** Synthetic and local distributions differ — local scan includes many small system files. Expected, but worth noting.

## Limitations

- Labels are heuristic-generated, not human-annotated. The model may reflect the labeling logic more than real user behavior.
- CO₂ formula uses published averages — not measured from actual infrastructure.
- Synthetic data is statistically realistic but not real.
- Local scan from one machine — may not generalize across different OS or usage patterns.

## How to Run

```bash
git clone https://github.com/lenkanaz/digital-waste-system
cd digital-waste-system
py -3.11 -m pip install -r requirements.txt
py -3.11 -m pytest tests/ -v
py -3.11 -m main
```

## Project Structure

```
digital-waste-system/
├── src/
│   ├── ingestion.py      # 3-source data collection
│   ├── features.py       # feature engineering + CO2
│   ├── labeling.py       # multi-criteria labeling
│   ├── models.py         # LR, RF, K-Means
│   ├── evaluation.py     # metrics + visualizations
│   ├── simulation.py     # user type simulation
│   ├── validation.py     # KS-test + distribution check
│   └── hypothesis.py     # statistical hypothesis tests
├── notebooks/
│   └── 01_exploration.ipynb
├── outputs/
│   ├── figures/          # all plots saved here
│   ├── models/           # saved model files
│   └── reports/          # CSV results
├── tests/
│   └── test_features.py  # 3 unit tests, all passing
├── config/
│   └── settings.yaml     # all parameters here, zero hardcoding
├── main.py               # full pipeline, single command
└── README.md
```

## References

- Green Software Foundation (2023) — Storage Energy Intensity
- IEA Turkey Grid Carbon Intensity (2022) — 0.475 kgCO₂/kWh
- Uptime Institute Global PUE Report (2023) — avg PUE 1.58