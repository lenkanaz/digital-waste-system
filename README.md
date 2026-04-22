# Digital Waste System

My laptop kept running out of storage. I didn't know which files were useful and which weren't — so I built a system to figure that out automatically, and along the way started thinking about the environmental cost of storing data we never use.

## What This Does

Three-source data pipeline (local filesystem + synthetic generation + Unsplash open data) feeding into an ML classification system that labels files as waste or not-waste, estimates CO₂ storage cost, and simulates behavior across different user types.

## Data

| Source | Size | Notes |
|---|---|---|
| Local files | ~466 rows | SHA-256 anonymized paths |
| Synthetic | 5,000 rows | Log-normal size, exponential age |
| Unsplash API | 30 rows | Real photo metadata |

Total: 5,766 rows

## Features

- `access_ratio` — how often a file is accessed relative to its age
- `entropy_score` — information density of file content  
- `co2_estimate_g` — monthly CO₂ cost of storage
- `norm_age`, `norm_size` — user-normalized features

## Results

| Model | F1 | ROC-AUC |
|---|---|---|
| Baseline (age > 180d) | 0.753 | 0.848 |
| Logistic Regression | 0.709 | 0.959 |
| Random Forest | 0.748 | **0.970** |

## Simulation

Tested the model against three synthetic user profiles:

| User Type | Waste Rate | Reclaimable |
|---|---|---|
| Messy hoarder | 44.8% | 24 GB |
| Active user | 1.1% | 0.02 GB |
| Heavy media | 36.8% | 46 GB |

The heavy media result was unexpected — I assumed the hoarder profile would produce more reclaimable storage, but large rarely-accessed video files turned out to have a bigger footprint.

## Things That Went Wrong

- **Python 3.14 broke everything.** pip wouldn't install any packages. Downgraded to 3.11, fixed it.
- **Label leakage.** My first instinct was to label files as waste if age > 180 days, then use the same rule as baseline. That would have made the whole comparison circular. Switched to multi-criteria labeling instead.
- **KeyboardInterrupt in VS Code terminal.** Long one-liner commands kept getting cut off. Solved by writing actual script files instead.

## Limitations

- Labels are heuristic-generated, not human-annotated. A model trained this way may reflect the labeling logic more than real user behavior.
- CO₂ formula uses published averages (IEA Turkey 2022, Uptime Institute 2023) — not measured from actual infrastructure.
- Synthetic data is statistically realistic but not real.

## How to Run

```bash
git clone https://github.com/lenkanaz/digital-waste-system
cd digital-waste-system
py -3.11 -m pip install -r requirements.txt
py -3.11 -m pytest tests/ -v
```

## References

- Green Software Foundation (2023) — Storage Energy Intensity
- IEA Turkey Grid Carbon Intensity (2022) — 0.475 kgCO₂/kWh
- Uptime Institute Global PUE Report (2023) — avg PUE 1.58