---
license: mit
library_name: lightgbm
tags:
  - time-series-forecasting
  - flood-forecasting
  - hydrology
  - sri-lanka
  - early-warning
  - tabular-regression
pipeline_tag: tabular-regression
---

# Lanka Flood Forecast: LightGBM river-level models

Forecasts water levels at 39 Sri Lankan Irrigation Department gauging stations **6, 12 and 24 hours ahead**. Two upper-quantile models (80th and 90th percentile, 12 h) give early flood warnings.

- **Code, evaluation and features:** https://github.com/Nipuna-Lakruwan/lanka-flood-forecast
- **Live map (refreshed every 3 hours):** https://nipuna-lakruwan.github.io/lanka-flood-forecast/
- **Data:** Sri Lanka Irrigation Department river levels via [nuuuwan/lk_irrigation](https://github.com/nuuuwan/lk_irrigation) (MIT), 23 Nov 2025 – 17 Nov 2026, 318,789 readings

> **Research prototype, not an official warning service.** Follow the Disaster Management Centre and the Irrigation Department.

## Files

| File | Predicts |
|---|---|
| `lgb_6h.txt`, `lgb_12h.txt`, `lgb_24h.txt` | change in level 6/12/24 h ahead, divided by the station's median absolute 6-hour change |
| `lgb_12h_q80.txt`, `lgb_12h_q90.txt` | 80th/90th-percentile change 12 h ahead, used for warnings |

All are LightGBM text models (`lightgbm.Booster(model_file=...)`). Features are built by `src/features.py` in the GitHub repo:
- recent changes;
- distance to the flood level;
- upstream stations on the same river;
- basin average;
- the national share of fast-rising stations;
- time of day.

## Results on held-out time blocks

| Horizon | No-change baseline MAE | Model MAE | Skill |
|---|---|---|---|
| 6 h | 20.5 cm | 15.9 cm | +22% |
| 12 h | 30.8 cm | 26.4 cm | +14% |
| 24 h | 37.7 cm | 35.0 cm | +7% |

These are from the recent block, 15 Sep – 17 Nov 2026, trained only on earlier data.

**12 h minor-flood warnings** (recent block, 37 crossings): the 90th-percentile model warned for **33 / 37**, with a median lead of **8 h**. The trend baseline caught the same number with 5 h lead. During Cyclone Ditwah, an event study, it caught 23 / 42.

**Limitations:**
- one year of data;
- extreme, fast-rising events are under-represented;
- rainfall from 25 gauges did not improve accuracy.
