# Lanka Flood Forecast 🌊🇱🇰

Forecast river water levels at 39 Sri Lankan gauging stations 6, 12 and 24 hours ahead, and warn before a river reaches its official alert or flood level.

**🗺️ Live map: https://nipuna-lakruwan.github.io/lanka-flood-forecast/** (refreshed every 3 hours by GitHub Actions) · **🤗 Models: https://huggingface.co/nipuna02/lanka-flood-forecast**

Built on open data scraped by [Nuwan I. Senaratna](https://github.com/nuuuwan) (MIT):
- [nuuuwan/lk_irrigation](https://github.com/nuuuwan/lk_irrigation): Irrigation Department river levels, roughly hourly, with official alert, minor-flood and major-flood levels per station.
- [nuuuwan/lk_weather_3h](https://github.com/nuuuwan/lk_weather_3h): Department of Meteorology 3-hourly rainfall from 25 stations.

> **Status: research prototype.** It's evaluated on held-out data below, but it is **not an official warning service**. Always follow the Disaster Management Centre and the Irrigation Department.

## Data

| | |
|---|---|
| River readings | 318,789, from 39 stations in 18 basins |
| Period | 23 Nov 2025 to 17 Nov 2026, including Cyclone Ditwah's floods |
| Hourly table | 8,633 hours × 39 stations, 3.9% missing after filling gaps of 3 hours or less |
| Cleaning | 133 one-hour glitches removed: a reading far from both neighbours while the neighbours agree, e.g. 5.3 m → 2.0 m → 5.3 m. Real floods rise over hours and aren't affected |
| Rainfall | 39,327 reports from 25 stations, from 5 Dec 2025. Every river station has a gauge within 80 km |

## Method

One gradient-boosting model (LightGBM) per horizon is shared by all stations. It predicts the change in level, scaled by each station's typical 6-hour change, so every station learns from every other station's floods.

Features available at forecast time:
- the station's recent changes (1–24 h);
- its distance below the flood level;
- the rise at **upstream stations on the same river**, using the river's upstream-to-downstream station order;
- the basin average;
- a country-wide signal: the share of stations rising fast right now;
- time of day;
- optionally, rainfall over the last 3–24 h.

Day of year is deliberately left out, because with one year of data it would memorise dates.

**Warnings** use separate 80th- and 90th-percentile models, which answer "how high could it plausibly get?". A warning is raised when that estimate reaches the threshold while the river is still below it.

## Evaluation

Two test blocks, each with a 48-hour buffer from the training data:
- **recent** (15 Sep – 17 Nov 2026): trained only on earlier data. **This is the honest forward-in-time result.**
- **ditwah** (23 Nov – 31 Dec 2025): trained on the later months. It's an event study of the cyclone, not a real-time setup.

### Water-level error (mean absolute error, cm)

| Horizon | Recent: no-change baseline | Recent: model | Skill | Ditwah: no-change baseline | Ditwah: model | Skill |
|---|---|---|---|---|---|---|
| 6 h | 20.5 | **15.9** | +22% | 17.5 | **13.9** | +20% |
| 12 h | 30.8 | **26.4** | +14% | 26.5 | **22.1** | +17% |
| 24 h | 37.7 | **35.0** | +7% | 39.1 | **34.8** | +11% |

A damped trend baseline is worse than "no change" beyond 6 hours.

### Flood warnings, 12 hours ahead, recent test block (37 minor-flood crossings)

| Method | Warned in time | False-alarm hours | Median lead |
|---|---|---|---|
| Trend baseline | 33 / 37 | 556 | 5 h |
| Model (central forecast) | 24 / 37 | 73 | 4 h |
| Model, 80th percentile | 27 / 37 | 405 | 9 h |
| Model, 90th percentile | **33 / 37** | 569 | **8 h** |

The 90th-percentile model catches as many floods as the trend baseline with the same false alarms, and warns **3 hours earlier** on median. The 80th-percentile model trades a few missed floods for a third fewer false alarms. The full tables, including alert-level warnings and the Ditwah block, are in [`results/metrics.json`](results/metrics.json).

### What didn't help: rainfall
Adding rainfall from the 25 Met Department stations left accuracy and warnings essentially unchanged (12 h error 26.35 vs 26.36 cm). Rain reports are only used from the hour they are published, so there is no look-ahead. Likely reasons:
- 25 gauges reporting every 3 hours are too coarse for local downpours;
- upstream river stations and the country-wide rising signal already carry the same information earlier.

Radar or satellite rainfall would be the next thing to try.

### Known limits
- **One year of data**, so each monsoon season appears once. Extreme events like Ditwah rise faster than anything else in the training data, and detection there is lower (23 of 42 floods with the 90th-percentile model).
- Thresholds come from the Irrigation Department's published levels. One station has none.
- Warnings are scored per station crossing. A crossing that happens in under an hour cannot be warned in time by design.

## Run it

```bash
uv sync
uv run python src/data.py        # download and build the hourly table
git clone --depth 1 https://github.com/nuuuwan/lk_weather_3h data/raw/lk_weather_3h   # rainfall (optional)
uv run python src/evaluate.py    # train, test, write results/metrics.json
```

## Live system

- `.github/workflows/forecast.yml`: every 3 hours, downloads the latest readings, runs `src/live.py forecast`, and commits `docs/forecast.json`. GitHub Pages serves `docs/` as the map.
- `.github/workflows/retrain.yml`: on the 1st of each month, retrains on all data collected so far.
- Models are in `models/`: LightGBM text files for the 6, 12 and 24 h forecasts, plus the 80th- and 90th-percentile 12 h warning models.

## Credits

River and weather data: Sri Lanka Irrigation Department and Department of Meteorology, scraped and published by [@nuuuwan](https://github.com/nuuuwan) under the MIT license. This project: MIT.
