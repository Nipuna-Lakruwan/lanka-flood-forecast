"""Live forecasts for the map.

    python src/live.py train      # fit the final models on all data and save them to models/
    python src/live.py forecast   # download the latest readings and write docs/forecast.json

The final models use river features only, because rainfall did not improve accuracy (see README).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from data import ROOT, build
from evaluate import WARN_QUANTILES, fit_lgb
from features import FEATURES, HORIZONS, build_table

MODELS = ROOT / "models"
DOCS = ROOT / "docs"
WARN_HORIZON = 12
WARN_QUANTILE = 0.9
HISTORY_HOURS = 72


def model_path(h: int, q: float | None = None):
    return MODELS / (f"lgb_{h}h.txt" if q is None else f"lgb_{h}h_q{int(q * 100)}.txt")


def train() -> None:
    levels, stations = build(force_download=True)
    table = build_table(levels, stations, with_rain=False)
    MODELS.mkdir(exist_ok=True)
    for h in HORIZONS:
        fit_lgb(table, h).booster_.save_model(model_path(h))
    for q in WARN_QUANTILES:
        fit_lgb(table, WARN_HORIZON, quantile=q).booster_.save_model(model_path(WARN_HORIZON, q))
    meta = {
        "trained_at": datetime.now(UTC).isoformat(timespec="minutes"),
        "rows": len(table),
        "data_from": str(levels.index.min()),
        "data_to": str(levels.index.max()),
    }
    (MODELS / "meta.json").write_text(json.dumps(meta, indent=2))
    print("saved models", meta)


def status(level: float, row: pd.Series) -> str:
    for name, col in (
        ("major", "major_flood_level_m"),
        ("minor", "minor_flood_level_m"),
        ("alert", "alert_level_m"),
    ):
        thr = row.get(col)
        if pd.notna(thr) and level >= thr:
            return name
    return "normal"


def forecast() -> None:
    levels, stations = build(force_download=True)
    table = build_table(levels, stations, with_rain=False)
    boosters = {h: lgb.Booster(model_file=str(model_path(h))) for h in HORIZONS}
    warn = lgb.Booster(model_file=str(model_path(WARN_HORIZON, WARN_QUANTILE)))

    out = []
    for name, g in table.groupby("station", observed=True):
        g = g.sort_values("time")
        last = g.iloc[-1]
        age_h = (levels.index.max() - last.time).total_seconds() / 3600
        info = stations.loc[name]
        x = g[FEATURES].iloc[[-1]]
        fc = {f"{h}h": round(float(last.level + boosters[h].predict(x)[0] * last.scale), 2) for h in HORIZONS}
        high12 = round(float(last.level + warn.predict(x)[0] * last.scale), 2)
        thresholds = {
            k: (None if pd.isna(info[c]) else float(info[c]))
            for k, c in (
                ("alert", "alert_level_m"),
                ("minor", "minor_flood_level_m"),
                ("major", "major_flood_level_m"),
            )
        }
        warning = None
        for lvl in ("major", "minor", "alert"):
            thr = thresholds[lvl]
            if thr is not None and last.level < thr <= high12:
                warning = lvl
                break
        hist = levels[name].iloc[-HISTORY_HOURS:]
        out.append(
            {
                "station": name,
                "river": info.river_name,
                "basin": info.basin_name,
                "lat": float(info.lat),
                "lng": float(info.lng),
                "time": last.time.isoformat(),
                "hours_since_reading": round(age_h, 1),
                "level_m": round(float(last.level), 2),
                "status": status(last.level, info),
                "forecast_m": fc,
                "possible_high_12h_m": high12,
                "warning_12h": warning,
                "thresholds_m": thresholds,
                "history": [
                    [t.isoformat(), None if np.isnan(v) else round(float(v), 2)] for t, v in hist.items()
                ],
            }
        )
    DOCS.mkdir(exist_ok=True)
    payload = {
        "generated_at": datetime.now(UTC).isoformat(timespec="minutes"),
        "latest_reading": str(levels.index.max()),
        "model": json.loads((MODELS / "meta.json").read_text()),
        "stations": sorted(out, key=lambda s: s["station"]),
    }
    (DOCS / "forecast.json").write_text(json.dumps(payload, separators=(",", ":")))
    n_warn = sum(1 for s in out if s["warning_12h"])
    print(f"wrote forecast for {len(out)} stations, {n_warn} with a 12h warning")


if __name__ == "__main__":
    {"train": train, "forecast": forecast}[sys.argv[1]]()
