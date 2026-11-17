"""Rainfall features from nuuuwan/lk_weather_3h (Department of Meteorology 3-hourly reports, MIT).

Each report gives the rain in the 3 hours before it. For every river station we use the two nearest weather
stations (within 80 km) and sum their recent rain. Data starts on 5 Dec 2025; earlier hours have no rain
features, which the model treats as missing.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from data import RAW

WEATHER = RAW / "lk_weather_3h" / "data"
WINDOWS = (3, 6, 12, 24)


def load_reports() -> pd.DataFrame:
    rows = []
    for path in (WEATHER / "wr").rglob("*.json"):
        r = json.loads(path.read_text())
        if r.get("rain_mm") is None:
            continue
        rows.append((r["station_name"], r["time_ut"], float(r["rain_mm"])))
    df = pd.DataFrame(rows, columns=["wstation", "time_ut", "rain_mm"])
    df["time"] = pd.to_datetime(df.time_ut, unit="s", utc=True).dt.floor("h")
    # Some stations report a trace or missing as negative numbers; treat those as zero rain.
    df["rain_mm"] = df.rain_mm.clip(lower=0)
    return df.drop_duplicates(["wstation", "time"]).drop(columns="time_ut")


def hourly_rain(reports: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Rain per hour per weather station: each 3-hour total is spread evenly over its 3 hours."""
    wide = reports.pivot_table(index="time", columns="wstation", values="rain_mm", aggfunc="max")
    per_hour = wide / 3.0
    spread = per_hour.reindex(index)
    # The report at time t covers t-3h .. t but is only known from t on, so carry it forwards (never backwards,
    # which would leak rain into hours before it was reported).
    return spread.ffill(limit=2)


def _km(a, b) -> float:
    (la1, lo1), (la2, lo2) = a, b
    p = math.pi / 180
    h = (
        0.5
        - math.cos((la2 - la1) * p) / 2
        + math.cos(la1 * p) * math.cos(la2 * p) * (1 - math.cos((lo2 - lo1) * p)) / 2
    )
    return 12742 * math.asin(math.sqrt(h))


def nearest_weather(
    stations: pd.DataFrame, rain: pd.DataFrame, k: int = 2, max_km: float = 80
) -> dict[str, list]:
    places = {p["name"]: p["lat_lng"] for p in json.loads((WEATHER / "static" / "places.json").read_text())}
    known = {w: places[w] for w in rain.columns if w in places}
    out = {}
    for s, row in stations.iterrows():
        d = sorted((_km((row.lat, row.lng), ll), w) for w, ll in known.items())
        out[s] = [w for km, w in d[:k] if km <= max_km]
    return out


def rain_features(levels_index: pd.DatetimeIndex, stations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rain = hourly_rain(load_reports(), levels_index)
    near = nearest_weather(stations, rain)
    national24 = rain.rolling(24, min_periods=12).sum().max(axis=1)
    feats = {}
    for s, ws in near.items():
        f = pd.DataFrame(index=levels_index)
        local = rain[ws].mean(axis=1) if ws else pd.Series(np.nan, index=levels_index)
        for w in WINDOWS:
            f[f"rain_{w}h"] = local.rolling(w, min_periods=max(1, w // 2)).sum()
        f["rain_national_max24"] = national24
        feats[s] = f
    return feats


def coverage_report(stations: pd.DataFrame, levels_index: pd.DatetimeIndex) -> str:
    reports = load_reports()
    rain = hourly_rain(reports, levels_index)
    near = nearest_weather(stations, rain)
    no_rain = [s for s, w in near.items() if not w]
    return (
        f"weather stations with rain: {reports.wstation.nunique()}, reports: {len(reports):,}, "
        f"from {reports.time.min()} to {reports.time.max()}; river stations without a weather station "
        f"within 80 km: {len(no_rain)} {no_rain}"
    )


if __name__ == "__main__":
    from data import PROCESSED

    levels = pd.read_parquet(PROCESSED / "levels_hourly.parquet")
    stations = pd.read_parquet(PROCESSED / "stations.parquet")
    print(coverage_report(stations.loc[stations.index.intersection(levels.columns)], levels.index))
    Path(PROCESSED).mkdir(exist_ok=True)
