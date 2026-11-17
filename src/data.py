"""Download and tidy river water levels from nuuuwan/lk_irrigation (MIT).

The Irrigation Department publishes readings roughly hourly. This module turns the raw readings into one
hourly table per station, filling only short gaps so that missing data is never invented over long outages.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
BASE = "https://raw.githubusercontent.com/nuuuwan/lk_irrigation/main/data"
STATIC = ["stations", "rivers", "basins", "locations"]

# Interpolate across gaps of at most this many hours; longer outages stay missing.
MAX_FILL_HOURS = 3


def download(force: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    targets = {"rwl_all.json": f"{BASE}/all.json"}
    targets |= {f"{name}.json": f"{BASE}/static/{name}.json" for name in STATIC}
    for filename, url in targets.items():
        path = RAW / filename
        if path.exists() and not force:
            continue
        r = requests.get(url, timeout=600)
        r.raise_for_status()
        path.write_bytes(r.content)
        print(f"downloaded {filename} ({len(r.content) / 1e6:.1f} MB)")


def load_stations() -> pd.DataFrame:
    stations = pd.DataFrame(json.loads((RAW / "stations.json").read_text()))
    rivers = pd.DataFrame(json.loads((RAW / "rivers.json").read_text()))
    stations = stations.merge(
        rivers[["name", "basin_name"]].rename(columns={"name": "river_name"}), on="river_name", how="left"
    )
    stations[["lat", "lng"]] = pd.DataFrame(stations.lat_lng.tolist(), index=stations.index)
    # A level of 0 means the department has not published a threshold for that station.
    for col in ["alert_level_m", "minor_flood_level_m", "major_flood_level_m"]:
        stations[col] = stations[col].replace(0.0, np.nan)
    return stations.drop(columns=["lat_lng"]).set_index("name")


def load_readings() -> pd.DataFrame:
    readings = pd.DataFrame(json.loads((RAW / "rwl_all.json").read_text()))
    readings["time"] = pd.to_datetime(readings.time_ut, unit="s", utc=True)
    return readings.drop(columns=["time_ut"]).rename(columns={"station_name": "station"})


def hourly_levels(readings: pd.DataFrame) -> pd.DataFrame:
    """Wide table: one row per UTC hour, one column per station."""
    readings = readings.assign(hour=readings.time.dt.floor("h"))
    wide = readings.pivot_table(index="hour", columns="station", values="water_level_m", aggfunc="mean")
    full = pd.date_range(wide.index.min(), wide.index.max(), freq="h", tz="UTC")
    wide = wide.reindex(full)
    wide.index.name = "time"
    wide = remove_spikes(wide)
    return wide.interpolate(limit=MAX_FILL_HOURS, limit_area="inside")


def remove_spikes(wide: pd.DataFrame, jump_m: float = 1.0, agree_m: float = 0.3) -> pd.DataFrame:
    """Drop one-hour glitches: a reading far from both neighbours while the neighbours agree with each other.

    A real flood rises over hours and does not fall straight back, so it is not affected.
    """
    prev, nxt = wide.shift(1), wide.shift(-1)
    spike = ((wide - prev).abs() > jump_m) & ((wide - nxt).abs() > jump_m) & ((prev - nxt).abs() < agree_m)
    remove_spikes.last_count = int(spike.sum().sum())
    return wide.mask(spike)


def build(force_download: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    download(force_download)
    stations = load_stations()
    levels = hourly_levels(load_readings())
    PROCESSED.mkdir(parents=True, exist_ok=True)
    levels.to_parquet(PROCESSED / "levels_hourly.parquet")
    stations.to_parquet(PROCESSED / "stations.parquet")
    return levels, stations


if __name__ == "__main__":
    levels, stations = build()
    print(f"hourly table: {levels.shape[0]:,} hours x {levels.shape[1]} stations")
    print(f"{levels.index.min()} -> {levels.index.max()}")
    print(f"one-hour glitches removed: {remove_spikes.last_count}")
    print(f"missing after short-gap fill: {levels.isna().mean().mean():.1%}")
    print(f"stations with flood thresholds: {stations.minor_flood_level_m.notna().sum()} of {len(stations)}")
